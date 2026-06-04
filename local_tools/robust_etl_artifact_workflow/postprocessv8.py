# ==========================================
# 🤖 POSTPROCESSOR (CHAT INTERFACE) - V7 (Cloud Storage Support)
# ==========================================
# What's new in V7:
#   - Cloud storage support: read Parquet from S3, GCS, or Cloudflare R2
#   - DuckDB httpfs extension configured securely (only for user-specified bucket)
#   - Cloud credentials via Streamlit secrets or manual input
#   - HTTP/HTTPS URLs blocked in SQL validator (prevents arbitrary URL access)
#   - All V6 features preserved:
#     Blocked file-reading functions, DuckDB extension lockdown, realpath paths,
#     stored relationships, cache TTL, EdTech prompts, PII redaction,
#     starter questions, conversational context, multi-LLM, auto-retry,
#     auto-viz, chat export, table inventory
# ==========================================

import os

# --- Environment variables MUST be set before importing libraries ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st

# --- SQLITE FIX FOR STREAMLIT CLOUD ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# --------------------------------------

import duckdb
import chromadb
import json
from openai import OpenAI
import datetime
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="🤖 Data Chat V8", layout="wide")

# ==========================================
# CONSTANTS
# ==========================================

HARD_ROW_LIMIT = 1000
MAX_ROWS_FOR_SUMMARY = 50

PROVIDER_CONFIG = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "secret_key": "OPENAI_API_KEY",
    },
    "xAI (Grok)": {
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-2-latest",
        "secret_key": "XAI_API_KEY",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "secret_key": "DEEPSEEK_API_KEY",
    },
}

DANGEROUS_KEYWORDS = [
    'COPY', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'INSTALL', 'LOAD', 'ATTACH', 'DETACH', 'EXPORT', 'IMPORT', 'CALL',
    'EXECUTE', 'SET', 'PRAGMA', 'HTTPFS', 'VACUUM', 'CHECKPOINT',
    'GRANT', 'REVOKE', 'TRUNCATE', 'MERGE', 'PREPARE',
    'READ_CSV', 'READ_CSV_AUTO', 'READ_TEXT', 'READ_JSON', 'READ_JSON_AUTO',
    'READ_BLOB', 'READ_PARQUET_SCHEMA', 'GLOB', 'PARQUET_SCAN',
]

PII_COLUMN_PATTERNS = [
    r'(?i)\bname\b', r'(?i)\bfirst.?name\b', r'(?i)\blast.?name\b',
    r'(?i)\bfull.?name\b', r'(?i)\bemail\b', r'(?i)\bphone\b',
    r'(?i)\baddress\b', r'(?i)\bssn\b', r'(?i)social.?sec',
    r'(?i)\bdob\b', r'(?i)birth.?date', r'(?i)date.?of.?birth',
    r'(?i)\bpassword\b', r'(?i)\blogin\b', r'(?i)\busername\b',
    r'(?i)\bstudent.?id\b', r'(?i)\bemployee.?id\b',
    r'(?i)\bparent.?name\b', r'(?i)\bguardian\b',
    r'(?i)\bcontact\b', r'(?i)\bpersonal\b',
]

# ==========================================
# SESSION STATE
# ==========================================

if "data_active" not in st.session_state:
    st.session_state.data_active = False
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Connect to a dataset to get started."}
    ]
if "active_path" not in st.session_state:
    st.session_state.active_path = "deploy_artifacts"
if "starter_questions" not in st.session_state:
    st.session_state.starter_questions = []
if "pii_redaction" not in st.session_state:
    st.session_state.pii_redaction = True
# V7: Cloud storage configuration (None = local mode)
if "cloud_config" not in st.session_state:
    st.session_state.cloud_config = None


# ==========================================
# HELPERS
# ==========================================

def _format_chat_export():
    """Format chat history as Markdown for download."""
    lines = []
    for msg in st.session_state.messages:
        role = "🧑 User" if msg["role"] == "user" else "🤖 Assistant"
        lines.append(f"### {role}\n{msg['content']}")
    return "\n\n---\n\n".join(lines)


def _cloud_config_to_tuple(cc):
    """Convert cloud config dict to a hashable tuple for cache keying.
    Returns None if cloud config is None (local mode)."""
    if cc is None:
        return None
    return (cc.get('provider', ''), cc.get('key_id', ''), cc.get('secret_key', ''),
            cc.get('region', ''), cc.get('endpoint', ''),
            cc.get('bucket', ''), cc.get('prefix', ''))


# ==========================================
# PII REDACTION
# ==========================================

def detect_pii_columns(df):
    """Scan column names for PII patterns. Returns list of matching column names."""
    pii_cols = []
    for col in df.columns:
        for pattern in PII_COLUMN_PATTERNS:
            if re.search(pattern, col):
                pii_cols.append(col)
                break
    return pii_cols


def redact_pii(df):
    """Return a copy with PII columns replaced by '[REDACTED]'. Original unchanged."""
    pii_cols = detect_pii_columns(df)
    if not pii_cols:
        return df, []
    redacted_df = df.copy()
    for col in pii_cols:
        redacted_df[col] = "[REDACTED]"
    return redacted_df, pii_cols


# ==========================================
# RELATIONSHIP READER
# ==========================================

def get_stored_relationships(metadata):
    """Read relationships from metadata.json. Falls back to column-scan for older metadata."""
    stored = metadata.get('relationships', None)
    if stored is not None:
        return [f"- {r['from_table']}.{r['from_column']} → {r['to_table']}.{r['to_column']}"
                for r in stored]

    # Fallback for pre-V6 metadata
    col_to_tables = {}
    for col_info in metadata.get('columns', []):
        col_name = col_info['name']
        table_name = col_info.get('table', 'data')
        if col_name not in col_to_tables:
            col_to_tables[col_name] = []
        if table_name not in col_to_tables[col_name]:
            col_to_tables[col_name].append(table_name)

    relationships = []
    for col_name, tables in col_to_tables.items():
        if len(tables) >= 2:
            for i in range(len(tables)):
                for j in range(i + 1, len(tables)):
                    relationships.append(f"- {tables[i]}.{col_name} ↔ {tables[j]}.{col_name}")
    return relationships


def build_relationship_context(metadata):
    """Build relationship context block for the SQL prompt."""
    relationships = get_stored_relationships(metadata)
    if not relationships:
        return ""
    header = ("\nKNOWN TABLE RELATIONSHIPS (from preprocessing analysis — "
              "use these for JOIN conditions):\n")
    return header + "\n".join(relationships) + "\n"


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🤖 Data Assistant V8")

    # ---- LLM Provider ----
    st.write("### 🧠 LLM Configuration")
    provider_name = st.selectbox("Provider", list(PROVIDER_CONFIG.keys()), key="provider_select")
    config = PROVIDER_CONFIG[provider_name]

    api_key = ""
    try:
        if config["secret_key"] in st.secrets:
            api_key = st.secrets[config["secret_key"]]
    except Exception:
        pass

    if api_key:
        st.success(f"🔑 {provider_name} key loaded from Secrets")
    else:
        api_key = st.text_input(f"{provider_name} API Key", type="password",
                                key=f"apikey_{provider_name}")

    model_name = st.text_input("Model Name", value=config["default_model"],
                               key=f"model_{provider_name}")

    st.markdown("---")

    # ---- Privacy Controls ----
    st.write("### 🔒 Privacy Controls")
    st.session_state.pii_redaction = st.toggle(
        "🛡️ **PII Redaction**",
        value=st.session_state.pii_redaction,
        help="When enabled, PII columns are replaced with [REDACTED] before "
             "data is sent to the LLM. Full data still shown in results table."
    )
    if st.session_state.pii_redaction:
        st.caption("✅ PII columns will be redacted before LLM summarization")
    else:
        st.caption("⚠️ PII redaction is OFF — raw data including any student PII will be sent to the LLM")

    st.markdown("---")

    # ---- Data Connection ----
    if not st.session_state.data_active:
        st.write("### 📂 Data Source")

        target_dir = st.text_input(
            "Artifacts Folder Path", value=st.session_state.active_path,
            help="Local folder (or repo path on Streamlit Cloud) containing metadata.json and chroma_db/"
        )

        # V7: Cloud storage toggle
        use_cloud = st.toggle(
            "☁️ **Read Parquet files from cloud storage**",
            value=False,
            help="Enable this if your Parquet files are hosted on AWS S3, Google Cloud Storage, "
                 "or Cloudflare R2 instead of in the local artifacts folder. "
                 "The metadata.json and chroma_db/ are still read from the local path above."
        )

        # Cloud configuration (only shown when toggle is on)
        cloud_config_input = None
        if use_cloud:
            st.markdown("---")
            st.caption("📡 **Cloud Storage Configuration**")
            st.info("ℹ️ The **metadata file** and **AI index** (chroma_db/) are still read from the "
                    "local folder above — only the large Parquet data files are read from cloud storage.")

            cloud_provider = st.selectbox(
                "Cloud Provider",
                ["AWS S3", "Google Cloud Storage", "Cloudflare R2"],
                key="cloud_provider_select"
            )

            # Provider-specific help text
            if cloud_provider == "AWS S3":
                st.caption("💡 Create an S3 bucket and IAM user with read access. "
                           "Use the **Access Key ID** and **Secret Access Key** from "
                           "IAM → Users → Security Credentials.")
            elif cloud_provider == "Google Cloud Storage":
                st.caption("💡 Create **HMAC keys** in Cloud Storage → Settings → Interoperability. "
                           "These work like S3 credentials and allow DuckDB to read your files directly.")
            elif cloud_provider == "Cloudflare R2":
                st.caption("💡 Create API tokens in R2 → **Manage R2 API Tokens**. "
                           "R2 has **zero egress fees** — the best value for large datasets.")

            cloud_bucket = st.text_input("Bucket Name", key="cloud_bucket_input",
                                         placeholder="my-education-data")
            cloud_prefix = st.text_input("Folder Prefix (path inside bucket)",
                                         value="deploy_artifacts/", key="cloud_prefix_input",
                                         help="The folder path inside your bucket where Parquet files are stored. "
                                              "Include the trailing slash.")

            # Credentials: try secrets first
            cloud_key_id = ""
            cloud_secret_key = ""
            try:
                if "CLOUD_KEY_ID" in st.secrets:
                    cloud_key_id = st.secrets["CLOUD_KEY_ID"]
                if "CLOUD_SECRET_KEY" in st.secrets:
                    cloud_secret_key = st.secrets["CLOUD_SECRET_KEY"]
            except Exception:
                pass

            if cloud_key_id and cloud_secret_key:
                st.success("🔑 Cloud credentials loaded from Secrets")
            else:
                if not cloud_key_id:
                    cloud_key_id = st.text_input("Access Key ID", type="password",
                                                 key="cloud_key_input")
                if not cloud_secret_key:
                    cloud_secret_key = st.text_input("Secret Access Key", type="password",
                                                     key="cloud_secret_input")

            # Provider-specific settings
            cloud_region = ""
            cloud_endpoint = ""

            if cloud_provider == "AWS S3":
                cloud_region = st.text_input("AWS Region", value="us-east-1", key="cloud_region_input")
                cloud_endpoint = ""
            elif cloud_provider == "Google Cloud Storage":
                cloud_region = "auto"
                cloud_endpoint = "storage.googleapis.com"
            elif cloud_provider == "Cloudflare R2":
                r2_account_id = st.text_input("Cloudflare Account ID", key="r2_account_input",
                                               help="Found in your Cloudflare dashboard URL or R2 settings.")
                cloud_region = "auto"
                cloud_endpoint = f"{r2_account_id}.r2.cloudflarestorage.com" if r2_account_id else ""

            # Build cloud config dict if all required fields are filled
            if cloud_bucket and cloud_key_id and cloud_secret_key:
                # Ensure prefix ends with /
                if cloud_prefix and not cloud_prefix.endswith('/'):
                    cloud_prefix += '/'
                cloud_config_input = {
                    "provider": cloud_provider,
                    "bucket": cloud_bucket,
                    "prefix": cloud_prefix,
                    "key_id": cloud_key_id,
                    "secret_key": cloud_secret_key,
                    "region": cloud_region,
                    "endpoint": cloud_endpoint,
                }

        # Connect button
        if st.button("🚀 Connect to Data", type="primary"):
            if not api_key:
                st.error("Please enter an API Key first.")
            elif not os.path.isfile(os.path.join(target_dir, "metadata.json")):
                st.error("❌ Invalid folder: `metadata.json` not found. "
                         "This file must be in the local artifacts folder even when using cloud storage.")
            elif use_cloud and not cloud_config_input:
                st.error("❌ Please fill in all cloud storage fields (bucket, credentials).")
            else:
                st.session_state.active_path = target_dir
                st.session_state.data_active = True
                st.session_state.cloud_config = cloud_config_input if use_cloud else None
                st.session_state.starter_questions = []
                st.session_state.messages = [
                    {"role": "assistant",
                     "content": f"Connected to `{target_dir}`"
                                + (f" with cloud storage ({cloud_config_input['provider']})" if cloud_config_input else "")
                                + ". Ask me anything about your data!"}
                ]
                st.rerun()
    else:
        st.write("### ✅ Data Connected")
        st.success(f"`{st.session_state.active_path}`")
        st.caption(f"🧠 {provider_name} → `{model_name}`")

        # Show cloud status if active
        if st.session_state.cloud_config:
            cc = st.session_state.cloud_config
            st.caption(f"☁️ {cc['provider']} → `{cc['bucket']}/{cc['prefix']}`")

        # Table inventory
        meta_path = os.path.join(st.session_state.active_path, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    _sidebar_meta = json.load(f)
                if 'tables' in _sidebar_meta:
                    with st.expander("📋 Available Tables", expanded=False):
                        for t_name, t_info in _sidebar_meta['tables'].items():
                            row_count = t_info.get('total_rows', '?')
                            if isinstance(row_count, (int, float)):
                                st.markdown(f"**`{t_name}`** — {row_count:,} rows")
                            else:
                                st.markdown(f"**`{t_name}`**")
                    rels = get_stored_relationships(_sidebar_meta)
                    if rels:
                        with st.expander("🔗 Table Relationships", expanded=False):
                            for r in rels:
                                st.markdown(r)
            except Exception:
                pass

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Switch"):
                st.session_state.data_active = False
                st.session_state.cloud_config = None
                st.session_state.starter_questions = []
                st.rerun()
        with col2:
            if st.button("🔃 Refresh"):
                load_resources.clear()
                execute_validated_sql.clear()
                st.session_state.starter_questions = []
                st.rerun()

        if len(st.session_state.messages) > 1:
            st.markdown("---")
            chat_export = _format_chat_export()
            st.download_button("📥 Export Chat History", chat_export,
                               file_name="chat_history.md", mime="text/markdown")

    st.markdown("---")

    # ---- Privacy Notice ----
    pii_status = "ON — PII columns are scrubbed" if st.session_state.pii_redaction else "OFF — raw data is sent"
    cloud_note = ""
    if st.session_state.cloud_config:
        cc = st.session_state.cloud_config
        cloud_note = f"\n    5. **Cloud Storage:** Parquet data is read from **{cc['provider']}** (`{cc['bucket']}`)."

    st.warning(f"""
    🔒 **Data Privacy Notice**

    1. **Storage:** Data lives as binary Parquet files (not human-readable).
    2. **SQL Generation:** The AI sees your **schema only** (column names/types) — no data values.
    3. **⚠️ Summarization:** After each query, up to **{MAX_ROWS_FOR_SUMMARY} rows of actual data**
       are sent to **{provider_name}** (`{model_name}`) to generate the plain-English summary.
    4. **PII Redaction:** Currently **{pii_status}** before summarization.{cloud_note}

    If your data contains student PII/PHI, ensure PII Redaction is enabled
    or consider using a self-hosted LLM.
    """)


# ==========================================
# RESOURCE LOADING (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Loading data resources...")
def load_resources(path):
    """Load metadata and ChromaDB collection from the artifacts directory."""
    try:
        if not os.path.exists(path):
            return None, None

        meta_path = os.path.join(path, "metadata.json")
        if not os.path.isfile(meta_path):
            return None, None

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        chroma_path = os.path.join(path, "chroma_db")
        if os.path.isdir(chroma_path):
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            collection = chroma_client.get_collection("dataset_schema")
        else:
            collection = None

        return metadata, collection
    except Exception as e:
        logging.error(f"Failed to load resources from {path}: {e}")
        return None, None


# ==========================================
# CONTEXT BUILDERS
# ==========================================

def build_table_inventory(metadata, artifacts_path, cloud_config=None):
    """Build table inventory for the SQL prompt.
    Uses s3:// URLs when cloud_config is provided, local paths otherwise."""
    table_list = []
    if 'tables' in metadata:
        for t_name, t_info in metadata['tables'].items():
            if cloud_config:
                # Cloud mode: build s3:// URL
                f_pattern = f"s3://{cloud_config['bucket']}/{cloud_config['prefix']}{t_info['file_pattern']}"
            else:
                # Local mode: build local path
                f_pattern = os.path.join(artifacts_path, t_info['file_pattern']).replace("\\", "/")
            table_list.append(f"- Table: '{t_name}' → File: '{f_pattern}'")
    else:
        if cloud_config:
            f_pattern = f"s3://{cloud_config['bucket']}/{cloud_config['prefix']}data_*.parquet"
        else:
            f_pattern = os.path.join(artifacts_path, "data_*.parquet").replace("\\", "/")
        table_list.append(f"- Table: 'data' → File: '{f_pattern}'")
    return "\n".join(table_list)


def build_context_block(metadata, collection, question):
    """Build schema context: full schema if <200 cols, vector search otherwise."""
    all_columns = metadata.get('columns', [])
    if len(all_columns) < 200:
        schema_lines = []
        for col in all_columns:
            t = col.get('table', 'data')
            schema_lines.append(f"  Table: {t} | Column: {col['name']} | Type: {col['type']}")
        return "FULL SCHEMA REFERENCE (use these EXACT column names):\n" + "\n".join(schema_lines)
    else:
        if collection is None:
            return "SCHEMA: (vector index unavailable — write best-effort SQL)"
        results = collection.query(query_texts=[question], n_results=8)
        docs = results['documents'][0] if results['documents'] else []
        return "RELEVANT COLUMNS (from vector search):\n" + "\n".join(docs)


# ==========================================
# STARTER QUESTIONS
# ==========================================

def generate_starter_questions(metadata, table_inventory, client, model_name):
    """Generate 5 schema-aware questions an educator would ask."""
    all_columns = metadata.get('columns', [])
    schema_summary_lines = []
    for col in all_columns[:100]:
        t = col.get('table', 'data')
        schema_summary_lines.append(f"  {t}.{col['name']} ({col['type']})")
    schema_summary = "\n".join(schema_summary_lines)
    if len(all_columns) > 100:
        schema_summary += f"\n  ... and {len(all_columns) - 100} more columns"

    rel_context = build_relationship_context(metadata)

    prompt = f"""You are an expert education data analyst. Below is the schema of a dataset
from a learning management system (LMS) or educational platform.

AVAILABLE TABLES:
{table_inventory}

SCHEMA:
{schema_summary}
{rel_context}
Based on this schema, generate exactly 5 natural-language questions that a school administrator,
instructor, or education analyst would realistically want to ask about this data.

RULES:
- Questions should be specific to the columns and tables available
- Mix question types: counts, averages, comparisons, trends, and "who/what" queries
- Frame questions in plain English, as a non-technical educator would ask them
- Keep each question under 15 words
- Output ONLY the 5 questions, one per line, numbered 1-5. No other text.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an education data expert. Output only the requested questions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if cleaned:
                    questions.append(cleaned)
        return questions[:5]
    except Exception as e:
        logging.error(f"Failed to generate starter questions: {e}")
        return []


# ==========================================
# SQL GENERATION
# ==========================================

def _strip_markdown_sql(text):
    """Remove markdown code fences from LLM SQL output."""
    text = text.strip()
    if text.startswith("```sql"):
        text = text[6:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def get_sql_query(question, table_inventory, context_block, client, model_name,
                  conversation_context="", relationship_context=""):
    """Generate a DuckDB SQL query for the user's question."""
    today = datetime.date.today().strftime("%Y-%m-%d")

    prompt = f"""You are an expert DuckDB SQL analyst specializing in education and learning management system (LMS) data.
Current Date: {today}

AVAILABLE TABLES & FILES:
{table_inventory}

{context_block}
{relationship_context}
{conversation_context}
USER QUESTION:
"{question}"

EDUCATION DOMAIN RULES:
- "Completion rate" = COUNT(completed) / COUNT(total) as a percentage
- "Pass rate" = students scoring above a threshold (typically 70% unless specified)
- "At-risk students" = low engagement, missing assignments, or declining scores
- "Cohort analysis" = grouping students by enrollment period or intake group
- When asked about "performance", default to assessment scores or grades
- When asked about "engagement", look for login frequency, session duration, activity counts, or participation metrics
- When asked about "progress", look for module/lesson/course completion percentages
- Academic terms like "semester", "term", "quarter" typically map to date ranges
- Grading scales vary: look for letter grades (A-F), percentages, or points
- When comparing groups, include both absolute numbers AND percentages

SQL RULES:
0. COLUMN NAMES: Use the EXACT names from the schema. Never rename or invent columns.
1. JOIN STRATEGY: Use the KNOWN TABLE RELATIONSHIPS above for JOIN conditions. If two tables don't share a key, look for a bridge table. Use LEFT JOIN for "missing" items.
2. Output ONLY valid DuckDB SQL. No markdown, no explanation.
3. SYNTAX: Use read_parquet('path/pattern') in FROM clauses. Do NOT use read_csv or any other file-reading function.
4. AGGREGATION: Always include GROUP BY when mixing aggregates with non-aggregates.
5. LIMIT: Default to 50 rows unless the user specifies otherwise.
6. SCHEMA QUESTIONS: Use DESCRIBE SELECT * FROM read_parquet('...') to show columns.
7. TEXT SEARCH: Use ILIKE '%term%' for fuzzy matching.
8. SAFE DIVISION: CASE WHEN denominator = 0 THEN 0 ELSE numerator/denominator END.
9. ORDERING: "Top" or "Best" means ORDER BY ... DESC.
10. COUNTING: Use COUNT(DISTINCT col) when appropriate.
11. DATE GROUPING: Use date_trunc('month', date_col) or similar.
12. NULLS: Exclude NULLs in averages/counts with WHERE col IS NOT NULL.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert education data SQL generator. Output only valid SQL. Use only read_parquet() for data access."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return _strip_markdown_sql(response.choices[0].message.content)


def fix_sql_query(question, failed_sql, error_msg, table_inventory, context_block,
                  client, model_name, relationship_context=""):
    """Ask the LLM to fix a failed SQL query."""
    prompt = f"""The following DuckDB SQL query failed.

FAILED SQL:
{failed_sql}

ERROR MESSAGE:
{error_msg}

AVAILABLE TABLES & FILES:
{table_inventory}

{context_block}
{relationship_context}
ORIGINAL QUESTION:
"{question}"

Fix the SQL. Use only read_parquet() for data access. Output ONLY the corrected SQL. No markdown, no explanation.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a SQL debugger. Output only corrected SQL."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return _strip_markdown_sql(response.choices[0].message.content)


# ==========================================
# SQL SECURITY VALIDATOR
# ==========================================

def validate_sql(sql_query, artifacts_path, cloud_config=None):
    """
    Validate SQL for security. Returns cleaned SQL on success.
    Raises ValueError on violation.

    V7: Accepts optional cloud_config. When provided, allows s3:// URLs
    matching the configured bucket/prefix. Blocks http(s):// URLs always.
    """
    clean_sql = sql_query.strip().rstrip(';').strip()

    if not clean_sql:
        raise ValueError("Security Alert: Empty query.")

    sql_no_strings = re.sub(r"'[^']*'", "", clean_sql)
    if ';' in sql_no_strings:
        raise ValueError("Security Alert: Multiple SQL statements are not allowed.")

    upper_sql = clean_sql.upper().lstrip()
    if not (upper_sql.startswith("SELECT") or
            upper_sql.startswith("DESCRIBE") or
            upper_sql.startswith("WITH")):
        raise ValueError("Security Alert: Only SELECT, DESCRIBE, or WITH (CTE) queries are permitted.")

    for kw in DANGEROUS_KEYWORDS:
        if re.search(r'\b' + kw + r'\b', upper_sql):
            raise ValueError(f"Security Alert: Forbidden keyword '{kw}' detected.")

    # Path containment check
    allowed_local_prefix = os.path.realpath(artifacts_path).replace("\\", "/")

    # Build allowed cloud prefix if cloud mode is active
    allowed_s3_prefix = None
    if cloud_config:
        allowed_s3_prefix = f"s3://{cloud_config['bucket']}/{cloud_config['prefix']}"

    sql_for_path_check = clean_sql.replace("''", "")
    string_literals = re.findall(r"'([^']*)'", sql_for_path_check)

    for literal in string_literals:
        # V7: Block all http(s):// URLs — even with httpfs loaded,
        # we only allow s3:// to the configured bucket.
        if literal.lower().startswith('http://') or literal.lower().startswith('https://'):
            raise ValueError("Security Alert: HTTP/HTTPS URLs are not permitted in queries. "
                             "Only local file paths or configured cloud storage (s3://) are allowed.")

        # Check for cloud storage references
        if literal.startswith('s3://'):
            if allowed_s3_prefix:
                # Strip wildcard for prefix matching
                check_literal = literal.split('*')[0] if '*' in literal else literal
                if not check_literal.startswith(allowed_s3_prefix):
                    raise ValueError("Security Alert: Query references an unauthorized cloud storage location.")
            else:
                raise ValueError("Security Alert: Cloud storage references (s3://) are not allowed in local mode.")
            continue  # Skip local path check for cloud URLs

        # Local path containment check
        looks_like_path = (
            '/' in literal or '\\' in literal or
            literal.endswith('.parquet') or literal.endswith('.csv')
        )
        if looks_like_path:
            literal_normalized = literal.replace("\\", "/")
            if '*' in literal_normalized:
                check_target = os.path.realpath(os.path.dirname(literal_normalized)).replace("\\", "/")
            else:
                check_target = os.path.realpath(literal_normalized).replace("\\", "/")

            if not check_target.startswith(allowed_local_prefix):
                raise ValueError("Security Alert: Query references files outside the allowed directory.")

    if 'LIMIT' not in upper_sql:
        clean_sql = f"SELECT * FROM ({clean_sql}) AS _limited LIMIT {HARD_ROW_LIMIT}"

    return clean_sql


# ==========================================
# SQL EXECUTOR (CACHED)
# ==========================================

@st.cache_data(show_spinner=False, ttl=86400)
def execute_validated_sql(clean_sql, cloud_config_tuple=None):
    """
    Execute pre-validated SQL against DuckDB.

    V7: When cloud_config_tuple is provided, installs and configures httpfs
    for S3-compatible cloud storage access. Extension auto-loading is disabled
    after httpfs is loaded to prevent other extensions from being activated.
    """
    conn = duckdb.connect(database=':memory:')
    try:
        if cloud_config_tuple:
            # Unpack cloud config
            provider, key_id, secret_key, region, endpoint, bucket, prefix = cloud_config_tuple

            # Install and load httpfs for S3 access
            conn.execute("INSTALL httpfs")
            conn.execute("LOAD httpfs")

            # Configure S3 credentials
            conn.execute(f"SET s3_access_key_id='{key_id}'")
            conn.execute(f"SET s3_secret_access_key='{secret_key}'")

            # Provider-specific settings
            if region and region != "auto":
                conn.execute(f"SET s3_region='{region}'")
            if endpoint:
                conn.execute(f"SET s3_endpoint='{endpoint}'")
                conn.execute("SET s3_url_style='path'")

        # Lock down extensions after httpfs is loaded (or in local mode)
        conn.execute("SET autoinstall_known_extensions=false")
        conn.execute("SET autoload_known_extensions=false")

        return conn.execute(clean_sql).df()
    finally:
        conn.close()


# ==========================================
# SUMMARIZATION
# ==========================================

def summarize_answer(question, df, client, model_name, pii_redaction_enabled):
    """EdTech-tuned summarization with optional PII redaction."""
    total_rows = len(df)
    summary_df = df.head(MAX_ROWS_FOR_SUMMARY) if total_rows > MAX_ROWS_FOR_SUMMARY else df.copy()

    redacted_cols = []
    if pii_redaction_enabled:
        summary_df, redacted_cols = redact_pii(summary_df)

    data_str = summary_df.to_string(index=False)
    truncation_note = f"\n(Showing first {MAX_ROWS_FOR_SUMMARY} of {total_rows} total rows)" if total_rows > MAX_ROWS_FOR_SUMMARY else ""
    redaction_note = (f"\n(Note: The following columns were redacted for privacy: "
                     f"{', '.join(redacted_cols)}. Do not speculate about their values.)") if redacted_cols else ""

    prompt = f"""You are an expert education data analyst summarizing query results for
school administrators and instructors.

User Question: {question}

SQL Result Data:
{data_str}{truncation_note}{redaction_note}

SUMMARIZATION RULES:
1. Frame findings in educational terms (e.g., "3 students are at risk of failing"
   not "3 rows have values below 70").
2. Mention specific key numbers, percentages, and patterns.
3. Where appropriate, suggest actionable next steps for educators.
4. If the sample size is small (< 20 records), note that percentages may be misleading.
5. If data was redacted for privacy, acknowledge it but do not speculate about redacted values.
6. Use clear, plain English appropriate for non-technical education professionals.
7. Be concise — aim for 3-5 sentences for simple results, up to a short paragraph for complex ones.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ==========================================
# AUTO-VISUALIZATION
# ==========================================

def attempt_visualization(df):
    """Auto-generate chart: line (date+num), bar (cat+num), scatter (2 num), histogram."""
    try:
        if df.empty or len(df.columns) < 1:
            return

        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

        if not date_cols and cat_cols:
            for c in list(cat_cols):
                try:
                    converted = pd.to_datetime(df[c], errors='coerce', format='mixed', utc=True)
                    if converted.notna().mean() > 0.8:
                        df[c] = converted
                        date_cols.append(c)
                        cat_cols.remove(c)
                        break
                except Exception:
                    pass

        st.caption("📊 Auto-Visualization")

        if date_cols and num_cols:
            chart_df = df.set_index(date_cols[0])[num_cols[:3]].sort_index()
            st.line_chart(chart_df)
            return
        if cat_cols and num_cols:
            chart_df = df.head(25).set_index(cat_cols[0])[num_cols[0]]
            st.bar_chart(chart_df)
            return
        if len(num_cols) >= 2:
            st.scatter_chart(df, x=num_cols[0], y=num_cols[1])
            return
        if len(num_cols) >= 1:
            col_data = df[num_cols[0]].dropna()
            if 0 < col_data.nunique() <= 50:
                st.bar_chart(col_data.value_counts().sort_index())
                return
    except Exception as viz_err:
        logging.debug(f"Auto-viz skipped: {viz_err}")


# ==========================================
# MAIN APP
# ==========================================
st.title("🤖 Chat with your Education Data")

with st.expander("💡 Pro-Tips for better answers & charts", expanded=False):
    st.markdown("""
    Because this AI writes literal database queries, specific questions work much better than vague ones!
    *   **Name the data:** Ask for *"5 random users"* rather than *"5 random rows of data"*.
    *   **Ask for the math:** If you want a total, ask *"What is the total count?"* instead of *"Show me all the records."* (The AI only reads the first 50 rows to write its summary).
    *   **Getting good charts:** The app auto-draws charts based on your question. For a clean line chart, ask for an aggregation over time (e.g., *"Show me the count of logins by month"*).
    *   **Use follow-ups:** Start broad, then narrow down. (*"Which 5 courses have the highest scores?"* → *"Now break those down by department."*)
    """)

if not st.session_state.data_active:
    st.info("👈 Select an LLM provider, enter your API key, and connect to a data folder in the sidebar.")
    st.stop()

if not api_key:
    st.warning("⚠️ No API key provided. Please enter one in the sidebar.")
    st.stop()

llm_client = OpenAI(api_key=api_key, base_url=config["base_url"])

resources = load_resources(st.session_state.active_path)
if resources is None or resources[0] is None:
    st.error(f"⚠️ Could not load artifacts from `{st.session_state.active_path}`. "
             f"Ensure it contains `metadata.json` and optionally `chroma_db/`.")
    st.stop()

metadata, collection = resources
artifacts_path = st.session_state.active_path
cloud_config = st.session_state.cloud_config

# Precompute context (V7: pass cloud_config to table inventory)
table_inventory = build_table_inventory(metadata, artifacts_path, cloud_config)
relationship_context = build_relationship_context(metadata)

# Generate starter questions (once per connection)
if not st.session_state.starter_questions and api_key:
    with st.spinner("🎓 Generating suggested questions for your dataset..."):
        st.session_state.starter_questions = generate_starter_questions(
            metadata, table_inventory, llm_client, model_name
        )

# Render chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Starter question buttons
if st.session_state.starter_questions and len(st.session_state.messages) <= 2:
    st.write("#### 💡 Suggested Questions")
    st.caption("Click any question below to get started, or type your own.")
    clicked_question = None
    for q in st.session_state.starter_questions:
        if st.button(f"📝 {q}", key=f"starter_{q}", width="stretch"):
            clicked_question = q
    if clicked_question:
        st.session_state.messages.append({"role": "user", "content": clicked_question})
        st.session_state.starter_questions = []
        st.rerun()

# Handle new chat input
if user_input := st.chat_input("Ask specifically... e.g., 'What is the total count of users by month?'"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    st.session_state.starter_questions = []

    # Prepare cloud config tuple for the cached executor
    cc_tuple = _cloud_config_to_tuple(cloud_config)

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=False) as status:
            try:
                # Step 1: Schema context
                context_block = build_context_block(metadata, collection, user_input)
                total_cols = len(metadata.get('columns', []))
                if total_cols < 200:
                    status.write("🧠 Full Schema Mode (all columns fit in context)")
                else:
                    status.write(f"🔍 Vector Search Mode ({total_cols} columns → top 8 relevant)")

                if cloud_config:
                    status.write(f"☁️ Cloud mode active — reading from {cloud_config['provider']}")

                # Step 2: Conversation context
                conversation_context = ""
                msgs = st.session_state.messages
                if len(msgs) >= 4:
                    if msgs[-2]["role"] == "assistant" and msgs[-3]["role"] == "user":
                        prev_q = msgs[-3]["content"]
                        prev_a = msgs[-2]["content"][:500]
                        conversation_context = (
                            f"\nPREVIOUS QUESTION: \"{prev_q}\"\n"
                            f"PREVIOUS ANSWER SUMMARY: \"{prev_a}\"\n"
                        )

                # Step 3: Generate SQL
                status.write(f"📝 Generating SQL via **{provider_name}** (`{model_name}`)...")
                sql = get_sql_query(
                    user_input, table_inventory, context_block,
                    llm_client, model_name,
                    conversation_context, relationship_context
                )
                st.code(sql, language="sql")

                # Step 4: Validate (V7: pass cloud_config for s3:// path validation)
                clean_sql = validate_sql(sql, artifacts_path, cloud_config)

                # Step 5: Execute (with retry)
                status.write("⚡ Executing query...")
                try:
                    df = execute_validated_sql(clean_sql, cc_tuple)
                except Exception as exec_err:
                    err_str = str(exec_err)
                    # V7: Detect cloud authentication errors
                    if any(hint in err_str for hint in ['HTTP 403', 'Access Denied', 'SignatureDoesNotMatch']):
                        raise Exception("☁️ Cloud storage authentication failed. "
                                        "Please check your Access Key ID and Secret Key in the sidebar.")
                    if any(hint in err_str for hint in ['HTTP 404', 'NoSuchBucket', 'NoSuchKey']):
                        raise Exception(f"☁️ Cloud storage resource not found. "
                                        f"Please verify your bucket name and folder prefix.")

                    logging.error(f"First SQL attempt failed: {exec_err}")
                    status.write("🔄 Query failed. Asking AI to self-correct...")
                    try:
                        sql_v2 = fix_sql_query(
                            user_input, sql, str(exec_err),
                            table_inventory, context_block,
                            llm_client, model_name, relationship_context
                        )
                        st.code(sql_v2, language="sql")
                        clean_sql_v2 = validate_sql(sql_v2, artifacts_path, cloud_config)
                        df = execute_validated_sql(clean_sql_v2, cc_tuple)
                    except ValueError as ve:
                        raise ve
                    except Exception as retry_err:
                        logging.error(f"Retry also failed: {retry_err}")
                        raise Exception("Query failed after automatic retry. Please try rephrasing your question.")

                # Step 6: Truncation warning
                if len(df) >= HARD_ROW_LIMIT:
                    st.warning(f"⚠️ Results were capped at {HARD_ROW_LIMIT:,} rows. "
                               "Add a more specific WHERE clause or aggregation to narrow results.")

                # Step 7: Display
                st.dataframe(df, width="stretch")
                if not df.empty:
                    csv_data = df.to_csv(index=False)
                    st.download_button("📥 Download Results (CSV)", csv_data,
                                       file_name="query_results.csv", mime="text/csv")

                # Step 8: Visualization
                attempt_visualization(df)

                # Step 9: Summarize
                if not df.empty:
                    status.write("💬 Generating summary...")
                    if st.session_state.pii_redaction:
                        pii_cols = detect_pii_columns(df)
                        if pii_cols:
                            status.write(f"🛡️ PII redaction active — scrubbing columns: {', '.join(pii_cols)}")
                        else:
                            status.write("🛡️ PII redaction active — no PII columns detected in results")
                    answer = summarize_answer(user_input, df, llm_client, model_name,
                                              st.session_state.pii_redaction)
                else:
                    answer = "The query returned no results. Try broadening your question or checking the table/column names."

                status.update(label="✅ Answer Ready", state="complete")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except ValueError as ve:
                st.error(f"🔒 {str(ve)}")
                status.update(label="🔒 Blocked", state="error")
            except Exception as e:
                logging.error(f"Unhandled chat error: {e}", exc_info=True)
                st.error(f"Error: {str(e)}")
                status.update(label="❌ Failed", state="error")
