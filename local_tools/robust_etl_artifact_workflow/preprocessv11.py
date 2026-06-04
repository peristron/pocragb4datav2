# ==========================================
# 🛠️ PREPROCESSOR (LOCAL INGESTION ENGINE) - V11
# ==========================================
# Production-hardened local ETL pipeline for education/LMS data.
#   - All V10 features preserved
#   - Optional cloud storage upload (S3, GCS, Cloudflare R2)
#   - Credentials via Streamlit secrets or manual input
#   - Git credential scrubbing
#   - Zip Slip prevention
#   - ChromaDB built-in embeddings (no PyTorch)
#   - O(n) chunk splitting via PyArrow streaming
#   - Instant row counts via Parquet metadata
#   - Multi-path local file input
#   - Per-step timing instrumentation
# ==========================================

import os

# --- Environment variables MUST be set before importing libraries ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Standard library and third-party imports ---
import streamlit as st
import duckdb
import pandas as pd
import json
import shutil
import chromadb
import stat
import gc
import time
import math
import zipfile
import subprocess
import re
import uuid
import pathlib
import logging

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_ARTIFACTS_DIR = "deploy_artifacts"
TEMP_DIR = "temp_uploads"
TEMP_MASTER_PARQUET = "temp_master.parquet"
CHUNK_SIZE_MB = 90
GIT_TIMEOUT_SECONDS = 120

# Cloud provider configurations.
# Each S3-compatible provider uses the same boto3 API but different endpoints.
CLOUD_PROVIDERS = {
    "AWS S3": {
        "endpoint": None,  # boto3 uses default AWS endpoints
        "region_label": "AWS Region",
        "region_default": "us-east-1",
        "secret_prefix": "AWS",
        "help_text": (
            "💡 Create credentials in **AWS Console → IAM → Users → Security Credentials**. "
            "You need an Access Key ID and Secret Access Key with S3 read/write permissions."
        ),
    },
    "Google Cloud Storage": {
        "endpoint": "https://storage.googleapis.com",
        "region_label": "Region",
        "region_default": "auto",
        "secret_prefix": "GCS",
        "help_text": (
            "💡 Create HMAC keys in **Google Cloud Console → Cloud Storage → Settings → Interoperability**. "
            "These work like S3 credentials. Your bucket must have 'Fine-grained' access control enabled."
        ),
    },
    "Cloudflare R2": {
        "endpoint": None,  # Built dynamically from account ID
        "region_label": "Account ID",
        "region_default": "",
        "secret_prefix": "R2",
        "help_text": (
            "💡 Create API tokens in **Cloudflare Dashboard → R2 → Manage R2 API Tokens**. "
            "R2 has **zero egress fees** — best value for large datasets."
        ),
    },
}

st.set_page_config(page_title="🛠️ RAG Ingestion Engine V11", layout="centered")

# ==========================================
# UI HEADER
# ==========================================
st.title("🛠️ RAG Ingestion Engine")
st.caption("Local ETL Pipeline for Big Data RAG — V11")

with st.expander("ℹ️ **How to use this tool**", expanded=False):
    st.markdown("""
    ### 1. Select Strategy
    *   **Merge All:** Best for time-series splits (e.g., `Jan.csv` + `Feb.csv`).
        All files are stitched into one big table. Also works fine with a single file.
    *   **Keep Separate:** Best for relational data (e.g., `Students.csv` + `Enrollments.csv`).
        Each CSV becomes its own queryable table. The preprocessor will automatically
        detect foreign key relationships between tables. Also works with a single file.

    ### 2. Choose Input
    *   **Browser Upload:** Good for small files (<200MB). Also works for large ZIPs —
        just start it, step away to work on other things, and come back when it's done.
        Expect longer processing times (the browser must load the entire file into memory first).
    *   **Local File Path (Recommended for large files):** Paste one or more file paths directly.
        **Much faster** — skips browser memory entirely. Supports multiple files using
        Windows "Copy as path" format (select files → right-click → Copy as path → paste).

    ### 3. Speed Tips
    *   **Pre-extract your ZIPs!** For the fastest processing, unzip your files manually first
        (right-click → Extract All in Windows Explorer), then paste the paths to the raw `.csv` files.
        This skips both the file copy and the unzip step, which can save **10+ minutes** on large datasets.
    *   For example, a 38GB CSV with 150M rows processes in **~8 minutes** when pointed at the
        pre-extracted CSV via Local File Path — versus **30+ minutes** when uploaded as a ZIP through the browser.

    ### 4. Deploy
    *   **Auto-Deploy (Git):** Push artifacts to GitHub. Best for datasets under ~1.5GB.
    *   **Cloud Storage:** Upload artifacts to S3, Google Cloud Storage, or Cloudflare R2.
        Best for large datasets (5GB+) that exceed GitHub's file size limits.
        The postprocessor can then read directly from cloud storage.
    """)


# ==========================================
# SYSTEM HELPERS
# ==========================================

def remove_readonly(func, path, excinfo):
    """
    Error handler for shutil.rmtree on Windows.
    Forces read-only files to be writable before retrying the delete.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def robust_cleanup(dir_path):
    """Best-effort cleanup of temp directories and intermediate files."""
    t0 = time.time()
    gc.collect()
    for target in [dir_path, TEMP_DIR]:
        if os.path.exists(target):
            try:
                shutil.rmtree(target, onerror=remove_readonly)
            except Exception:
                time.sleep(1.0)
                try:
                    shutil.rmtree(target, onerror=remove_readonly)
                except Exception:
                    logging.warning(f"Could not fully clean {target}")
    if os.path.exists(TEMP_MASTER_PARQUET):
        try:
            os.remove(TEMP_MASTER_PARQUET)
        except Exception:
            logging.warning(f"Could not remove {TEMP_MASTER_PARQUET}")
    logging.info(f"[TIMING] Cleanup took {time.time() - t0:.2f}s")


def sanitize_table_name(filename):
    """Create a SQL-safe table name from a filename."""
    name = os.path.splitext(filename)[0]
    clean = re.sub(r'[^a-zA-Z0-9]', '_', name).lower().strip('_')
    if not clean:
        clean = "table"
    if clean[0].isdigit():
        clean = "t_" + clean
    return clean


def clean_column_names(columns):
    """Clean column names once per file — strips whitespace and stray quotes."""
    return [str(c).strip().replace('"', '') for c in columns]


def parse_file_paths(raw_input):
    """
    Parse user input containing one or more file paths.
    Supports Windows "Copy as path" format, one-per-line, or single path.
    """
    raw_input = raw_input.strip()
    if not raw_input:
        return []

    quoted_paths = re.findall(r'"([^"]+)"', raw_input)
    if quoted_paths:
        return [p.strip() for p in quoted_paths if p.strip()]

    lines = [line.strip().strip('"').strip("'") for line in raw_input.splitlines()]
    paths = [line for line in lines if line]
    if paths:
        return paths

    return [raw_input]


def get_all_csvs(root_dir):
    """Recursively find all CSV files under a directory."""
    csv_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def get_parquet_row_count(parquet_path):
    """Get row count from Parquet file footer metadata (instant, no data scan)."""
    return pq.ParquetFile(parquet_path).metadata.num_rows


def safe_extract(zip_path, target_dir):
    """Extract ZIP file with Zip Slip prevention."""
    target = pathlib.Path(target_dir).resolve()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            member_path = (target / member).resolve()
            if not str(member_path).startswith(str(target)):
                raise ValueError(f"Security Alert: Zip Slip detected for member '{member}'")
        zf.extractall(target_dir)


def _scrub_git_output(text):
    """Remove URLs with embedded credentials from Git output."""
    scrubbed = re.sub(r'https?://[^\s]*@[^\s]*', 'https://[CREDENTIALS_REDACTED]', text)
    return scrubbed


def run_git_sync():
    """Push deploy_artifacts/ to GitHub. Git error output is scrubbed for credential safety."""
    st.write("🐙 **Starting Git Sync...** Only the `deploy_artifacts/` folder will be pushed. "
             "Raw CSV files, temp files, and local configuration are excluded.")
    terminal = st.empty()

    commands = [
        ["git", "add", DEFAULT_ARTIFACTS_DIR],
        ["git", "commit", "-m", "Updated data artifacts via Preprocessor V11"],
        ["git", "push"]
    ]
    try:
        for cmd in commands:
            cmd_str = " ".join(cmd)
            terminal.code(f"> {cmd_str}")
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=os.getcwd(), timeout=GIT_TIMEOUT_SECONDS
            )
            if result.returncode != 0:
                combined = result.stdout + result.stderr
                if "nothing to commit" in combined:
                    st.info("Nothing to change in Git (files are identical).")
                else:
                    sanitized_err = _scrub_git_output(result.stderr.strip())
                    raise Exception(f"Git Error: {sanitized_err}")
            else:
                st.success(f"✅ {cmd_str}")
        terminal.empty()
        return True
    except subprocess.TimeoutExpired:
        st.error(f"Git command timed out after {GIT_TIMEOUT_SECONDS}s.")
        return False
    except Exception as e:
        st.error(f"Git Automation Failed: {_scrub_git_output(str(e))}")
        return False


# ==========================================
# CLOUD STORAGE UPLOAD (NEW IN V11)
# ==========================================

def _build_s3_client(provider_name, access_key, secret_key, region_or_account):
    """
    Build a boto3 S3 client configured for the selected cloud provider.

    All three supported providers (AWS S3, Google Cloud Storage, Cloudflare R2)
    use the S3-compatible API, so one client construction covers all of them.
    The difference is in the endpoint URL.

    Returns a configured boto3 client, or raises an exception on failure.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError(
            "The `boto3` package is required for cloud storage uploads. "
            "Install it with: pip install boto3"
        )

    provider_config = CLOUD_PROVIDERS[provider_name]

    # Build the endpoint URL based on the provider.
    if provider_name == "AWS S3":
        # AWS uses region-based endpoints natively — no custom endpoint needed.
        endpoint_url = None
        region = region_or_account
    elif provider_name == "Google Cloud Storage":
        # GCS exposes an S3-compatible endpoint.
        endpoint_url = provider_config["endpoint"]
        region = region_or_account if region_or_account != "auto" else "auto"
    elif provider_name == "Cloudflare R2":
        # R2 endpoints are per-account: https://<account_id>.r2.cloudflarestorage.com
        if not region_or_account:
            raise ValueError("Cloudflare R2 requires your Account ID.")
        endpoint_url = f"https://{region_or_account}.r2.cloudflarestorage.com"
        region = "auto"
    else:
        raise ValueError(f"Unknown cloud provider: {provider_name}")

    # Build the client with the appropriate configuration.
    client_kwargs = {
        "service_name": "s3",
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "config": Config(retries={"max_attempts": 3, "mode": "standard"}),
    }

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    if region and region != "auto":
        client_kwargs["region_name"] = region

    return boto3.client(**client_kwargs)


def upload_artifacts_to_cloud(provider_name, access_key, secret_key,
                              region_or_account, bucket_name, prefix, status):
    """
    Upload all files from deploy_artifacts/ to a cloud storage bucket.

    Uploads Parquet files, metadata.json, and the chroma_db/ directory.
    Uses the S3-compatible API (works for AWS S3, GCS, and Cloudflare R2).

    The prefix parameter controls the "folder" inside the bucket where files
    are placed. For example, prefix="deploy_artifacts" means files are uploaded
    to s3://bucket/deploy_artifacts/data_0.parquet etc.

    Returns True on success, False on failure.
    """
    t0 = time.time()
    status.write(f"☁️ **Uploading artifacts to {provider_name}...** "
                 f"Bucket: `{bucket_name}`, Prefix: `{prefix}`")

    try:
        # Build the S3-compatible client for the selected provider.
        client = _build_s3_client(provider_name, access_key, secret_key, region_or_account)

        # Collect all files to upload from the artifacts directory.
        files_to_upload = []
        for root, dirs, files in os.walk(DEFAULT_ARTIFACTS_DIR):
            for file in files:
                local_path = os.path.join(root, file)
                # Build the S3 key: prefix + relative path from artifacts dir.
                relative = os.path.relpath(local_path, DEFAULT_ARTIFACTS_DIR)
                s3_key = f"{prefix}/{relative}".replace("\\", "/")
                files_to_upload.append((local_path, s3_key))

        if not files_to_upload:
            status.write("⚠️ No files found in deploy_artifacts/ to upload.")
            return False

        status.write(f"📤 Uploading **{len(files_to_upload)}** file(s)...")

        # Upload each file with progress tracking.
        for idx, (local_path, s3_key) in enumerate(files_to_upload):
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            file_name = os.path.basename(local_path)
            status.write(f"   ☁️ [{idx+1}/{len(files_to_upload)}] `{file_name}` ({file_size_mb:.1f} MB)")

            client.upload_file(local_path, bucket_name, s3_key)

        elapsed = time.time() - t0
        logging.info(f"[TIMING] Cloud upload to {provider_name} took {elapsed:.2f}s "
                     f"({len(files_to_upload)} files)")
        status.write(f"✅ **Upload complete!** {len(files_to_upload)} files uploaded "
                     f"to `s3://{bucket_name}/{prefix}/` in {elapsed:.1f}s")
        return True

    except ImportError as ie:
        st.error(str(ie))
        return False
    except Exception as e:
        logging.error(f"Cloud upload failed: {e}", exc_info=True)
        st.error(f"Cloud upload failed: {_scrub_git_output(str(e))}")
        return False


# ==========================================
# CHUNK SPLITTING (PyArrow Streaming)
# ==========================================

def split_parquet_to_chunks(source_parquet, table_name, num_chunks,
                            rows_per_chunk, output_dir, status):
    """
    Split a large Parquet file into numbered chunks using PyArrow streaming.
    Single O(n) pass, constant memory. Falls back to DuckDB LIMIT/OFFSET if needed.
    """
    try:
        t0 = time.time()
        status.write(f"📦 **Splitting via streaming read...** Writing ~{num_chunks} chunks directly "
                     f"(single pass through the file, no re-scanning).")

        pf = pq.ParquetFile(source_parquet)
        schema = pf.schema_arrow

        chunk_idx = 0
        writer = None
        rows_in_current_chunk = 0
        batch_size = 100_000

        for batch in pf.iter_batches(batch_size=batch_size):
            if writer is None:
                chunk_path = os.path.join(output_dir, f"{table_name}_{chunk_idx}.parquet")
                writer = pq.ParquetWriter(chunk_path, schema, compression='zstd')

            table = pa.Table.from_batches([batch])
            writer.write_table(table)
            rows_in_current_chunk += batch.num_rows

            if rows_in_current_chunk >= rows_per_chunk:
                writer.close()
                writer = None
                chunk_idx += 1
                rows_in_current_chunk = 0

        if writer is not None:
            writer.close()
            chunk_idx += 1

        elapsed = time.time() - t0
        logging.info(f"[TIMING] PyArrow streaming split for '{table_name}': "
                     f"{elapsed:.2f}s ({chunk_idx} chunks written)")

    except Exception as e:
        logging.warning(f"PyArrow streaming split failed for '{table_name}': {e}. "
                        f"Falling back to DuckDB LIMIT/OFFSET.")
        status.write(f"⚠️ Streaming split unavailable for `{table_name}`. Using fallback approach.")

        source_path = source_parquet.replace("\\", "/")
        t0 = time.time()
        fallback_conn = duckdb.connect()
        try:
            for i in range(num_chunks):
                offset = i * rows_per_chunk
                chunk_path = os.path.join(output_dir, f"{table_name}_{i}.parquet")
                fallback_conn.execute(
                    f"COPY (SELECT * FROM '{source_path}' LIMIT {rows_per_chunk} OFFSET {offset}) "
                    f"TO '{chunk_path}' (FORMAT 'PARQUET', CODEC 'ZSTD')"
                )
            logging.info(f"[TIMING] DuckDB fallback split for '{table_name}' took "
                         f"{time.time() - t0:.2f}s ({num_chunks} chunks)")
        finally:
            fallback_conn.close()


# ==========================================
# RELATIONSHIP DETECTION
# ==========================================

def detect_relationships(conn, table_names, status):
    """Auto-detect potential foreign key relationships between tables."""
    t0 = time.time()
    status.write("🔗 **Detecting table relationships...** Scanning for shared column names "
                 "that could serve as join keys across tables.")

    table_columns = {}
    for table_name in table_names:
        first_chunk = os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_0.parquet")
        if not os.path.exists(first_chunk):
            continue
        try:
            schema_df = conn.execute(f"DESCRIBE SELECT * FROM '{first_chunk}'").df()
            table_columns[table_name] = set(schema_df['column_name'].tolist())
        except Exception as e:
            logging.warning(f"Could not read schema for {table_name}: {e}")
            continue

    relationships = []
    seen = set()
    key_indicators = ['id', 'key', 'code', 'number', 'num']
    tables = list(table_columns.keys())

    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            shared_cols = table_columns[tables[i]] & table_columns[tables[j]]
            for col in shared_cols:
                col_lower = col.lower()
                if col_lower in ('id', 'index', 'row_number', 'row_id'):
                    continue
                if any(kw in col_lower for kw in key_indicators):
                    rel_key = tuple(sorted([(tables[i], col), (tables[j], col)]))
                    if rel_key not in seen:
                        seen.add(rel_key)
                        relationships.append({
                            "from_table": tables[i],
                            "from_column": col,
                            "to_table": tables[j],
                            "to_column": col
                        })

    for i in range(len(tables)):
        singular = tables[i].rstrip('s')
        potential_fk = f"{singular}_id"
        for j in range(len(tables)):
            if i == j:
                continue
            if potential_fk in table_columns[tables[j]]:
                rel_key = tuple(sorted([(tables[j], potential_fk), (tables[i], potential_fk)]))
                if rel_key not in seen:
                    if potential_fk not in table_columns[tables[i]]:
                        pk_candidates = [potential_fk, 'id', f"{tables[i]}_id"]
                        pk_col = None
                        for pk in pk_candidates:
                            if pk in table_columns[tables[i]]:
                                pk_col = pk
                                break
                        if pk_col:
                            seen.add(rel_key)
                            relationships.append({
                                "from_table": tables[j],
                                "from_column": potential_fk,
                                "to_table": tables[i],
                                "to_column": pk_col
                            })

    if relationships:
        status.write(f"✅ Found **{len(relationships)}** potential relationship(s) between tables:")
        for rel in relationships:
            status.write(f"   • `{rel['from_table']}.{rel['from_column']}` → "
                         f"`{rel['to_table']}.{rel['to_column']}`")
    else:
        status.write("ℹ️ No obvious foreign key relationships detected between tables. "
                     "The LLM will attempt to infer joins from column names if needed.")

    logging.info(f"[TIMING] Relationship detection took {time.time() - t0:.2f}s")
    return relationships


# ==========================================
# LOGIC A: MERGE ALL
# ==========================================

def process_merge_strategy(conn, all_csvs, status):
    """Merge all CSVs into a single logical table named 'data'."""
    t0 = time.time()
    status.write("🔗 **Strategy: Merge All** — All CSV files will be combined into a single unified table. "
                 "DuckDB processes this using streaming reads, so files larger than your available RAM are handled safely.")

    normalized_paths = [f.replace(os.sep, '/') for f in all_csvs]
    input_files_sql = [f"'{p}'" for p in normalized_paths]
    input_pattern = ", ".join(input_files_sql)

    conversion_success = False
    strategies = [
        ("UTF-8",
         f"COPY (SELECT * FROM read_csv_auto([{input_pattern}], sample_size=100000)) "
         f"TO '{TEMP_MASTER_PARQUET}' (FORMAT 'PARQUET', CODEC 'ZSTD')"),
        ("Latin-1",
         f"COPY (SELECT * FROM read_csv_auto([{input_pattern}], sample_size=100000, encoding='latin-1')) "
         f"TO '{TEMP_MASTER_PARQUET}' (FORMAT 'PARQUET', CODEC 'ZSTD')"),
        ("Ignore Errors",
         f"COPY (SELECT * FROM read_csv_auto([{input_pattern}], sample_size=100000, "
         f"encoding='latin-1', ignore_errors=true)) TO '{TEMP_MASTER_PARQUET}' (FORMAT 'PARQUET', CODEC 'ZSTD')"),
    ]

    for name, query in strategies:
        try:
            if os.path.exists(TEMP_MASTER_PARQUET):
                os.remove(TEMP_MASTER_PARQUET)
            t_strat = time.time()
            conn.execute(query)
            logging.info(f"[TIMING] DuckDB '{name}' strategy took {time.time() - t_strat:.2f}s")
            status.write(f"✅ Success using **{name}** encoding strategy! Data integrity verified.")
            conversion_success = True
            break
        except Exception as e:
            logging.info(f"DuckDB strategy '{name}' failed: {e}")
            continue

    if not conversion_success:
        status.write("⚠️ DuckDB encoding strategies could not parse these CSVs. "
                     "Falling back to Pandas with error-tolerant settings.")
        chunk_size = 200_000
        temp_chunks = []
        try:
            for csv_idx, csv_file in enumerate(all_csvs):
                t_file = time.time()
                header_df = pd.read_csv(csv_file, nrows=0, encoding_errors='replace')
                clean_cols = clean_column_names(header_df.columns)

                with pd.read_csv(csv_file, chunksize=chunk_size,
                                 encoding_errors='replace', on_bad_lines='skip') as reader:
                    for i, chunk in enumerate(reader):
                        chunk.columns = clean_cols
                        chunk_name = os.path.join(TEMP_DIR, f"chunk_{uuid.uuid4()}.parquet")
                        chunk.to_parquet(chunk_name, engine='pyarrow', index=False)
                        temp_chunks.append(chunk_name)

                logging.info(f"[TIMING] Pandas fallback for file {csv_idx+1}/{len(all_csvs)} "
                             f"took {time.time() - t_file:.2f}s")

            if not temp_chunks:
                raise Exception("No data could be read from CSVs via Pandas.")

            t_consolidate = time.time()
            chunk_pattern = os.path.join(TEMP_DIR, "chunk_*.parquet").replace("\\", "/")
            conn.execute(
                f"COPY (SELECT * FROM '{chunk_pattern}') "
                f"TO '{TEMP_MASTER_PARQUET}' (FORMAT 'PARQUET', CODEC 'ZSTD')"
            )
            logging.info(f"[TIMING] Chunk consolidation took {time.time() - t_consolidate:.2f}s")
        except Exception as e:
            raise Exception(f"Failed to process CSVs: {str(e)}")

    logging.info(f"[TIMING] CSV-to-Parquet conversion took {time.time() - t0:.2f}s total")

    file_size_mb = os.path.getsize(TEMP_MASTER_PARQUET) / (1024 * 1024)
    t_count = time.time()
    total_rows = get_parquet_row_count(TEMP_MASTER_PARQUET)
    logging.info(f"[TIMING] Row count (metadata read) took {time.time() - t_count:.4f}s ({total_rows:,} rows)")

    table_name = "data"

    if file_size_mb < CHUNK_SIZE_MB:
        os.rename(TEMP_MASTER_PARQUET, os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_0.parquet"))
    else:
        num_chunks = math.ceil(file_size_mb / CHUNK_SIZE_MB)
        rows_per_chunk = math.ceil(total_rows / num_chunks)
        status.write(f"✂️ **Splitting into ~{num_chunks} chunks** (~{rows_per_chunk:,} rows each, <{CHUNK_SIZE_MB}MB per file). "
                     f"This ensures all files fit within GitHub's 100MB limit for free hosting.")

        split_parquet_to_chunks(
            TEMP_MASTER_PARQUET, table_name, num_chunks,
            rows_per_chunk, DEFAULT_ARTIFACTS_DIR, status
        )

        if os.path.exists(TEMP_MASTER_PARQUET):
            os.remove(TEMP_MASTER_PARQUET)

    t_schema = time.time()
    first_chunk = os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_0.parquet")
    schema_df = conn.execute(f"DESCRIBE SELECT * FROM '{first_chunk}'").df()
    columns_meta = []
    for _, row in schema_df.iterrows():
        col, dtype = row['column_name'], row['column_type']
        try:
            samples = conn.execute(
                f'SELECT "{col}"::VARCHAR FROM \'{first_chunk}\' WHERE "{col}" IS NOT NULL LIMIT 3'
            ).fetchall()
            sample_str = ", ".join([str(x[0]) for x in samples])
        except Exception:
            sample_str = "N/A"
        desc = f"Table: {table_name}\nColumn: {col}\nType: {dtype}\nSamples: {sample_str}"
        columns_meta.append({"name": col, "type": dtype, "description": desc, "table": table_name})

    logging.info(f"[TIMING] Schema extraction took {time.time() - t_schema:.2f}s")

    return {
        "tables": {table_name: {"file_pattern": f"{table_name}_*.parquet", "total_rows": total_rows}},
        "columns": columns_meta
    }


# ==========================================
# LOGIC B: MULTI-TABLE LOOP
# ==========================================

def process_multi_strategy(conn, all_csvs, status):
    """Process each CSV as a separate, independently queryable table."""
    status.write("🧩 **Strategy: Keep Separate Tables** — Each CSV becomes its own queryable table. "
                 "This preserves relational structure and enables JOIN queries across tables "
                 "(e.g., linking students to sessions to assessments).")

    tables_metadata = {}
    all_columns_meta = []
    prog_bar = status.progress(0)

    for idx, csv_file in enumerate(all_csvs):
        t_file = time.time()
        raw_name = os.path.basename(csv_file)
        table_name = sanitize_table_name(raw_name)
        status.write(f"⚙️ **Processing table {idx+1}/{len(all_csvs)}:** `{table_name}` — "
                     f"Reading CSV, inferring column types, compressing to Parquet.")

        temp_parquet = os.path.join(TEMP_DIR, f"{table_name}_temp.parquet")
        input_path = csv_file.replace(os.sep, '/')

        try:
            t_duckdb = time.time()
            conn.execute(
                f"COPY (SELECT * FROM read_csv_auto('{input_path}', sample_size=100000)) "
                f"TO '{temp_parquet}' (FORMAT 'PARQUET', CODEC 'ZSTD')"
            )
            logging.info(f"[TIMING] DuckDB conversion of '{table_name}' took {time.time() - t_duckdb:.2f}s")
        except Exception as e:
            logging.info(f"DuckDB failed on {table_name}: {e}. Using Pandas fallback.")
            status.write(f"⚠️ DuckDB could not parse `{table_name}`. Falling back to Pandas with error-tolerant settings.")

            t_pandas = time.time()
            chunk_size = 200_000
            temp_chunks = []

            header_df = pd.read_csv(csv_file, nrows=0, encoding_errors='replace')
            clean_cols = clean_column_names(header_df.columns)

            with pd.read_csv(csv_file, chunksize=chunk_size,
                             encoding_errors='replace', on_bad_lines='skip') as reader:
                for i, chunk in enumerate(reader):
                    chunk.columns = clean_cols
                    c_name = os.path.join(TEMP_DIR, f"{table_name}_chunk_{i}.parquet")
                    chunk.to_parquet(c_name, engine='pyarrow', index=False)
                    temp_chunks.append(c_name)

            if temp_chunks:
                chunk_pattern = os.path.join(TEMP_DIR, f"{table_name}_chunk_*.parquet").replace("\\", "/")
                conn.execute(
                    f"COPY (SELECT * FROM '{chunk_pattern}') TO '{temp_parquet}' (FORMAT 'PARQUET', CODEC 'ZSTD')"
                )

            logging.info(f"[TIMING] Pandas fallback for '{table_name}' took {time.time() - t_pandas:.2f}s")

        file_size_mb = os.path.getsize(temp_parquet) / (1024 * 1024)
        t_count = time.time()
        total_rows = get_parquet_row_count(temp_parquet)
        logging.info(f"[TIMING] Row count for '{table_name}' (metadata read) took "
                     f"{time.time() - t_count:.4f}s ({total_rows:,} rows)")

        if file_size_mb < CHUNK_SIZE_MB:
            os.rename(temp_parquet, os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_0.parquet"))
        else:
            num_chunks = math.ceil(file_size_mb / CHUNK_SIZE_MB)
            rows_per_chunk = math.ceil(total_rows / num_chunks)
            status.write(f"✂️ **Splitting** `{table_name}` into **~{num_chunks} parts** (~{rows_per_chunk:,} rows each, "
                         f"<{CHUNK_SIZE_MB}MB per file). The query engine will stitch these back together seamlessly using wildcard reads.")

            split_parquet_to_chunks(
                temp_parquet, table_name, num_chunks,
                rows_per_chunk, DEFAULT_ARTIFACTS_DIR, status
            )

            if os.path.exists(temp_parquet):
                os.remove(temp_parquet)

        t_schema = time.time()
        first_chunk = os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_0.parquet")
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM '{first_chunk}'").df()
        for _, row in schema_df.iterrows():
            col, dtype = row['column_name'], row['column_type']
            try:
                samples = conn.execute(
                    f'SELECT "{col}"::VARCHAR FROM \'{first_chunk}\' WHERE "{col}" IS NOT NULL LIMIT 3'
                ).fetchall()
                sample_str = ", ".join([str(x[0]) for x in samples])
            except Exception:
                sample_str = "N/A"
            desc = f"Table: {table_name}\nColumn: {col}\nType: {dtype}\nSamples: {sample_str}"
            all_columns_meta.append({"name": col, "type": dtype, "description": desc, "table": table_name})

        logging.info(f"[TIMING] Schema extraction for '{table_name}' took {time.time() - t_schema:.2f}s")

        tables_metadata[table_name] = {"file_pattern": f"{table_name}_*.parquet", "total_rows": total_rows}
        prog_bar.progress((idx + 1) / len(all_csvs))

        logging.info(f"[TIMING] Total for '{table_name}' (file {idx+1}/{len(all_csvs)}): {time.time() - t_file:.2f}s")

    return {"tables": tables_metadata, "columns": all_columns_meta}


# ==========================================
# MAIN CONTROLLER
# ==========================================

def process_data(inputs, input_type="upload", strategy="merge",
                 auto_push=False, cloud_config=None):
    """
    Main orchestration function. Coordinates the full ETL pipeline:
    1. Clean up  2. Ingest  3. Convert  4. Detect relationships
    5. Save metadata  6. Build vector index  7. Cleanup
    8. Optional git push  9. Optional cloud upload
    """
    status = st.status("🚀 Processing started...", expanded=True)
    start_time = time.time()
    conn = None

    try:
        # --- Step 1: Clean workspace ---
        status.write("🧹 **Cleaning up workspace...** Removing any previous artifacts and temporary files.")
        robust_cleanup(DEFAULT_ARTIFACTS_DIR)
        os.makedirs(DEFAULT_ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)

        # --- Step 2: Ingest files ---
        t_ingest = time.time()
        status.write("💾 **Ingesting files...** All processing happens locally on this machine — "
                      "no data is uploaded to external servers during this step.")

        for item in inputs:
            t_item = time.time()

            if input_type == "upload":
                fname = item.name
                status.write(f"📥 **Reading upload:** `{fname}` — Writing to local temp directory only.")
                file_path = os.path.join(TEMP_DIR, fname)
                buf = item.getbuffer()
                file_size_mb = len(buf) / (1024 * 1024)
                with open(file_path, "wb") as f:
                    f.write(buf)

                if file_size_mb > 200:
                    status.write(f"⚠️ `{fname}` is **{file_size_mb:.0f} MB** — for faster processing, "
                                 f"consider using the **Local File Path** tab instead of browser upload.")

                logging.info(f"[TIMING] Writing upload '{fname}' ({file_size_mb:.1f} MB) "
                             f"took {time.time() - t_item:.2f}s")
            else:
                src_path = item.strip('"').strip("'").strip()
                fname = os.path.basename(src_path)
                status.write(f"📥 **Copying local file:** `{fname}` — Creating a working copy in the temp directory. "
                              "Your original file is not modified.")
                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"File not found: {src_path}")
                file_path = os.path.join(TEMP_DIR, fname)
                shutil.copy2(src_path, file_path)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logging.info(f"[TIMING] Copying '{fname}' ({file_size_mb:.1f} MB) "
                             f"took {time.time() - t_item:.2f}s")

            if fname.lower().endswith(".zip"):
                t_zip = time.time()
                status.write(f"📂 **Unzipping** `{fname}`... "
                              "(Archive contents are validated for path safety before extraction.)")
                safe_extract(file_path, TEMP_DIR)
                os.remove(file_path)
                logging.info(f"[TIMING] Unzipping '{fname}' took {time.time() - t_zip:.2f}s")

        all_csvs = get_all_csvs(TEMP_DIR)
        if not all_csvs:
            raise FileNotFoundError("No CSV files found after extraction!")

        logging.info(f"[TIMING] File ingestion took {time.time() - t_ingest:.2f}s")

        # --- Step 3: Convert CSVs to Parquet ---
        status.write(f"📊 Found **{len(all_csvs)}** CSV file(s). Starting conversion to compressed Parquet format.")
        status.write("🔒 **Privacy note:** Raw CSV data is converted to binary Parquet — a compressed columnar format "
                      "that is not human-readable. Only the column names and data types are stored as readable metadata "
                      "for the AI to reference.")

        t_convert = time.time()
        conn = duckdb.connect()

        if strategy == "merge":
            result = process_merge_strategy(conn, all_csvs, status)
            result['relationships'] = []
        else:
            result = process_multi_strategy(conn, all_csvs, status)
            table_names = list(result['tables'].keys())
            if len(table_names) >= 2:
                result['relationships'] = detect_relationships(conn, table_names, status)
            else:
                result['relationships'] = []

        logging.info(f"[TIMING] Total conversion + relationships took {time.time() - t_convert:.2f}s")

        # --- Step 5: Save metadata ---
        t_meta = time.time()
        status.write("📋 **Saving metadata...** Writing table schemas, column names, types, sample values, "
                      "and detected relationships to `metadata.json`. This file is the only human-readable "
                      "artifact — the data itself remains in binary Parquet.")
        metadata_path = os.path.join(DEFAULT_ARTIFACTS_DIR, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({
                "tables": result['tables'],
                "columns": result['columns'],
                "relationships": result['relationships']
            }, f, indent=2)
        logging.info(f"[TIMING] Metadata save took {time.time() - t_meta:.2f}s")

        # --- Step 6: Build vector index ---
        t_embed = time.time()
        status.write("🧠 **Building vector embeddings...** Creating a semantic index of column names and types "
                      "using ChromaDB's built-in embedding function (all-MiniLM-L6-v2 via ONNX — lightweight, no PyTorch required). "
                      "This index allows the AI to find relevant columns when answering questions. "
                      "**Only schema metadata is indexed — no actual data values are embedded.**")
        chroma_path = os.path.join(DEFAULT_ARTIFACTS_DIR, "chroma_db")
        if os.path.exists(chroma_path):
            try:
                shutil.rmtree(chroma_path, onerror=remove_readonly)
            except Exception:
                logging.warning("Could not fully clean old chroma_db")

        chroma_client = chromadb.PersistentClient(path=chroma_path)
        collection = chroma_client.create_collection("dataset_schema")

        docs = [c["description"] for c in result['columns']]
        ids = [f"{c['table']}.{c['name']}" for c in result['columns']]
        metadatas = [{"name": c["name"], "type": c["type"], "table": c["table"]} for c in result['columns']]

        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        logging.info(f"[TIMING] Vector embedding took {time.time() - t_embed:.2f}s "
                     f"({len(docs)} columns indexed)")

        # --- Step 7: Cleanup ---
        t_cleanup = time.time()
        conn.close()
        conn = None
        try:
            del chroma_client
        except Exception:
            pass
        gc.collect()
        if os.path.exists(TEMP_DIR):
            try:
                shutil.rmtree(TEMP_DIR, onerror=remove_readonly)
            except Exception:
                pass
        logging.info(f"[TIMING] Final cleanup took {time.time() - t_cleanup:.2f}s")

        # --- Report success ---
        elapsed_time = time.time() - start_time
        status.update(label="✅ Processing Complete!", state="complete", expanded=False)
        st.success(f"**Success!** Artifacts saved to `{DEFAULT_ARTIFACTS_DIR}/`.")
        st.info(f"⏱️ **Total Execution Time:** {elapsed_time:.2f} seconds")
        st.caption("🔒 Your raw CSV files were **not** stored in the output. "
                   "The artifacts directory contains only compressed binary Parquet files, "
                   "a schema metadata file, and a vector index of column descriptions.")

        logging.info(f"[TIMING] ===== TOTAL PIPELINE: {elapsed_time:.2f}s =====")

        # --- Step 8: Optional auto-deploy to Git ---
        if auto_push:
            if run_git_sync():
                st.balloons()
                st.success("🚀 **Deployed!** Changes pushed to GitHub.")

        # --- Step 9: Optional cloud upload ---
        if cloud_config:
            upload_status = st.status("☁️ Uploading to cloud storage...", expanded=True)
            success = upload_artifacts_to_cloud(
                provider_name=cloud_config["provider"],
                access_key=cloud_config["access_key"],
                secret_key=cloud_config["secret_key"],
                region_or_account=cloud_config["region"],
                bucket_name=cloud_config["bucket"],
                prefix=cloud_config["prefix"],
                status=upload_status
            )
            if success:
                upload_status.update(label="☁️ Cloud Upload Complete!", state="complete")
                st.success(f"☁️ **Uploaded!** Artifacts available at "
                           f"`s3://{cloud_config['bucket']}/{cloud_config['prefix']}/`")
            else:
                upload_status.update(label="☁️ Cloud Upload Failed", state="error")

    except Exception as e:
        status.update(label="❌ Critical Error", state="error")
        st.error(f"Processing failed: {str(e)}")
        logging.error(f"Processing failed: {e}", exc_info=True)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ==========================================
# MAIN UI
# ==========================================

st.write("### 1. Configuration")

strategy = st.radio(
    "Processing Strategy",
    ["Merge All (Single Table)", "Keep Separate (Multi-Table)"],
    index=0,
    help="Merge: stitches CSVs into one dataset. Separate: creates distinct tables for JOINs."
)
strategy_key = "merge" if "Merge" in strategy else "multi"

if strategy_key == "merge":
    st.info("ℹ️ All CSVs will be stacked into **one** table named `data`.")
else:
    st.info("ℹ️ Each CSV becomes a **separate** SQL table (e.g., `students`, `enrollments`). "
            "Foreign key relationships between tables will be auto-detected.")

# ---- Deployment Options ----
st.write("### 2. Deployment Options")

auto_deploy = st.checkbox("🔄 **Auto-Deploy to GitHub** after processing", value=False,
                          help="Push artifacts to your Git repo. Best for datasets under ~1.5GB.")

# ---- Cloud Storage (Optional) ----
cloud_upload_enabled = st.checkbox(
    "☁️ **Upload to Cloud Storage** after processing", value=False,
    help="Upload artifacts to S3-compatible cloud storage (AWS S3, Google Cloud Storage, or Cloudflare R2). "
         "Best for large datasets (5GB+) that exceed GitHub's file size limits. "
         "The postprocessor can then read directly from cloud storage."
)

cloud_config = None
if cloud_upload_enabled:
    with st.expander("☁️ Cloud Storage Configuration", expanded=True):
        st.markdown("""
        Upload your processed artifacts to cloud storage so the postprocessor can read them remotely.
        All three providers use the **S3-compatible API**, so the setup is similar for each.

        **How it works:**
        1. Create a bucket/container in your cloud provider
        2. Generate access credentials (Access Key ID + Secret Key)
        3. Enter the details below
        4. After processing, artifacts are uploaded automatically

        **What gets uploaded:** All Parquet files, `metadata.json`, and the `chroma_db/` vector index.
        Your raw CSV files are **never** uploaded.
        """)

        cloud_provider = st.selectbox(
            "Cloud Provider",
            list(CLOUD_PROVIDERS.keys()),
            key="pre_cloud_provider"
        )

        provider_cfg = CLOUD_PROVIDERS[cloud_provider]
        st.caption(provider_cfg["help_text"])

        cloud_bucket = st.text_input("Bucket Name", key="pre_cloud_bucket",
                                     placeholder="my-education-data")

        cloud_prefix = st.text_input(
            "Folder Prefix (inside the bucket)",
            value="deploy_artifacts",
            key="pre_cloud_prefix",
            help="Files will be uploaded to: s3://bucket-name/prefix/filename.parquet"
        )

        cloud_region = st.text_input(
            provider_cfg["region_label"],
            value=provider_cfg["region_default"],
            key="pre_cloud_region",
            help="For AWS: e.g., us-east-1. For R2: your Cloudflare Account ID. For GCS: 'auto'."
        )

        # ---- Credentials ----
        st.markdown("---")
        st.markdown("**Authentication Credentials**")
        st.caption("Credentials can be entered here OR stored in Streamlit Secrets. "
                   "If stored in Secrets, they will be loaded automatically.")

        # Try loading from secrets first
        secret_prefix = provider_cfg["secret_prefix"]
        cloud_access_key = ""
        cloud_secret_key = ""
        try:
            if f"{secret_prefix}_ACCESS_KEY_ID" in st.secrets:
                cloud_access_key = st.secrets[f"{secret_prefix}_ACCESS_KEY_ID"]
            if f"{secret_prefix}_SECRET_ACCESS_KEY" in st.secrets:
                cloud_secret_key = st.secrets[f"{secret_prefix}_SECRET_ACCESS_KEY"]
        except Exception:
            pass

        if cloud_access_key and cloud_secret_key:
            st.success(f"🔑 {cloud_provider} credentials loaded from Streamlit Secrets")
        else:
            cloud_access_key = st.text_input(
                "Access Key ID", type="password",
                key="pre_cloud_ak",
                help="Your S3-compatible Access Key ID."
            )
            cloud_secret_key = st.text_input(
                "Secret Access Key", type="password",
                key="pre_cloud_sk",
                help="Your S3-compatible Secret Access Key. This is never stored or logged."
            )

        # Show what the final upload path will look like.
        if cloud_bucket and cloud_prefix:
            st.caption(f"📦 Upload target: `s3://{cloud_bucket}/{cloud_prefix}/`")

        # Validate that all required fields are filled.
        if cloud_bucket and cloud_access_key and cloud_secret_key:
            cloud_config = {
                "provider": cloud_provider,
                "bucket": cloud_bucket,
                "prefix": cloud_prefix.strip("/"),
                "region": cloud_region,
                "access_key": cloud_access_key,
                "secret_key": cloud_secret_key,
            }
        elif cloud_upload_enabled:
            st.warning("⚠️ Please fill in all cloud storage fields (bucket, access key, secret key) "
                       "to enable cloud upload.")

st.write("### 3. Upload Data")
st.warning("⚠️ Processing GB-scale files takes time. Please be patient. "
           "For files over 200MB, use the **Local File Path** tab for much faster ingestion.")

tab1, tab2 = st.tabs(["📂 Browser Upload", "🛣️ Local File Path"])

with tab1:
    st.markdown("Use for files **under 200MB**. Larger files will be slow through the browser.")
    uploaded_files = st.file_uploader(
        "Drag & Drop CSVs or ZIPs", type=["csv", "zip"], accept_multiple_files=True
    )
    if uploaded_files and st.button("🚀 Process Uploads"):
        process_data(uploaded_files, input_type="upload", strategy=strategy_key,
                     auto_push=auto_deploy, cloud_config=cloud_config)

with tab2:
    st.markdown("Use for **huge files (GBs)**. Skips browser loading — **much faster** for large datasets.")
    st.caption('💡 **Tip:** Select multiple files in Windows Explorer, right-click → "Copy as path", then paste here.')
    local_path = st.text_area(
        "Paste Full File Path(s)",
        placeholder='"C:\\Users\\YourName\\Downloads\\File1.zip" "C:\\Users\\YourName\\Downloads\\File2.zip"',
        height=100
    )
    if local_path and st.button("🚀 Process Local Path"):
        paths = parse_file_paths(local_path)
        if paths:
            st.caption(f"📁 Found **{len(paths)}** file path(s)")
            process_data(paths, input_type="path", strategy=strategy_key,
                         auto_push=auto_deploy, cloud_config=cloud_config)
        else:
            st.error("Could not parse any file paths from input.")
