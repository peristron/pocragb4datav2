import datetime
import gc
import io
import json
import logging
import math
import os
import pathlib
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
from openai import OpenAI


os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


APP_TITLE = "Direction 2: AI Data Planner"
APP_SUBTITLE = "Profile -> plan -> query -> critique for sanitized or dummy datasets"
SESSION_ROOT = os.path.join(tempfile.gettempdir(), "streamlit_direction2")
CHUNK_SIZE_MB = 64
UPLOAD_WARNING_MB = 400
HARD_ROW_LIMIT = 1000
MAX_ROWS_FOR_SUMMARY = 50
DEFAULT_PROVIDER_NAME = "DeepSeek"

PROVIDER_CONFIG = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "secret_key": "DEEPSEEK_API_KEY",
    },
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
}

DANGEROUS_KEYWORDS = [
    "COPY",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "INSTALL",
    "LOAD",
    "ATTACH",
    "DETACH",
    "EXPORT",
    "IMPORT",
    "CALL",
    "EXECUTE",
    "SET",
    "PRAGMA",
    "HTTPFS",
    "VACUUM",
    "CHECKPOINT",
    "GRANT",
    "REVOKE",
    "TRUNCATE",
    "MERGE",
    "PREPARE",
    "READ_CSV",
    "READ_CSV_AUTO",
    "READ_TEXT",
    "READ_JSON",
    "READ_JSON_AUTO",
    "READ_BLOB",
    "READ_PARQUET_SCHEMA",
    "GLOB",
    "PARQUET_SCAN",
]

PII_COLUMN_PATTERNS = [
    r"(?i)\bname\b",
    r"(?i)\bfirst.?name\b",
    r"(?i)\blast.?name\b",
    r"(?i)\bfull.?name\b",
    r"(?i)\bemail\b",
    r"(?i)\bphone\b",
    r"(?i)\baddress\b",
    r"(?i)\bssn\b",
    r"(?i)social.?sec",
    r"(?i)\bdob\b",
    r"(?i)birth.?date",
    r"(?i)date.?of.?birth",
    r"(?i)\bpassword\b",
    r"(?i)\blogin\b",
    r"(?i)\busername\b",
    r"(?i)\bstudent.?id\b",
    r"(?i)\bemployee.?id\b",
]

st.set_page_config(page_title=APP_TITLE, layout="wide")


def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def robust_rmtree(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path, onerror=remove_readonly)
        except Exception:
            logging.warning("Could not fully remove %s", path)


def ensure_session_state():
    defaults = {
        "session_id": uuid.uuid4().hex,
        "messages": [{"role": "assistant", "content": "Upload sanitized data, then ask a question."}],
        "dataset_ready": False,
        "artifacts_dir": "",
        "bundle_dir": "",
        "metadata": None,
        "dataset_profile": None,
        "starter_questions": [],
        "pii_redaction": True,
        "processing_summary": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_session_paths():
    root = os.path.join(SESSION_ROOT, st.session_state.session_id)
    bundle_dir = os.path.join(root, "bundle")
    upload_dir = os.path.join(root, "uploads")
    artifacts_dir = os.path.join(bundle_dir, "artifacts")
    return root, bundle_dir, upload_dir, artifacts_dir


def reset_dataset_state():
    root, _, _, _ = build_session_paths()
    robust_rmtree(root)
    st.session_state.dataset_ready = False
    st.session_state.bundle_dir = ""
    st.session_state.artifacts_dir = ""
    st.session_state.metadata = None
    st.session_state.dataset_profile = None
    st.session_state.starter_questions = []
    st.session_state.processing_summary = {}
    st.session_state.messages = [{"role": "assistant", "content": "Upload sanitized data, then ask a question."}]
    execute_validated_sql.clear()


def sanitize_table_name(filename):
    name = os.path.splitext(filename)[0]
    clean = re.sub(r"[^a-zA-Z0-9]", "_", name).lower().strip("_")
    if not clean:
        clean = "table"
    if clean[0].isdigit():
        clean = "t_" + clean
    return clean


def clean_column_names(columns):
    return [str(col).strip().replace('"', "") for col in columns]


def normalize_identifier(value):
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def singularize_name(value):
    clean = normalize_identifier(value)
    if clean.endswith("ies") and len(clean) > 3:
        return clean[:-3] + "y"
    if clean.endswith("ses") and len(clean) > 3:
        return clean[:-2]
    if clean.endswith("s") and not clean.endswith("ss") and len(clean) > 1:
        return clean[:-1]
    return clean


def candidate_entity_names(table_name):
    normalized = normalize_identifier(table_name)
    singular = singularize_name(table_name)
    names = {normalized, singular}
    if singular.endswith("history"):
        names.add(singular.replace("history", ""))
    if singular.endswith("enrollment"):
        names.add("user")
    return {name for name in names if name}


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def get_all_csvs(root_dir):
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file_name))
    return sorted(csv_files)


def get_parquet_row_count(parquet_path):
    return pq.ParquetFile(parquet_path).metadata.num_rows


def safe_extract(zip_path, target_dir):
    target = pathlib.Path(target_dir).resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_path = (target / member).resolve()
            if not str(member_path).startswith(str(target)):
                raise ValueError(f"Security alert: ZIP member escapes target directory: {member}")
        zf.extractall(target_dir)


def inspect_csv_headers(csv_paths):
    header_map = {}
    signature_map = {}
    for csv_path in csv_paths:
        try:
            header_df = pd.read_csv(csv_path, nrows=0, encoding_errors="replace")
            clean_headers = tuple(clean_column_names(header_df.columns))
        except Exception:
            clean_headers = tuple()
        header_map[os.path.basename(csv_path)] = clean_headers
        signature_map.setdefault(clean_headers, []).append(os.path.basename(csv_path))
    return header_map, signature_map


def redact_pii(df):
    pii_cols = []
    for col in df.columns:
        for pattern in PII_COLUMN_PATTERNS:
            if re.search(pattern, col):
                pii_cols.append(col)
                break
    if not pii_cols:
        return df, []
    redacted = df.copy()
    for col in pii_cols:
        redacted[col] = "[REDACTED]"
    return redacted, pii_cols


def split_parquet_to_chunks(source_parquet, table_name, rows_per_chunk, output_dir, status):
    pf = pq.ParquetFile(source_parquet)
    schema = pf.schema_arrow
    chunk_idx = 0
    buffered_tables = []
    buffered_rows = 0

    for batch in pf.iter_batches(batch_size=min(rows_per_chunk, 100_000)):
        table = pa.Table.from_batches([batch], schema=schema)
        offset = 0
        while offset < table.num_rows:
            take_rows = min(rows_per_chunk - buffered_rows, table.num_rows - offset)
            buffered_tables.append(table.slice(offset, take_rows))
            buffered_rows += take_rows
            offset += take_rows
            if buffered_rows >= rows_per_chunk:
                out_path = os.path.join(output_dir, f"{table_name}_{chunk_idx}.parquet")
                pq.write_table(pa.concat_tables(buffered_tables), out_path, compression="zstd")
                status.write(f"   • Wrote chunk `{os.path.basename(out_path)}`")
                buffered_tables = []
                buffered_rows = 0
                chunk_idx += 1

    if buffered_tables:
        out_path = os.path.join(output_dir, f"{table_name}_{chunk_idx}.parquet")
        pq.write_table(pa.concat_tables(buffered_tables), out_path, compression="zstd")
        status.write(f"   • Wrote chunk `{os.path.basename(out_path)}`")


def extract_schema_metadata(conn, first_chunk_path, table_name):
    schema_df = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{first_chunk_path.replace(os.sep, '/')}')"
    ).df()
    columns_meta = []
    for _, row in schema_df.iterrows():
        col_name = row["column_name"]
        dtype = row["column_type"]
        try:
            samples = conn.execute(
                f"""
                SELECT "{col_name}"::VARCHAR
                FROM read_parquet('{first_chunk_path.replace(os.sep, '/')}')
                WHERE "{col_name}" IS NOT NULL
                LIMIT 3
                """
            ).fetchall()
            sample_str = ", ".join(str(item[0]) for item in samples)
        except Exception:
            sample_str = "N/A"
        columns_meta.append(
            {
                "table": table_name,
                "name": col_name,
                "type": dtype,
                "description": f"Table: {table_name}\nColumn: {col_name}\nType: {dtype}\nSamples: {sample_str}",
            }
        )
    return columns_meta


def detect_relationships(conn, tables_metadata, artifacts_dir):
    normalized_columns = {}
    relationships = []
    seen = set()

    for table_name in tables_metadata:
        first_chunk = os.path.join(artifacts_dir, f"{table_name}_0.parquet").replace("\\", "/")
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{first_chunk}')").df()
        normalized_columns[table_name] = {
            normalize_identifier(col): col for col in schema_df["column_name"].tolist()
        }

    table_names = list(normalized_columns.keys())
    for left_table in table_names:
        for right_table in table_names:
            if left_table == right_table:
                continue
            left_entities = candidate_entity_names(left_table)
            left_norm_map = normalized_columns[left_table]
            right_norm_map = normalized_columns[right_table]

            for right_norm, right_col in right_norm_map.items():
                if not right_norm.endswith("id"):
                    continue
                if right_norm[:-2] in left_entities:
                    for pk_norm in ["id"] + [f"{entity}id" for entity in left_entities]:
                        if pk_norm in left_norm_map:
                            pk_col = left_norm_map[pk_norm]
                            rel_key = (right_table, right_col, left_table, pk_col)
                            if rel_key not in seen:
                                seen.add(rel_key)
                                relationships.append(
                                    {
                                        "from_table": right_table,
                                        "from_column": right_col,
                                        "to_table": left_table,
                                        "to_column": pk_col,
                                    }
                                )
                            break
    return relationships


def analyze_table_overlaps(metadata):
    normalized_columns = {}
    for column in metadata.get("columns", []):
        normalized_columns.setdefault(column["table"], {})[normalize_identifier(column["name"])] = column["name"]

    overlaps = []
    table_names = sorted(normalized_columns.keys())
    for idx, left_table in enumerate(table_names):
        for right_table in table_names[idx + 1 :]:
            left_norm = normalized_columns.get(left_table, {})
            right_norm = normalized_columns.get(right_table, {})
            common_norms = sorted(set(left_norm.keys()).intersection(set(right_norm.keys())))
            if not common_norms:
                continue
            shared_columns = []
            shared_key_columns = []
            for common_norm in common_norms:
                left_col = left_norm[common_norm]
                right_col = right_norm[common_norm]
                shared_columns.append(
                    {
                        "normalized_name": common_norm,
                        "left_column": left_col,
                        "right_column": right_col,
                    }
                )
                if common_norm == "id" or common_norm.endswith("id") or common_norm.endswith("key"):
                    shared_key_columns.append(
                        {
                            "normalized_name": common_norm,
                            "left_column": left_col,
                            "right_column": right_col,
                        }
                    )
            overlaps.append(
                {
                    "left_table": left_table,
                    "right_table": right_table,
                    "shared_column_count": len(shared_columns),
                    "shared_key_count": len(shared_key_columns),
                    "shared_columns": shared_columns,
                    "shared_key_columns": shared_key_columns,
                }
            )
    return overlaps


def infer_table_kind(table_name, column_names):
    name = table_name.lower()
    norm_cols = {normalize_identifier(col) for col in column_names}
    id_like = [col for col in norm_cols if col.endswith("id")]
    time_like = [col for col in norm_cols if "date" in col or "time" in col]

    if "discussion" in name or "session" in name or "history" in name:
        return "event"
    if "enrollment" in name or (len(id_like) >= 2 and "rolename" in norm_cols):
        return "bridge"
    if "content" in name:
        return "content"
    if "user" in name or ("userid" in norm_cols and len(time_like) < 4):
        return "entity"
    if len(id_like) >= 2 and len(time_like) >= 1:
        return "event"
    return "table"


def infer_table_description(table_name, column_names):
    kind = infer_table_kind(table_name, column_names)
    descriptions = {
        "entity": "An entity-style table that likely represents people or accounts.",
        "bridge": "A bridge or membership table that links users, roles, or org units.",
        "event": "An event/activity table with actions, posts, sessions, or time-based records.",
        "content": "A content/object table with type, status, date, or hierarchy metadata.",
        "table": "A processed table from the uploaded dataset.",
    }
    return descriptions.get(kind, "A processed table from the uploaded dataset.")


def build_dataset_profile(metadata):
    table_columns = {}
    column_types = {}
    for column in metadata.get("columns", []):
        table_columns.setdefault(column["table"], []).append(column["name"])
        column_types.setdefault(column["table"], {})[column["name"]] = column["type"]

    tables = []
    for table_name, table_info in metadata.get("tables", {}).items():
        columns = table_columns.get(table_name, [])
        norm_cols = [normalize_identifier(col) for col in columns]
        primary_key_candidates = []
        foreign_key_candidates = []
        for col in columns:
            norm = normalize_identifier(col)
            if norm == "id" or norm == f"{singularize_name(table_name)}id":
                primary_key_candidates.append(col)
            if norm.endswith("id") and col not in primary_key_candidates:
                foreign_key_candidates.append(col)

        tables.append(
            {
                "table_name": table_name,
                "row_count": table_info.get("total_rows", 0),
                "column_count": len(columns),
                "table_kind": infer_table_kind(table_name, columns),
                "appears_to_represent": infer_table_description(table_name, columns),
                "primary_key_candidates": primary_key_candidates[:5],
                "foreign_key_candidates": foreign_key_candidates[:10],
                "columns": [{"name": name, "type": column_types[table_name][name]} for name in columns],
            }
        )

    return {
        "tables": tables,
        "relationships": metadata.get("relationships", []),
        "table_overlaps": metadata.get("table_overlaps", []),
    }


def process_merge_strategy(conn, csv_paths, artifacts_dir, temp_dir, status):
    temp_master = os.path.join(temp_dir, "master.parquet")
    normalized_paths = [path.replace(os.sep, "/") for path in csv_paths]
    input_files_sql = ", ".join(f"'{path}'" for path in normalized_paths)
    strategies = [
        (
            "UTF-8",
            f"""
            COPY (
                SELECT * FROM read_csv_auto([{input_files_sql}], sample_size=100000)
            ) TO '{temp_master.replace(os.sep, "/")}' (FORMAT 'PARQUET', CODEC 'ZSTD')
            """,
        ),
        (
            "Latin-1",
            f"""
            COPY (
                SELECT * FROM read_csv_auto([{input_files_sql}], sample_size=100000, encoding='latin-1')
            ) TO '{temp_master.replace(os.sep, "/")}' (FORMAT 'PARQUET', CODEC 'ZSTD')
            """,
        ),
    ]

    success = False
    for label, sql in strategies:
        try:
            if os.path.exists(temp_master):
                os.remove(temp_master)
            conn.execute(sql)
            status.write(f"✅ CSV parse succeeded with the {label} strategy.")
            success = True
            break
        except Exception as exc:
            logging.info("Merge strategy failed %s: %s", label, exc)

    if not success:
        raise ValueError("Merge mode could not parse these CSV files reliably.")

    table_name = "data"
    total_rows = get_parquet_row_count(temp_master)
    file_size_mb = os.path.getsize(temp_master) / (1024 * 1024)
    if file_size_mb < CHUNK_SIZE_MB:
        os.rename(temp_master, os.path.join(artifacts_dir, f"{table_name}_0.parquet"))
    else:
        chunk_count = max(1, math.ceil(file_size_mb / CHUNK_SIZE_MB))
        rows_per_chunk = max(1, math.ceil(total_rows / chunk_count))
        split_parquet_to_chunks(temp_master, table_name, rows_per_chunk, artifacts_dir, status)
        os.remove(temp_master)

    first_chunk = os.path.join(artifacts_dir, f"{table_name}_0.parquet")
    columns_meta = extract_schema_metadata(conn, first_chunk, table_name)
    return {
        "tables": {table_name: {"file_pattern": f"{table_name}_*.parquet", "total_rows": total_rows}},
        "columns": columns_meta,
        "relationships": [],
    }


def process_multi_strategy(conn, csv_paths, artifacts_dir, temp_dir, status):
    tables_metadata = {}
    all_columns_meta = []
    progress = status.progress(0.0)
    for idx, csv_file in enumerate(csv_paths, start=1):
        raw_name = os.path.basename(csv_file)
        table_name = sanitize_table_name(raw_name)
        temp_parquet = os.path.join(temp_dir, f"{table_name}_temp.parquet")
        input_path = csv_file.replace(os.sep, "/")
        status.write(f"⚙️ Processing table {idx}/{len(csv_paths)}: `{table_name}`")

        try:
            conn.execute(
                f"""
                COPY (
                    SELECT * FROM read_csv_auto('{input_path}', sample_size=100000)
                ) TO '{temp_parquet.replace(os.sep, "/")}' (FORMAT 'PARQUET', CODEC 'ZSTD')
                """
            )
        except Exception:
            header_df = pd.read_csv(csv_file, nrows=0, encoding_errors="replace")
            clean_headers = clean_column_names(header_df.columns)
            temp_chunks = []
            with pd.read_csv(csv_file, chunksize=200_000, encoding_errors="replace", on_bad_lines="skip") as reader:
                for chunk_idx, chunk in enumerate(reader):
                    chunk.columns = clean_headers
                    out_chunk = os.path.join(temp_dir, f"{table_name}_{chunk_idx}.parquet")
                    chunk.to_parquet(out_chunk, engine="pyarrow", index=False)
                    temp_chunks.append(out_chunk)
            if not temp_chunks:
                raise ValueError(f"No readable rows were found in {raw_name}.")
            chunk_pattern = os.path.join(temp_dir, f"{table_name}_*.parquet").replace(os.sep, "/")
            conn.execute(
                f"""
                COPY (
                    SELECT * FROM read_parquet('{chunk_pattern}')
                ) TO '{temp_parquet.replace(os.sep, "/")}' (FORMAT 'PARQUET', CODEC 'ZSTD')
                """
            )

        total_rows = get_parquet_row_count(temp_parquet)
        file_size_mb = os.path.getsize(temp_parquet) / (1024 * 1024)
        if file_size_mb < CHUNK_SIZE_MB:
            os.rename(temp_parquet, os.path.join(artifacts_dir, f"{table_name}_0.parquet"))
        else:
            chunk_count = max(1, math.ceil(file_size_mb / CHUNK_SIZE_MB))
            rows_per_chunk = max(1, math.ceil(total_rows / chunk_count))
            split_parquet_to_chunks(temp_parquet, table_name, rows_per_chunk, artifacts_dir, status)
            os.remove(temp_parquet)

        first_chunk = os.path.join(artifacts_dir, f"{table_name}_0.parquet")
        all_columns_meta.extend(extract_schema_metadata(conn, first_chunk, table_name))
        tables_metadata[table_name] = {"file_pattern": f"{table_name}_*.parquet", "total_rows": total_rows}
        progress.progress(idx / len(csv_paths))

    relationships = detect_relationships(conn, tables_metadata, artifacts_dir)
    return {"tables": tables_metadata, "columns": all_columns_meta, "relationships": relationships}


def process_uploaded_files(uploaded_files, strategy):
    if not uploaded_files:
        raise ValueError("Please upload at least one CSV or ZIP file.")

    root, bundle_dir, upload_dir, artifacts_dir = build_session_paths()
    robust_rmtree(root)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    status = st.status("🚀 Processing uploaded data...", expanded=True)
    start_time = time.time()
    conn = None
    try:
        total_uploaded_mb = 0.0
        for uploaded in uploaded_files:
            out_path = os.path.join(upload_dir, uploaded.name)
            buf = uploaded.getbuffer()
            total_uploaded_mb += len(buf) / (1024 * 1024)
            with open(out_path, "wb") as handle:
                handle.write(buf)
            status.write(f"📥 Saved upload `{uploaded.name}`")
            if uploaded.name.lower().endswith(".zip"):
                status.write(f"📂 Extracting `{uploaded.name}`")
                safe_extract(out_path, upload_dir)
                os.remove(out_path)

        csv_paths = get_all_csvs(upload_dir)
        if not csv_paths:
            raise ValueError("No CSV files were found after upload and extraction.")

        header_map, signature_map = inspect_csv_headers(csv_paths)
        if strategy == "merge" and len(signature_map) > 1:
            groups = []
            for files in list(signature_map.values())[:3]:
                groups.append(", ".join(files[:3]))
            raise ValueError(
                "Merge mode is not recommended for this upload because the CSV files do not share the same column structure. "
                "Please switch to 'Keep files as separate tables'. "
                f"Detected multiple header patterns such as: {' | '.join(groups)}"
            )

        if total_uploaded_mb > UPLOAD_WARNING_MB:
            status.write("⚠️ Large upload detected. Local execution may be more reliable for very large files.")

        conn = duckdb.connect()
        if strategy == "merge":
            result = process_merge_strategy(conn, csv_paths, artifacts_dir, upload_dir, status)
        else:
            result = process_multi_strategy(conn, csv_paths, artifacts_dir, upload_dir, status)

        metadata = {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "chunk_size_mb": CHUNK_SIZE_MB,
            "source_mode": "cloud_upload",
            "header_map": header_map,
            "tables": result["tables"],
            "columns": result["columns"],
            "relationships": result["relationships"],
        }
        metadata["table_overlaps"] = analyze_table_overlaps(metadata)
        metadata["dataset_profile"] = build_dataset_profile(metadata)

        with open(os.path.join(artifacts_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        st.session_state.dataset_ready = True
        st.session_state.bundle_dir = bundle_dir
        st.session_state.artifacts_dir = artifacts_dir
        st.session_state.metadata = metadata
        st.session_state.dataset_profile = metadata["dataset_profile"]
        st.session_state.starter_questions = []
        st.session_state.messages = [{"role": "assistant", "content": "Dataset processed. Ask a question."}]
        st.session_state.processing_summary = {
            "csv_count": len(csv_paths),
            "uploaded_mb": round(total_uploaded_mb, 1),
            "tables": len(metadata["tables"]),
            "columns": len(metadata["columns"]),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        execute_validated_sql.clear()
        status.update(label="✅ Processing complete", state="complete", expanded=False)
        return metadata
    except Exception as exc:
        status.update(label="❌ Processing failed", state="error", expanded=True)
        robust_rmtree(root)
        raise exc
    finally:
        if conn is not None:
            conn.close()
        gc.collect()


def build_table_inventory(metadata, artifacts_dir):
    lines = []
    for table_name, info in metadata.get("tables", {}).items():
        pattern = os.path.join(artifacts_dir, info["file_pattern"]).replace("\\", "/")
        lines.append(f"- {table_name}: read_parquet('{pattern}') | rows={info.get('total_rows', 0)}")
    return "\n".join(lines)


def sanitize_sql_for_display(sql_text, metadata, artifacts_dir):
    clean_sql = sql_text or ""
    for table_name, info in metadata.get("tables", {}).items():
        pattern = os.path.join(artifacts_dir, info["file_pattern"]).replace("\\", "/")
        clean_sql = clean_sql.replace(pattern, f"ARTIFACT::{table_name}")
    return clean_sql


def strip_leading_sql_comments(sql_text):
    if not sql_text:
        return sql_text
    cleaned = sql_text.lstrip()
    while True:
        if cleaned.startswith("--"):
            newline_index = cleaned.find("\n")
            if newline_index == -1:
                return ""
            cleaned = cleaned[newline_index + 1 :].lstrip()
            continue
        if cleaned.startswith("/*"):
            end_index = cleaned.find("*/")
            if end_index == -1:
                return ""
            cleaned = cleaned[end_index + 2 :].lstrip()
            continue
        break
    return cleaned


def _strip_markdown_sql(text):
    clean = (text or "").strip()
    if clean.startswith("```sql"):
        clean = clean[6:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


def extract_json_object(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def find_table_name_in_question(question, metadata):
    question_norm = normalize_identifier(question)
    best_match = None
    best_score = 0
    for table_name in metadata.get("tables", {}).keys():
        candidates = candidate_entity_names(table_name) | {normalize_identifier(table_name)}
        for candidate in candidates:
            if candidate and candidate in question_norm and len(candidate) > best_score:
                best_match = table_name
                best_score = len(candidate)
    return best_match


def find_explicit_table_mentions(question, metadata):
    question_norm = normalize_identifier(question)
    matches = []
    for table_name in metadata.get("tables", {}).keys():
        candidates = candidate_entity_names(table_name) | {normalize_identifier(table_name)}
        if any(candidate and candidate in question_norm for candidate in candidates):
            matches.append(table_name)
    return sorted(set(matches))


def build_columns_dataframe(metadata, table_name):
    rows = []
    for column in metadata.get("columns", []):
        if column["table"] == table_name:
            rows.append(
                {
                    "column_name": column["name"],
                    "data_type": column["type"],
                    "sample_values": column["description"].split("Samples: ", 1)[-1],
                }
            )
    return pd.DataFrame(rows)


def build_table_overview_rows(metadata):
    column_counts = {}
    for column in metadata.get("columns", []):
        column_counts[column["table"]] = column_counts.get(column["table"], 0) + 1
    rows = []
    for table_name, info in metadata.get("tables", {}).items():
        rows.append(
            {
                "table_name": table_name,
                "row_count": info.get("total_rows", 0),
                "column_count": column_counts.get(table_name, 0),
                "table_kind": infer_table_kind(table_name, list(column_counts.keys())),
                "appears_to_represent": infer_table_description(table_name, list(column_counts.keys())),
            }
        )
    return pd.DataFrame(rows)


def classify_question(question, metadata):
    q = question.lower()
    explicit_tables = find_explicit_table_mentions(question, metadata)
    metadata_markers = ["what tables", "which tables", "what columns", "which columns", "available in", "how many total rows", "shared columns", "overlap"]
    analysis_markers = [
        " then ",
        " before ",
        " using ",
        " together",
        " summary",
        " diagnostic",
        " percentage",
        " percent",
        " average",
        " order ",
        " exclude ",
        " compare",
        " group",
        " one row per",
        " posted",
        " include",
        " join",
        " metric",
        " metrics",
        "trend",
        "top ",
        "rate",
    ]
    metadata_score = sum(marker in q for marker in metadata_markers)
    analysis_score = sum(marker in q for marker in analysis_markers) + max(0, len(explicit_tables) - 1) * 2

    if analysis_score > metadata_score:
        return "analysis"
    if metadata_score > 0:
        return "metadata"
    return "analysis"


def handle_metadata_question(question, metadata):
    q = question.lower()
    table_name = find_table_name_in_question(question, metadata)

    if (("what tables" in q or "which tables" in q) and ("represent" in q or "included" in q)) or ("what tables are included" in q):
        df = build_table_overview_rows(metadata)
        answer = "Here are the processed tables and what they appear to represent."
        return {"dataframe": df, "answer": answer, "title": "Answered directly from dataset profile"}

    if "how many total rows" in q and "each table" in q:
        rows = [{"table_name": t, "row_count": info.get("total_rows", 0)} for t, info in metadata.get("tables", {}).items()]
        df = pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
        answer = "Row counts by table are shown below."
        return {"dataframe": df, "answer": answer, "title": "Answered directly from dataset profile"}

    if table_name and ("which columns" in q or "what columns" in q or "available in" in q):
        df = build_columns_dataframe(metadata, table_name)
        if not df.empty:
            return {
                "dataframe": df,
                "answer": f"The `{table_name}` table has {len(df)} columns. The table below lists each column, type, and sample values.",
                "title": f"Answered directly from metadata for `{table_name}`",
            }

    if ("common" in q or "shared" in q or "overlap" in q) and ("columns" in q or "tables" in q):
        rows = []
        for item in metadata.get("table_overlaps", []):
            rows.append(
                {
                    "left_table": item["left_table"],
                    "right_table": item["right_table"],
                    "shared_column_count": item["shared_column_count"],
                    "shared_key_count": item["shared_key_count"],
                    "shared_keys_preview": ", ".join(shared["left_column"] for shared in item.get("shared_key_columns", [])[:5]),
                }
            )
        return {
            "dataframe": pd.DataFrame(rows),
            "answer": "The table below shows which processed tables share columns or likely key fields.",
            "title": "Answered directly from overlap analysis",
        }

    return None


def get_overlap_relationship_hints(metadata):
    hints = []
    for item in metadata.get("table_overlaps", []):
        if item.get("shared_key_count", 0) > 0:
            key_labels = ", ".join(
                f"{shared['left_column']} / {shared['right_column']}" for shared in item.get("shared_key_columns", [])[:5]
            )
            hints.append(f"- {item['left_table']} and {item['right_table']} share likely key columns: {key_labels}")
    return hints


def build_relationship_context(metadata):
    relationships = [
        f"- {item['from_table']}.{item['from_column']} -> {item['to_table']}.{item['to_column']}"
        for item in metadata.get("relationships", [])
    ]
    overlap_hints = get_overlap_relationship_hints(metadata)
    blocks = []
    if relationships:
        blocks.append("KNOWN RELATIONSHIPS:\n" + "\n".join(relationships))
    if overlap_hints:
        blocks.append("OVERLAP HINTS:\n" + "\n".join(overlap_hints))
    return "\n\n".join(blocks)


def build_context_block(metadata, question):
    columns = metadata.get("columns", [])
    question_tokens = tokenize(question)
    scored = []
    for column in columns:
        score = 0
        score += len(question_tokens & tokenize(column["name"])) * 5
        score += len(question_tokens & tokenize(column["table"])) * 3
        score += len(question_tokens & tokenize(column["description"]))
        scored.append((score, column))
    selected = [item[1] for item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:16]]
    lines = ["RELEVANT SCHEMA:"]
    for item in selected:
        lines.append(
            f"- Table `{item['table']}` | Column `{item['name']}` | Type `{item['type']}` | Samples `{item['description'].split('Samples: ', 1)[-1]}`"
        )
    return "\n".join(lines)


def build_profile_prompt(dataset_profile):
    compact = {
        "tables": [
            {
                "table_name": table["table_name"],
                "row_count": table["row_count"],
                "table_kind": table["table_kind"],
                "appears_to_represent": table["appears_to_represent"],
                "primary_key_candidates": table["primary_key_candidates"],
                "foreign_key_candidates": table["foreign_key_candidates"][:6],
                "columns": table["columns"][:12],
            }
            for table in dataset_profile.get("tables", [])
        ],
        "relationships": dataset_profile.get("relationships", []),
        "table_overlaps": dataset_profile.get("table_overlaps", [])[:10],
    }
    return json.dumps(compact, indent=2)


def build_supplemental_table_review(metadata, table_names):
    if not table_names:
        return []

    profile_tables = {table["table_name"]: table for table in (metadata.get("dataset_profile", {}) or {}).get("tables", [])}
    overlap_items = metadata.get("table_overlaps", [])
    reviews = []

    for table_name in table_names:
        profile = profile_tables.get(table_name, {})
        related = []
        for item in overlap_items:
            if item["left_table"] == table_name or item["right_table"] == table_name:
                other = item["right_table"] if item["left_table"] == table_name else item["left_table"]
                shared_keys = ", ".join(shared["left_column"] for shared in item.get("shared_key_columns", [])[:4])
                related.append(
                    {
                        "related_table": other,
                        "shared_key_count": item.get("shared_key_count", 0),
                        "shared_column_count": item.get("shared_column_count", 0),
                        "shared_keys_preview": shared_keys,
                    }
                )

        reviews.append(
            {
                "table_name": table_name,
                "table_kind": profile.get("table_kind", "unknown"),
                "appears_to_represent": profile.get("appears_to_represent", "No profile description available."),
                "row_count": profile.get("row_count"),
                "primary_key_candidates": profile.get("primary_key_candidates", []),
                "foreign_key_candidates": profile.get("foreign_key_candidates", []),
                "related_tables": related[:5],
            }
        )

    return reviews


def make_fallback_plan(question, metadata):
    explicit_tables = find_explicit_table_mentions(question, metadata)
    if not explicit_tables:
        explicit_tables = list(metadata.get("tables", {}).keys())[:2]
    return {
        "question_type": "analysis",
        "required_tables": explicit_tables,
        "optional_tables_for_cursory_review": [],
        "likely_grain": "depends_on_question",
        "join_strategy": "Prefer detected relationships and shared key columns such as UserId or OrgUnitId.",
        "metrics": ["Answer the user question with careful use of distinct counts where appropriate."],
        "filters": [],
        "duplication_risks": ["Joins may multiply rows; consider pre-aggregation or COUNT(DISTINCT ...)."],
        "assumptions": [],
        "reasoning_notes": ["Fallback plan used because planner JSON could not be parsed."],
        "sufficiency_confidence": "medium",
        "omission_rationale": "",
    }


def generate_query_plan(question, dataset_profile, explicit_tables, client, model_name):
    profile_prompt = build_profile_prompt(dataset_profile)
    explicit_block = json.dumps(explicit_tables)
    prompt = f"""You are planning how to answer a question against a structured dataset.
Return JSON only with the following keys:
- question_type
- required_tables
- optional_tables_for_cursory_review
- likely_grain
- join_strategy
- metrics
- filters
- duplication_risks
- assumptions
- reasoning_notes
- sufficiency_confidence
- omission_rationale

Question:
{question}

Explicitly named tables:
{explicit_block}

Dataset profile:
{profile_prompt}

Rules:
1. If the user explicitly named tables, include them in required_tables unless clearly irrelevant.
2. If one explicitly named table is probably redundant for the main SQL, put it in optional_tables_for_cursory_review and explain why in omission_rationale.
3. Prefer exact shared keys and explicit relationship hints over loose joins.
4. Call out duplication risks whenever enrollments or event tables may multiply rows.
5. sufficiency_confidence should be low, medium, or high.
6. Be concrete and concise.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    parsed = extract_json_object(raw)
    if not parsed:
        return {}

    def _ensure_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        return [str(value)]

    normalized = dict(parsed)
    normalized["required_tables"] = _ensure_list(parsed.get("required_tables"))
    normalized["optional_tables_for_cursory_review"] = _ensure_list(parsed.get("optional_tables_for_cursory_review"))
    normalized["metrics"] = _ensure_list(parsed.get("metrics"))
    normalized["filters"] = _ensure_list(parsed.get("filters"))
    normalized["duplication_risks"] = _ensure_list(parsed.get("duplication_risks"))
    normalized["assumptions"] = _ensure_list(parsed.get("assumptions"))
    normalized["reasoning_notes"] = _ensure_list(parsed.get("reasoning_notes"))
    normalized["question_type"] = str(parsed.get("question_type", "analysis")).strip() or "analysis"
    normalized["likely_grain"] = str(parsed.get("likely_grain", "depends_on_question")).strip() or "depends_on_question"
    normalized["join_strategy"] = str(parsed.get("join_strategy", "")).strip()
    normalized["sufficiency_confidence"] = str(parsed.get("sufficiency_confidence", "medium")).strip().lower() or "medium"
    normalized["omission_rationale"] = str(parsed.get("omission_rationale", "")).strip()
    return normalized


def build_plan_markdown(plan):
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        return [str(value)]

    lines = []
    lines.append(f"- Question type: {plan.get('question_type', 'unknown')}")
    lines.append(f"- Required tables: {', '.join(_as_list(plan.get('required_tables'))) or 'n/a'}")
    lines.append(
        f"- Optional tables for cursory review: {', '.join(_as_list(plan.get('optional_tables_for_cursory_review'))) or 'n/a'}"
    )
    lines.append(f"- Likely grain: {plan.get('likely_grain', 'n/a')}")
    lines.append(f"- Join strategy: {plan.get('join_strategy', 'n/a')}")
    lines.append(f"- Sufficiency confidence: {plan.get('sufficiency_confidence', 'n/a')}")
    metrics = _as_list(plan.get("metrics"))
    risks = _as_list(plan.get("duplication_risks"))
    assumptions = _as_list(plan.get("assumptions"))
    notes = _as_list(plan.get("reasoning_notes"))
    if metrics:
        lines.append(f"- Metrics: {', '.join(metrics)}")
    if risks:
        lines.append(f"- Duplication risks: {', '.join(risks)}")
    if assumptions:
        lines.append(f"- Assumptions: {', '.join(assumptions)}")
    if notes:
        lines.append(f"- Notes: {', '.join(notes)}")
    if plan.get("omission_rationale"):
        lines.append(f"- Omission rationale: {plan.get('omission_rationale')}")
    return "\n".join(lines)


def get_sql_query(question, plan, table_inventory, context_block, relationship_context, client, model_name):
    prompt = f"""You are an expert DuckDB SQL analyst.
Output valid DuckDB SQL only.

QUESTION:
{question}

QUERY PLAN:
{json.dumps(plan, indent=2)}

AVAILABLE TABLES:
{table_inventory}

{context_block}
{relationship_context}

SQL RULES:
1. Use read_parquet('path/pattern') for data access.
2. Use the plan. If required_tables are listed, use them unless impossible.
3. Prefer CTEs and pre-aggregation when joins can multiply rows.
4. Use exact column names from the schema.
5. Use COUNT(DISTINCT ...) when the metric asks for distinct users or entities.
6. Default to LIMIT 50 unless the query is already aggregated.
7. Output only SQL, but leading SQL comments are allowed if helpful.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return _strip_markdown_sql(response.choices[0].message.content)


def fix_sql_query(question, failed_sql, error_msg, plan, table_inventory, context_block, relationship_context, client, model_name):
    prompt = f"""The following DuckDB SQL query failed.
Output corrected DuckDB SQL only.

QUESTION:
{question}

QUERY PLAN:
{json.dumps(plan, indent=2)}

FAILED SQL:
{failed_sql}

ERROR:
{error_msg}

AVAILABLE TABLES:
{table_inventory}

{context_block}
{relationship_context}
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return _strip_markdown_sql(response.choices[0].message.content)


def validate_sql(sql_query, artifacts_dir):
    clean_sql = strip_leading_sql_comments(sql_query).strip().rstrip(";").strip()
    if not clean_sql:
        raise ValueError("Security alert: empty SQL query.")
    sql_without_strings = re.sub(r"'[^']*'", "", clean_sql)
    if ";" in sql_without_strings:
        raise ValueError("Security alert: multiple SQL statements are not allowed.")
    upper_sql = clean_sql.upper().lstrip()
    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("DESCRIBE") or upper_sql.startswith("WITH")):
        raise ValueError("Security alert: only SELECT, DESCRIBE, or WITH queries are permitted.")
    for keyword in DANGEROUS_KEYWORDS:
        if re.search(r"\b" + keyword + r"\b", upper_sql):
            raise ValueError(f"Security alert: forbidden keyword detected: {keyword}")

    allowed_prefix = os.path.realpath(artifacts_dir).replace("\\", "/")

    file_call_pattern = re.compile(
        r"""(?ix)
        \b(read_parquet|read_csv|read_csv_auto)\s*\(\s*'([^']*)'
        """
    )
    file_literals = [match.group(2) for match in file_call_pattern.finditer(clean_sql.replace("''", ""))]

    for literal in file_literals:
        if literal.lower().startswith(("http://", "https://", "s3://")):
            raise ValueError("Security alert: remote URLs are not permitted in this prototype.")

        normalized = literal.replace("\\", "/")
        target = os.path.realpath(os.path.dirname(normalized) if "*" in normalized else normalized).replace("\\", "/")
        if not target.startswith(allowed_prefix):
            raise ValueError("Security alert: the query references files outside the processed dataset.")

    if "LIMIT" not in upper_sql and not upper_sql.startswith("DESCRIBE"):
        clean_sql = f"SELECT * FROM ({clean_sql}) AS _limited LIMIT {HARD_ROW_LIMIT}"
    return clean_sql


def referenced_tables_in_sql(sql_text, metadata, artifacts_dir):
    found = []
    for table_name, info in metadata.get("tables", {}).items():
        pattern = os.path.join(artifacts_dir, info["file_pattern"]).replace("\\", "/")
        if pattern in sql_text:
            found.append(table_name)
    return sorted(set(found))


@st.cache_data(show_spinner=False, ttl=3600)
def execute_validated_sql(clean_sql, _artifacts_dir):
    conn = duckdb.connect(database=":memory:")
    try:
        conn.execute("SET autoinstall_known_extensions=false")
        conn.execute("SET autoload_known_extensions=false")
        return conn.execute(clean_sql).df()
    finally:
        conn.close()


def critique_result(question, df, plan):
    warnings = []
    for col in df.columns:
        name = col.lower()
        if "percent" in name or "rate" in name:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any() and series.max() > 100:
                warnings.append(f"Column `{col}` exceeds 100 for some rows.")
    lower_cols = {col.lower(): col for col in df.columns}
    if "distinct_users_who_posted" in lower_cols and "distinct_enrolled_users" in lower_cols:
        posters = pd.to_numeric(df[lower_cols["distinct_users_who_posted"]], errors="coerce")
        enrolled = pd.to_numeric(df[lower_cols["distinct_enrolled_users"]], errors="coerce")
        if ((posters > enrolled) & posters.notna() & enrolled.notna()).any():
            warnings.append("Some rows show more posting users than enrolled users.")
    if len(df) == 0:
        warnings.append("Query returned no rows.")
    if len(df) >= HARD_ROW_LIMIT:
        warnings.append("Results hit the row cap.")
    if plan.get("required_tables"):
        warnings.append(f"Plan expected tables: {', '.join(plan['required_tables'])}")
    return warnings


def summarize_answer(question, plan, df, warnings, client, model_name, pii_redaction_enabled):
    sample_df = df.head(MAX_ROWS_FOR_SUMMARY) if len(df) > MAX_ROWS_FOR_SUMMARY else df.copy()
    redacted_cols = []
    if pii_redaction_enabled:
        sample_df, redacted_cols = redact_pii(sample_df)
    prompt = f"""You are summarizing a structured-data analysis for a non-technical user.

Question:
{question}

Query plan:
{json.dumps(plan, indent=2)}

Result sample:
{sample_df.to_string(index=False)}

Warnings:
{json.dumps(warnings)}

Rules:
1. Be concise but informative.
2. Mention assumptions or data-quality concerns if they matter.
3. If warnings exist, explain them clearly.
4. Do not speculate about redacted values.
"""
    if redacted_cols:
        prompt += f"\nRedacted columns: {', '.join(redacted_cols)}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def attempt_visualization(df):
    try:
        if df.empty or len(df) < 2:
            return
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if len(df.columns) <= 2 and len(df) <= 5 and numeric_cols:
            return
        st.caption("Auto-visualization")
        if categorical_cols and numeric_cols:
            chart_df = df.head(25).set_index(categorical_cols[0])[numeric_cols[0]]
            st.bar_chart(chart_df)
            return
        if len(numeric_cols) >= 2:
            st.scatter_chart(df, x=numeric_cols[0], y=numeric_cols[1])
    except Exception as exc:
        logging.info("Visualization skipped: %s", exc)


def _format_chat_export():
    sections = []
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        sections.append(f"### {role}\n{msg['content']}")
    return "\n\n---\n\n".join(sections)


def generate_starter_questions(dataset_profile, client, model_name):
    prompt = f"""Generate 5 short starter questions for a non-technical user exploring a dataset.
Use only this profile:

{build_profile_prompt(dataset_profile)}

Output one question per line and nothing else.
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        cleaned = []
        for line in response.choices[0].message.content.splitlines():
            line = re.sub(r"^\s*[-0-9.]+\s*", "", line).strip()
            if line:
                cleaned.append(line)
        return cleaned[:5]
    except Exception as exc:
        logging.info("Starter question generation failed: %s", exc)
        return []


def render_sidebar():
    with st.sidebar:
        st.title("Configuration")
        provider_names = list(PROVIDER_CONFIG.keys())
        provider_name = st.selectbox("LLM provider", provider_names, index=provider_names.index(DEFAULT_PROVIDER_NAME))
        provider_config = PROVIDER_CONFIG[provider_name]

        api_key = ""
        try:
            if provider_config["secret_key"] in st.secrets:
                api_key = st.secrets[provider_config["secret_key"]]
        except Exception:
            pass
        if api_key:
            st.success(f"{provider_name} key loaded from secrets")
        else:
            api_key = st.text_input(f"{provider_name} API key", type="password")

        model_name = st.text_input("Model name", value=provider_config["default_model"])
        st.markdown("---")
        st.session_state.pii_redaction = st.toggle(
            "PII redaction before summarization",
            value=st.session_state.pii_redaction,
        )
        st.markdown("---")
        if st.button("Clear processed dataset", use_container_width=True):
            reset_dataset_state()
            st.rerun()
        if len(st.session_state.messages) > 1:
            st.download_button(
                "Export chat history",
                data=_format_chat_export(),
                file_name="chat_history.md",
                mime="text/markdown",
                use_container_width=True,
            )
        return provider_name, api_key, model_name, provider_config


def render_processing_ui():
    st.subheader("1. Upload and profile")
    st.caption("Direction 2 uses a stronger dataset profile and explicit planning before SQL generation.")

    with st.expander("How Direction 2 works", expanded=True):
        st.markdown(
            """
            1. Upload sanitized or dummy CSV/ZIP files.
            2. The app preprocesses them into queryable Parquet artifacts.
            3. The app builds a dataset profile with table kinds, likely keys, overlaps, and relationship hints.
            4. For each analytical question, the app:
               - classifies the question
               - creates an explicit answer plan
               - generates SQL from the plan
               - critiques the result before summarizing it

            This prototype is meant to test whether stronger AI planning improves accuracy and reliability for large structured datasets.
            """
        )

    strategy = st.radio(
        "Preprocessing strategy",
        options=["merge", "separate"],
        index=1,
        format_func=lambda value: "Merge all files into one table" if value == "merge" else "Keep files as separate tables",
        horizontal=True,
    )

    uploaded_files = st.file_uploader(
        "Upload CSV or ZIP files",
        type=["csv", "zip"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        total_mb = sum(len(item.getbuffer()) for item in uploaded_files) / (1024 * 1024)
        st.info(f"Selected {len(uploaded_files)} file(s), about {total_mb:.1f} MB total.")

    if st.button("Process uploads", type="primary", use_container_width=True):
        process_uploaded_files(uploaded_files, strategy)
        st.rerun()


def render_dataset_summary():
    if not st.session_state.dataset_ready or not st.session_state.metadata:
        return
    metadata = st.session_state.metadata
    summary = st.session_state.processing_summary
    profile = st.session_state.dataset_profile or {}

    st.subheader("2. Dataset profile")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CSV files", summary.get("csv_count", 0))
    col2.metric("Tables", len(metadata.get("tables", {})))
    col3.metric("Columns", len(metadata.get("columns", [])))
    col4.metric("Upload size (MB)", summary.get("uploaded_mb", 0.0))

    with st.expander("Profile summary", expanded=False):
        st.json(profile)


def render_chat_ui(provider_name, api_key, model_name, provider_config):
    st.subheader("3. Ask questions")
    if not st.session_state.dataset_ready:
        st.info("Process a dataset first.")
        return
    if not api_key:
        st.warning("Add an API key in the sidebar to enable planning and analysis.")
        return

    client = OpenAI(api_key=api_key, base_url=provider_config["base_url"])
    metadata = st.session_state.metadata
    dataset_profile = st.session_state.dataset_profile
    artifacts_dir = st.session_state.artifacts_dir
    table_inventory = build_table_inventory(metadata, artifacts_dir)
    relationship_context = build_relationship_context(metadata)

    if not st.session_state.starter_questions:
        with st.spinner("Generating suggested questions..."):
            st.session_state.starter_questions = generate_starter_questions(dataset_profile, client, model_name)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if st.session_state.starter_questions and len(st.session_state.messages) <= 2:
        st.caption("Suggested questions")
        clicked = None
        for question in st.session_state.starter_questions:
            if st.button(question, key=f"starter_{question}", use_container_width=True):
                clicked = question
        if clicked:
            st.session_state.messages.append({"role": "user", "content": clicked})
            st.session_state.starter_questions = []
            st.rerun()

    user_input = st.chat_input("Ask about the processed dataset")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    st.session_state.starter_questions = []

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            try:
                route = classify_question(user_input, metadata)
                status.write(f"Question route: {route}")

                if route == "metadata":
                    direct = handle_metadata_question(user_input, metadata)
                    if direct is not None:
                        status.write(direct["title"])
                        df = direct["dataframe"]
                        if not df.empty:
                            st.dataframe(df, use_container_width=True)
                        status.update(label="✅ Answer ready", state="complete")
                        st.write(direct["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": direct["answer"]})
                        return

                explicit_tables = find_explicit_table_mentions(user_input, metadata)
                status.write("Building answer plan")
                plan = generate_query_plan(user_input, dataset_profile, explicit_tables, client, model_name)
                if not plan:
                    plan = make_fallback_plan(user_input, metadata)

                missing_explicit = [table for table in explicit_tables if table not in plan.get("required_tables", [])]
                if missing_explicit:
                    existing_optional = set(plan.get("optional_tables_for_cursory_review", []))
                    plan["optional_tables_for_cursory_review"] = sorted(existing_optional.union(set(missing_explicit)))
                    plan.setdefault("reasoning_notes", []).append(
                        f"Explicitly named tables not used in main SQL will receive a cursory supplemental review: {', '.join(missing_explicit)}"
                    )

                st.markdown("**Plan**")
                st.markdown(build_plan_markdown(plan))
                with st.expander("Planner JSON", expanded=False):
                    st.json(plan)

                supplemental_review = build_supplemental_table_review(
                    metadata,
                    [table for table in plan.get("optional_tables_for_cursory_review", []) if table not in plan.get("required_tables", [])],
                )
                if supplemental_review:
                    with st.expander("Supplemental Table Review", expanded=False):
                        st.json(supplemental_review)

                context_block = build_context_block(metadata, user_input)
                status.write("Generating SQL from plan")
                sql = get_sql_query(user_input, plan, table_inventory, context_block, relationship_context, client, model_name)

                referenced = referenced_tables_in_sql(sql, metadata, artifacts_dir)
                missing_required = [table for table in plan.get("required_tables", []) if table not in referenced]
                if missing_required:
                    status.write("Repairing SQL to include planned tables")
                    sql = fix_sql_query(
                        user_input,
                        sql,
                        f"Generated SQL omitted planned tables: {', '.join(missing_required)}",
                        plan,
                        table_inventory,
                        context_block,
                        relationship_context,
                        client,
                        model_name,
                    )

                st.code(sanitize_sql_for_display(sql, metadata, artifacts_dir), language="sql")
                status.write("Validating and executing SQL")
                clean_sql = validate_sql(sql, artifacts_dir)
                try:
                    df = execute_validated_sql(clean_sql, artifacts_dir)
                except Exception as first_error:
                    status.write("SQL failed. Asking the model to repair it.")
                    sql_retry = fix_sql_query(
                        user_input,
                        sql,
                        str(first_error),
                        plan,
                        table_inventory,
                        context_block,
                        relationship_context,
                        client,
                        model_name,
                    )
                    st.code(sanitize_sql_for_display(sql_retry, metadata, artifacts_dir), language="sql")
                    clean_sql = validate_sql(sql_retry, artifacts_dir)
                    df = execute_validated_sql(clean_sql, artifacts_dir)

                st.dataframe(df, use_container_width=True)
                if not df.empty:
                    st.download_button(
                        "Download results as CSV",
                        data=df.to_csv(index=False),
                        file_name="query_results.csv",
                        mime="text/csv",
                    )
                    attempt_visualization(df)

                warnings = critique_result(user_input, df, plan)
                if warnings:
                    st.markdown("**Critique**")
                    for warning in warnings:
                        st.warning(warning)

                status.write("Summarizing result")
                if df.empty:
                    answer = "The query returned no rows. Try broadening the question or checking the filter assumptions."
                else:
                    answer = summarize_answer(
                        user_input,
                        plan,
                        df,
                        warnings,
                        client,
                        model_name,
                        st.session_state.pii_redaction,
                    )

                status.update(label="✅ Answer ready", state="complete")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except ValueError as exc:
                status.update(label="🔒 Blocked", state="error")
                st.error(str(exc))
            except Exception as exc:
                logging.exception("Direction 2 failed")
                status.update(label="❌ Failed", state="error")
                st.error(str(exc))


def main():
    ensure_session_state()
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    with st.expander("Important notes", expanded=True):
        st.markdown(
            """
            - This prototype is for sanitized or dummy data only.
            - Direction 2 is intentionally different from Direction 1: it tries to use AI not just for SQL generation, but also for dataset interpretation and answer planning.
            - It is still a prototype, so the goal is to learn whether explicit planning improves reliability on large structured datasets.
            """
        )

    provider_name, api_key, model_name, provider_config = render_sidebar()
    render_processing_ui()
    render_dataset_summary()
    render_chat_ui(provider_name, api_key, model_name, provider_config)


if __name__ == "__main__":
    main()
