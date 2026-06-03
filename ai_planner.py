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
from pathlib import Path

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
JOIN_COVERAGE_SAMPLE_LIMIT = 50000
MAX_JOIN_COVERAGE_HINTS = 12

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

ANONYMIZER_APP_SOURCE = '# streamlit run csv_anonymizer_chunker.py\n# streamlit run csv_anonymizer_chunker.py --server.maxUploadSize 10000\n# cd \n#   psutil>=5.9,<6\n\n\nimport csv\nimport hashlib\nimport io\nimport json\nimport os\nimport random\nimport re\nimport shutil\nimport tempfile\nimport time\nimport zipfile\nfrom contextlib import contextmanager\nfrom dataclasses import dataclass\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom urllib.parse import urlsplit, urlunsplit\n\nimport pandas as pd\nimport streamlit as st\n\ntry:\n    import psutil\nexcept ImportError:  # The app still works, but memory checks are disabled.\n    psutil = None\n\n\nAPP_TITLE = "Local CSV Anonymizer and Chunker"\nDEFAULT_TARGET_DOMAIN = "acme.brightspace.com"\nDEFAULT_CHUNK_SIZE_MB = 500\nMAX_CHUNK_SIZE_MB = 2048\nDEFAULT_READ_ROWS = 100_000\nPREVIEW_ROWS = 100\nDEFAULT_RESOURCE_CHECK_EVERY = 10\nDEFAULT_STATUS_UPDATE_EVERY = 10\nDEFAULT_MIN_FREE_DISK_GB = 20.0\nDEFAULT_MIN_FREE_MEMORY_GB = 4.0\n\nACTION_NONE = "No change"\nACTION_NAME = "Randomized name misspelling"\nACTION_NUMERIC_ID = "Random numeric ID"\nACTION_ALPHANUMERIC_ID = "Random alphanumeric ID"\nACTION_EMAIL = "Fake email"\nACTION_URL_DOMAIN = "Replace Brightspace URL/domain"\nACTION_STATIC = "Static replacement"\nACTION_REDACT = "Redact"\nACTION_HASH = "Stable hash token"\n\nACTION_OPTIONS = [\n    ACTION_NONE,\n    ACTION_NAME,\n    ACTION_NUMERIC_ID,\n    ACTION_ALPHANUMERIC_ID,\n    ACTION_EMAIL,\n    ACTION_URL_DOMAIN,\n    ACTION_STATIC,\n    ACTION_REDACT,\n    ACTION_HASH,\n]\n\nLIKELY_NAME_PATTERNS = [r"\\bname\\b", r"first.?name", r"last.?name", r"full.?name"]\nLIKELY_ID_PATTERNS = [r"\\bid\\b", r"user.?id", r"student.?id", r"employee.?id", r"person.?id"]\nLIKELY_EMAIL_PATTERNS = [r"email", r"e.?mail"]\nLIKELY_URL_PATTERNS = [r"url", r"link", r"domain", r"host", r"href"]\nLIKELY_REDACT_PATTERNS = [r"password", r"secret", r"token", r"ssn", r"social.?sec"]\n\n\nst.set_page_config(page_title=APP_TITLE, layout="wide")\n\n\n@dataclass\nclass SourceDescriptor:\n    kind: str\n    display_name: str\n    source_name: str\n    size_bytes: int\n    path: str = ""\n    zip_path: str = ""\n    zip_member: str = ""\n    compressed_size_bytes: int = 0\n\n    def to_dict(self):\n        return {\n            "kind": self.kind,\n            "display_name": self.display_name,\n            "source_name": self.source_name,\n            "size_bytes": self.size_bytes,\n            "path": self.path,\n            "zip_path": self.zip_path,\n            "zip_member": self.zip_member,\n            "compressed_size_bytes": self.compressed_size_bytes,\n        }\n\n    @classmethod\n    def from_dict(cls, data):\n        return cls(**data)\n\n\nclass ResourceLimitError(RuntimeError):\n    def __init__(self, message, partial_result=None):\n        super().__init__(message)\n        self.partial_result = partial_result\n\n\ndef ensure_state():\n    defaults = {\n        "source_signature": "",\n        "source_files": [],\n        "profile": None,\n        "rules_df": None,\n        "last_manifest": None,\n        "work_dir": None,\n        "cleanup_candidates": [],\n    }\n    for key, value in defaults.items():\n        if key not in st.session_state:\n            st.session_state[key] = value\n\n\ndef normalize_column_name(value):\n    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())\n\n\ndef safe_table_stem(file_name):\n    stem = Path(file_name).stem\n    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem).strip("._-")\n    return clean or "data"\n\n\ndef bytes_to_mb(value):\n    return round(float(value or 0) / (1024 * 1024), 2)\n\n\ndef bytes_to_gb(value):\n    return round(float(value or 0) / (1024 * 1024 * 1024), 2)\n\n\ndef clean_local_path_piece(value):\n    cleaned = (value or "").strip()\n    cleaned = cleaned.strip("\\ufeff").strip()\n    # Windows Explorer often copies paths wrapped in straight or smart quotes.\n    quote_pairs = [(\'"\', \'"\'), ("\'", "\'"), ("“", "”"), ("‘", "’")]\n    changed = True\n    while changed and len(cleaned) >= 2:\n        changed = False\n        for left, right in quote_pairs:\n            if cleaned.startswith(left) and cleaned.endswith(right):\n                cleaned = cleaned[1:-1].strip()\n                changed = True\n    return cleaned\n\n\ndef parse_local_paths(raw_text):\n    paths = []\n    for raw_line in (raw_text or "").splitlines():\n        line = raw_line.strip()\n        if not line:\n            continue\n        # Prefer one path per line. Comma splitting is supported only when csv parsing is unambiguous.\n        try:\n            parsed = next(csv.reader([line], skipinitialspace=True))\n        except Exception:\n            parsed = [line]\n        pieces = parsed if len(parsed) > 1 else [line]\n        for piece in pieces:\n            cleaned = clean_local_path_piece(piece)\n            if cleaned:\n                paths.append(cleaned)\n    return paths\n\n\nTEMP_CLEANUP_PREFIXES = ["csv_anonymizer_", "streamlit_direction2"]\n\n\ndef get_directory_size_bytes(path):\n    total = 0\n    root = Path(path)\n    if not root.exists():\n        return 0\n    for child in root.rglob("*"):\n        try:\n            if child.is_file() or child.is_symlink():\n                total += child.stat().st_size\n        except OSError:\n            continue\n    return total\n\n\ndef discover_cleanup_candidates(include_ai_planner=True, min_age_hours=0):\n    temp_root = Path(tempfile.gettempdir())\n    now = time.time()\n    prefixes = ["csv_anonymizer_"]\n    if include_ai_planner:\n        prefixes.append("streamlit_direction2")\n\n    current_work_dir = st.session_state.get("work_dir")\n    current_work_dir = str(Path(current_work_dir).resolve()) if current_work_dir else ""\n\n    candidates = []\n    if not temp_root.exists():\n        return candidates\n\n    for child in temp_root.iterdir():\n        try:\n            if not child.is_dir():\n                continue\n            if not any(child.name.startswith(prefix) for prefix in prefixes):\n                continue\n            child_resolved = str(child.resolve())\n            if current_work_dir and child_resolved == current_work_dir:\n                continue\n            stat = child.stat()\n            age_hours = max(0.0, (now - stat.st_mtime) / 3600.0)\n            if age_hours < float(min_age_hours):\n                continue\n            candidates.append(\n                {\n                    "path": str(child),\n                    "name": child.name,\n                    "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),\n                    "age_hours": round(age_hours, 1),\n                    "size_gb": bytes_to_gb(get_directory_size_bytes(child)),\n                }\n            )\n        except OSError:\n            continue\n\n    return sorted(candidates, key=lambda item: item["size_gb"], reverse=True)\n\n\ndef cleanup_candidate_dirs(candidates):\n    deleted = []\n    failed = []\n    for item in candidates:\n        path = Path(item["path"])\n        try:\n            if path.exists() and path.is_dir():\n                shutil.rmtree(path)\n            deleted.append(str(path))\n        except Exception as exc:\n            failed.append({"path": str(path), "error": str(exc)})\n    return deleted, failed\n\n\ndef ensure_safe_zip_member(member_name):\n    candidate = Path(member_name)\n    if candidate.is_absolute() or ".." in candidate.parts:\n        raise ValueError(f"Unsafe ZIP member path: {member_name}")\n\n\ndef csv_sources_from_zip(zip_path):\n    sources = []\n    with zipfile.ZipFile(zip_path, "r") as zf:\n        for info in zf.infolist():\n            ensure_safe_zip_member(info.filename)\n            if info.is_dir() or not info.filename.lower().endswith(".csv"):\n                continue\n            display_name = f"{Path(zip_path).name}::{info.filename}"\n            sources.append(\n                SourceDescriptor(\n                    kind="zip_member",\n                    display_name=display_name,\n                    source_name=Path(info.filename).name,\n                    size_bytes=int(info.file_size or 0),\n                    zip_path=str(zip_path),\n                    zip_member=info.filename,\n                    compressed_size_bytes=int(info.compress_size or 0),\n                )\n            )\n    return sources\n\n\ndef collect_local_sources(raw_text):\n    sources = []\n    seen = set()\n\n    for raw_path in parse_local_paths(raw_text):\n        path = Path(raw_path).expanduser().resolve()\n        if not path.exists():\n            raise ValueError(f"Local source not found: {raw_path}")\n\n        candidates = []\n        if path.is_dir():\n            candidates = [child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in {".csv", ".zip"}]\n        else:\n            candidates = [path]\n\n        for candidate in candidates:\n            key = str(candidate)\n            if candidate.suffix.lower() == ".csv":\n                if key in seen:\n                    continue\n                seen.add(key)\n                sources.append(\n                    SourceDescriptor(\n                        kind="csv",\n                        display_name=candidate.name,\n                        source_name=candidate.name,\n                        size_bytes=candidate.stat().st_size,\n                        path=str(candidate),\n                    )\n                )\n            elif candidate.suffix.lower() == ".zip":\n                for source in csv_sources_from_zip(candidate):\n                    zip_key = f"{source.zip_path}::{source.zip_member}"\n                    if zip_key not in seen:\n                        seen.add(zip_key)\n                        sources.append(source)\n            else:\n                raise ValueError(f"Unsupported source type: {raw_path}")\n\n    return sorted(sources, key=lambda item: item.display_name.lower())\n\n\ndef save_uploaded_sources(uploaded_files, work_dir):\n    upload_dir = Path(work_dir) / "uploads"\n    upload_dir.mkdir(parents=True, exist_ok=True)\n    sources = []\n    seen = set()\n\n    for uploaded in uploaded_files or []:\n        target_path = upload_dir / Path(uploaded.name).name\n        with open(target_path, "wb") as handle:\n            handle.write(uploaded.getbuffer())\n\n        if uploaded.name.lower().endswith(".csv"):\n            sources.append(\n                SourceDescriptor(\n                    kind="csv",\n                    display_name=target_path.name,\n                    source_name=target_path.name,\n                    size_bytes=target_path.stat().st_size,\n                    path=str(target_path),\n                )\n            )\n        elif uploaded.name.lower().endswith(".zip"):\n            for source in csv_sources_from_zip(target_path):\n                key = f"{source.zip_path}::{source.zip_member}"\n                if key not in seen:\n                    seen.add(key)\n                    sources.append(source)\n\n    return sorted(sources, key=lambda item: item.display_name.lower())\n\n\n@contextmanager\ndef open_source_binary(source):\n    if source.kind == "csv":\n        with open(source.path, "rb") as handle:\n            yield handle\n    elif source.kind == "zip_member":\n        with zipfile.ZipFile(source.zip_path, "r") as zf:\n            ensure_safe_zip_member(source.zip_member)\n            with zf.open(source.zip_member, "r") as handle:\n                yield handle\n    else:\n        raise ValueError(f"Unsupported source kind: {source.kind}")\n\n\ndef read_csv_preview(source, nrows):\n    try:\n        with open_source_binary(source) as handle:\n            return pd.read_csv(handle, nrows=nrows, encoding_errors="replace", on_bad_lines="skip")\n    except UnicodeDecodeError:\n        with open_source_binary(source) as handle:\n            return pd.read_csv(handle, nrows=nrows, encoding="latin-1", on_bad_lines="skip")\n\n\ndef read_csv_chunks(source, rows_per_chunk):\n    try:\n        with open_source_binary(source) as handle:\n            yield from pd.read_csv(handle, chunksize=rows_per_chunk, encoding_errors="replace", on_bad_lines="skip")\n    except UnicodeDecodeError:\n        with open_source_binary(source) as handle:\n            yield from pd.read_csv(handle, chunksize=rows_per_chunk, encoding="latin-1", on_bad_lines="skip")\n\n\ndef infer_action(column_name):\n    normalized = normalize_column_name(column_name)\n    pattern_groups = [\n        (LIKELY_REDACT_PATTERNS, ACTION_REDACT),\n        (LIKELY_EMAIL_PATTERNS, ACTION_EMAIL),\n        (LIKELY_URL_PATTERNS, ACTION_URL_DOMAIN),\n        (LIKELY_NAME_PATTERNS, ACTION_NAME),\n        (LIKELY_ID_PATTERNS, ACTION_NUMERIC_ID),\n    ]\n    for patterns, action in pattern_groups:\n        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):\n            return action\n    return ACTION_NONE\n\n\ndef profile_source(source):\n    preview = read_csv_preview(source, PREVIEW_ROWS)\n    columns = [str(col) for col in preview.columns]\n    return {\n        "source": source.to_dict(),\n        "file_name": source.display_name,\n        "source_name": source.source_name,\n        "size_mb": bytes_to_mb(source.size_bytes),\n        "compressed_size_mb": bytes_to_mb(source.compressed_size_bytes),\n        "columns": columns,\n        "preview": preview,\n    }\n\n\ndef build_profile(sources):\n    files = [profile_source(source) for source in sources]\n    all_columns = []\n    for item in files:\n        for column in item["columns"]:\n            if column not in all_columns:\n                all_columns.append(column)\n    return {"files": files, "all_columns": all_columns}\n\n\ndef initial_rules_df(columns):\n    return pd.DataFrame(\n        [\n            {\n                "column": column,\n                "action": infer_action(column),\n                "id_length": 10,\n                "static_value": "",\n                "target_domain": DEFAULT_TARGET_DOMAIN,\n            }\n            for column in columns\n        ]\n    )\n\n\ndef digest_text(value, salt, length=32):\n    text = "" if pd.isna(value) else str(value)\n    return hashlib.sha256(f"{salt}|{text}".encode("utf-8", errors="replace")).hexdigest()[:length]\n\n\ndef seeded_random(value, salt):\n    seed_hex = digest_text(value, salt, length=16)\n    return random.Random(int(seed_hex, 16))\n\n\ndef randomized_name(value, salt):\n    if pd.isna(value) or str(value).strip() == "":\n        return value\n\n    rng = seeded_random(value, salt)\n    parts = re.split(r"(\\s+|-|\'|\\.)", str(value))\n    out_parts = []\n    vowels = "aeiou"\n\n    for part in parts:\n        if not part.isalpha() or len(part) < 3:\n            out_parts.append(part)\n            continue\n\n        chars = list(part.lower())\n        choice = rng.choice(["swap", "vowel", "drop", "double"])\n        if choice == "swap" and len(chars) > 3:\n            idx = rng.randrange(1, len(chars) - 1)\n            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]\n        elif choice == "vowel":\n            vowel_positions = [idx for idx, char in enumerate(chars) if char in vowels]\n            if vowel_positions:\n                idx = rng.choice(vowel_positions)\n                chars[idx] = rng.choice([v for v in vowels if v != chars[idx]])\n        elif choice == "drop" and len(chars) > 4:\n            chars.pop(rng.randrange(1, len(chars) - 1))\n        elif choice == "double" and len(chars) > 3:\n            idx = rng.randrange(1, len(chars) - 1)\n            chars.insert(idx, chars[idx])\n\n        new_part = "".join(chars)\n        if part[0].isupper():\n            new_part = new_part.capitalize()\n        out_parts.append(new_part)\n\n    return "".join(out_parts)\n\n\ndef numeric_id(value, salt, length):\n    if pd.isna(value) or str(value).strip() == "":\n        return value\n    digest = digest_text(value, salt, length=64)\n    digits = "".join(str(int(char, 16) % 10) for char in digest)\n    digits = digits[: max(1, int(length))]\n    if digits and digits[0] == "0":\n        digits = "7" + digits[1:]\n    return digits\n\n\ndef alphanumeric_id(value, salt, length):\n    if pd.isna(value) or str(value).strip() == "":\n        return value\n    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"\n    rng = seeded_random(value, salt)\n    return "".join(rng.choice(alphabet) for _ in range(max(1, int(length))))\n\n\ndef fake_email(value, salt):\n    if pd.isna(value) or str(value).strip() == "":\n        return value\n    return f"user_{alphanumeric_id(value, salt, 12).lower()}@example.invalid"\n\n\ndef replace_brightspace_domain(value, target_domain):\n    if pd.isna(value) or str(value).strip() == "":\n        return value\n\n    text = str(value)\n    domain = (target_domain or DEFAULT_TARGET_DOMAIN).strip()\n\n    if "://" in text:\n        try:\n            parsed = urlsplit(text)\n            if parsed.netloc.lower().endswith("brightspace.com"):\n                return urlunsplit((parsed.scheme, domain, parsed.path, parsed.query, parsed.fragment))\n        except Exception:\n            pass\n\n    return re.sub(r"\\b[a-zA-Z0-9.-]+\\.brightspace\\.com\\b", domain, text)\n\n\ndef stable_hash_token(value, salt):\n    if pd.isna(value) or str(value).strip() == "":\n        return value\n    return "anon_" + digest_text(value, salt, length=16)\n\n\ndef apply_rules_to_chunk(df, rules, salt):\n    output = df.copy()\n    for rule in rules:\n        column = rule["column"]\n        if column not in output.columns:\n            continue\n\n        action = rule["action"]\n        id_length = int(rule.get("id_length") or 10)\n        static_value = str(rule.get("static_value") or "")\n        target_domain = str(rule.get("target_domain") or DEFAULT_TARGET_DOMAIN)\n        column_salt = f"{salt}|{column}"\n\n        if action == ACTION_NONE:\n            continue\n        if action == ACTION_NAME:\n            output[column] = output[column].map(lambda value: randomized_name(value, column_salt))\n        elif action == ACTION_NUMERIC_ID:\n            output[column] = output[column].map(lambda value: numeric_id(value, column_salt, id_length))\n        elif action == ACTION_ALPHANUMERIC_ID:\n            output[column] = output[column].map(lambda value: alphanumeric_id(value, column_salt, id_length))\n        elif action == ACTION_EMAIL:\n            output[column] = output[column].map(lambda value: fake_email(value, column_salt))\n        elif action == ACTION_URL_DOMAIN:\n            output[column] = output[column].map(lambda value: replace_brightspace_domain(value, target_domain))\n        elif action == ACTION_STATIC:\n            output[column] = static_value\n        elif action == ACTION_REDACT:\n            output[column] = "[REDACTED]"\n        elif action == ACTION_HASH:\n            output[column] = output[column].map(lambda value: stable_hash_token(value, column_salt))\n\n    return output\n\n\ndef open_new_output_file(output_dir, source_stem, part_number):\n    output_path = Path(output_dir) / f"{source_stem}_anonymized_part_{part_number:03d}.csv"\n    handle = open(output_path, "wb")\n    return output_path, handle\n\n\ndef csv_bytes_for_frame(df, include_header):\n    buffer = io.StringIO()\n    df.to_csv(buffer, index=False, header=include_header, quoting=csv.QUOTE_MINIMAL)\n    return buffer.getvalue().encode("utf-8")\n\n\ndef largest_csv_slice_that_fits(df, start_row, capacity_bytes, include_header):\n    remaining = len(df) - start_row\n    if remaining <= 0:\n        return 0, b""\n\n    one_row_bytes = csv_bytes_for_frame(df.iloc[start_row : start_row + 1], include_header)\n    if len(one_row_bytes) > capacity_bytes:\n        return 1, one_row_bytes\n\n    low = 1\n    high = remaining\n    best_count = 1\n    best_bytes = one_row_bytes\n\n    while low <= high:\n        mid = (low + high) // 2\n        candidate_bytes = csv_bytes_for_frame(df.iloc[start_row : start_row + mid], include_header)\n        if len(candidate_bytes) <= capacity_bytes:\n            best_count = mid\n            best_bytes = candidate_bytes\n            low = mid + 1\n        else:\n            high = mid - 1\n\n    return best_count, best_bytes\n\n\ndef get_resource_snapshot(output_dir):\n    output_path = Path(output_dir)\n    output_path.mkdir(parents=True, exist_ok=True)\n    disk = os.statvfs(output_path) if hasattr(os, "statvfs") else None\n    if disk:\n        free_disk_bytes = disk.f_bavail * disk.f_frsize\n    else:\n        import shutil\n\n        free_disk_bytes = shutil.disk_usage(output_path).free\n\n    memory_available_bytes = None\n    memory_percent_used = None\n    if psutil is not None:\n        mem = psutil.virtual_memory()\n        memory_available_bytes = int(mem.available)\n        memory_percent_used = round(float(mem.percent), 1)\n\n    return {\n        "free_disk_bytes": int(free_disk_bytes),\n        "free_disk_gb": bytes_to_gb(free_disk_bytes),\n        "memory_available_bytes": memory_available_bytes,\n        "memory_available_gb": None if memory_available_bytes is None else bytes_to_gb(memory_available_bytes),\n        "memory_percent_used": memory_percent_used,\n    }\n\n\ndef assert_resources_ok(output_dir, min_free_disk_gb, min_free_memory_gb):\n    snapshot = get_resource_snapshot(output_dir)\n    problems = []\n    if snapshot["free_disk_gb"] < float(min_free_disk_gb):\n        problems.append(\n            f"free disk space is {snapshot[\'free_disk_gb\']} GB, below the configured minimum of {min_free_disk_gb} GB"\n        )\n    if psutil is not None and snapshot["memory_available_gb"] is not None:\n        if snapshot["memory_available_gb"] < float(min_free_memory_gb):\n            problems.append(\n                f"available memory is {snapshot[\'memory_available_gb\']} GB, below the configured minimum of {min_free_memory_gb} GB"\n            )\n    if problems:\n        raise ResourceLimitError("Processing stopped safely because " + "; ".join(problems) + ".")\n    return snapshot\n\n\ndef format_resource_line(snapshot):\n    memory = "memory unavailable"\n    if snapshot.get("memory_available_gb") is not None:\n        memory = f"available memory: {snapshot[\'memory_available_gb\']} GB ({snapshot[\'memory_percent_used\']}% used)"\n    return f"free disk: {snapshot[\'free_disk_gb\']} GB; {memory}"\n\n\ndef selected_rules_from_df(rules_df):\n    cleaned = rules_df.copy()\n    cleaned["action"] = cleaned["action"].where(cleaned["action"].isin(ACTION_OPTIONS), ACTION_NONE)\n    cleaned["id_length"] = pd.to_numeric(cleaned["id_length"], errors="coerce").fillna(10).clip(1, 64).astype(int)\n    return cleaned.to_dict("records")\n\n\ndef source_signature(uploaded_files, local_text):\n    upload_bits = [f"{item.name}:{item.size}" for item in uploaded_files or []]\n    return hashlib.sha256(("|".join(upload_bits) + "||" + (local_text or "")).encode("utf-8")).hexdigest()\n\n\ndef write_chunked_csv(\n    source,\n    output_dir,\n    rules,\n    salt,\n    max_chunk_mb,\n    rows_per_read,\n    progress_text,\n    resource_check_every,\n    status_update_every,\n    min_free_disk_gb,\n    min_free_memory_gb,\n):\n    source_stem = safe_table_stem(source.source_name)\n    max_bytes = int(max_chunk_mb * 1024 * 1024)\n    part_number = 1\n    rows_written = 0\n    batches_read = 0\n    output_files = []\n    output_path, handle = open_new_output_file(output_dir, source_stem, part_number)\n\n    partial_result = {\n        "source": source.to_dict(),\n        "rows_written": 0,\n        "batches_read": 0,\n        "output_files": output_files,\n        "stopped_early": False,\n    }\n\n    try:\n        snapshot = assert_resources_ok(output_dir, min_free_disk_gb, min_free_memory_gb)\n        progress_text.write(f"Starting `{source.display_name}`. {format_resource_line(snapshot)}")\n\n        wrote_header = False\n        last_status_time = time.time()\n        for chunk_idx, chunk in enumerate(read_csv_chunks(source, rows_per_read), start=1):\n            batches_read = chunk_idx\n\n            if chunk_idx == 1 or chunk_idx % max(1, int(resource_check_every)) == 0:\n                try:\n                    snapshot = assert_resources_ok(output_dir, min_free_disk_gb, min_free_memory_gb)\n                except ResourceLimitError as exc:\n                    partial_result.update(\n                        {\n                            "rows_written": rows_written,\n                            "batches_read": batches_read,\n                            "stopped_early": True,\n                        }\n                    )\n                    if handle and not handle.closed:\n                        handle.flush()\n                    exc.partial_result = partial_result\n                    raise\n\n            if chunk_idx == 1 or chunk_idx % max(1, int(status_update_every)) == 0:\n                progress_text.write(\n                    f"Processing `{source.display_name}`: batch {chunk_idx:,}; "\n                    f"rows written so far: {rows_written:,}; {format_resource_line(snapshot)}"\n                )\n                last_status_time = time.time()\n            elif time.time() - last_status_time > 30:\n                progress_text.write(\n                    f"Still processing `{source.display_name}`: batch {chunk_idx:,}; rows written so far: {rows_written:,}."\n                )\n                last_status_time = time.time()\n\n            transformed = apply_rules_to_chunk(chunk, rules, salt)\n            row_offset = 0\n\n            while row_offset < len(transformed):\n                include_header = not wrote_header\n                remaining_bytes = max_bytes - handle.tell()\n                if remaining_bytes <= 0 or (not include_header and remaining_bytes < 1024):\n                    handle.close()\n                    output_files.append(str(output_path))\n                    part_number += 1\n                    snapshot = assert_resources_ok(output_dir, min_free_disk_gb, min_free_memory_gb)\n                    output_path, handle = open_new_output_file(output_dir, source_stem, part_number)\n                    wrote_header = False\n                    include_header = True\n                    remaining_bytes = max_bytes\n                    progress_text.write(\n                        f"Started output chunk `{output_path.name}` for `{source.display_name}`. {format_resource_line(snapshot)}"\n                    )\n\n                rows_to_write, encoded = largest_csv_slice_that_fits(\n                    transformed,\n                    start_row=row_offset,\n                    capacity_bytes=remaining_bytes,\n                    include_header=include_header,\n                )\n                handle.write(encoded)\n                wrote_header = True\n                row_offset += rows_to_write\n\n            rows_written += len(transformed)\n            partial_result.update({"rows_written": rows_written, "batches_read": batches_read})\n\n        if handle and not handle.closed:\n            handle.close()\n        if output_path.exists() and output_path.stat().st_size > 0:\n            output_files.append(str(output_path))\n        partial_result.update(\n            {\n                "rows_written": rows_written,\n                "batches_read": batches_read,\n                "output_files": output_files,\n                "stopped_early": False,\n            }\n        )\n        return partial_result\n\n    except ResourceLimitError:\n        try:\n            if handle and not handle.closed:\n                handle.close()\n            if output_path.exists() and output_path.stat().st_size > 0 and str(output_path) not in output_files:\n                output_files.append(str(output_path))\n        finally:\n            partial_result.update(\n                {\n                    "rows_written": rows_written,\n                    "batches_read": batches_read,\n                    "output_files": output_files,\n                    "stopped_early": True,\n                }\n            )\n        raise\n    except Exception:\n        try:\n            if handle and not handle.closed:\n                handle.close()\n        except Exception:\n            pass\n        raise\n\n\ndef render_cleanup_section():\n    st.subheader("0. Clean up old temp files")\n    with st.expander("Find and remove old work folders", expanded=False):\n        st.caption(\n            "Use this after stopping prior test runs. It looks in your system temp folder for previous "\n            "csv_anonymizer_* folders, and optionally the earlier AI planner temp folder. "\n            "The current active work folder is skipped."\n        )\n        col1, col2 = st.columns([1, 1])\n        with col1:\n            include_ai_planner = st.checkbox(\n                "Also include streamlit_direction2 folders",\n                value=True,\n                help="This was used by the earlier AI planner workflow and can hold uploaded/extracted/parquet artifacts.",\n            )\n        with col2:\n            min_age_hours = st.number_input(\n                "Only show folders older than N hours",\n                min_value=0.0,\n                max_value=168.0,\n                value=0.0,\n                step=1.0,\n                help="Use 0 to show all matching old folders. Increase this if another run may still be active.",\n            )\n\n        if st.button("Scan temp folders", use_container_width=True):\n            candidates = discover_cleanup_candidates(include_ai_planner=include_ai_planner, min_age_hours=min_age_hours)\n            st.session_state.cleanup_candidates = candidates\n\n        candidates = st.session_state.get("cleanup_candidates", [])\n        if candidates:\n            total_gb = round(sum(float(item.get("size_gb") or 0) for item in candidates), 2)\n            st.info(f"Found {len(candidates)} matching folder(s), totaling about {total_gb} GB.")\n            st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)\n            st.warning("Delete only after Streamlit/Python runs that may be using these folders are stopped.")\n            confirm = st.checkbox("I understand these temp folders will be deleted permanently.")\n            if st.button("Delete listed temp folders", type="primary", disabled=not confirm, use_container_width=True):\n                deleted, failed = cleanup_candidate_dirs(candidates)\n                st.session_state.cleanup_candidates = []\n                if deleted:\n                    st.success(f"Deleted {len(deleted)} temp folder(s).")\n                if failed:\n                    st.error("Some folders could not be deleted:")\n                    st.dataframe(pd.DataFrame(failed), use_container_width=True, hide_index=True)\n        else:\n            st.caption(f"System temp folder: {tempfile.gettempdir()}")\n\n\ndef render_source_section():\n    st.subheader("1. Choose input files")\n    uploaded_files = st.file_uploader(\n        "Upload CSV or ZIP files",\n        type=["csv", "zip"],\n        accept_multiple_files=True,\n        help="For very large files, local paths are more reliable than browser upload.",\n    )\n    st.caption("For multi-GB files, prefer the local path box below so Python reads directly from disk.")\n    local_source_text = st.text_area(\n        "Or enter local CSV/ZIP file paths or folders",\n        placeholder=\'C:\\\\Users\\\\oakhtar\\\\Downloads\\\\System Access Log(1).zip\\n/path/to/large-file.csv\\n/path/to/export-folder\',\n        help="Paste one path per line. Quoted Windows paths are okay; surrounding quotes will be removed.",\n    )\n    st.caption(\'Tip: paste one path per line. Paths copied as "C:\\\\path\\\\file.zip" are handled automatically.\')\n\n    if st.button("Inspect columns", type="primary", use_container_width=True):\n        current_signature = source_signature(uploaded_files, local_source_text)\n        if not st.session_state.work_dir:\n            st.session_state.work_dir = tempfile.mkdtemp(prefix="csv_anonymizer_")\n        work_dir = st.session_state.work_dir\n        sources = []\n        if uploaded_files:\n            sources.extend(save_uploaded_sources(uploaded_files, work_dir))\n        if local_source_text.strip():\n            sources.extend(collect_local_sources(local_source_text))\n\n        unique = {}\n        for source in sources:\n            key = f"{source.kind}|{source.path}|{source.zip_path}|{source.zip_member}"\n            unique[key] = source\n        sources = sorted(unique.values(), key=lambda item: item.display_name.lower())\n\n        if not sources:\n            st.error("No CSV files were found. Upload CSV/ZIP files or enter local paths.")\n            return\n\n        with st.spinner("Reading headers and previews without fully extracting ZIP files..."):\n            profile = build_profile(sources)\n\n        st.session_state.source_signature = current_signature\n        st.session_state.source_files = [source.to_dict() for source in sources]\n        st.session_state.profile = profile\n        st.session_state.rules_df = initial_rules_df(profile["all_columns"])\n        st.session_state.last_manifest = None\n        st.success(f"Found {len(sources)} CSV source(s) with {len(profile[\'all_columns\'])} unique column(s).")\n\n\ndef render_profile_section():\n    profile = st.session_state.profile\n    if not profile:\n        return\n\n    st.subheader("2. Review detected columns")\n    file_rows = [\n        {\n            "file": item["file_name"],\n            "uncompressed_size_mb": item["size_mb"],\n            "compressed_size_mb": item.get("compressed_size_mb") or "",\n            "columns": len(item["columns"]),\n        }\n        for item in profile["files"]\n    ]\n    st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)\n\n    with st.expander("Preview files", expanded=False):\n        for item in profile["files"]:\n            st.markdown(f"**{item[\'file_name\']}**")\n            st.dataframe(item["preview"], use_container_width=True, height=260)\n\n\ndef render_rules_section():\n    if st.session_state.rules_df is None:\n        return\n\n    st.subheader("3. Choose anonymization rules")\n    st.caption("Suggested actions are based on column names. Review them before processing.")\n    st.session_state.rules_df = st.data_editor(\n        st.session_state.rules_df,\n        use_container_width=True,\n        hide_index=True,\n        column_config={\n            "column": st.column_config.TextColumn("Column", disabled=True),\n            "action": st.column_config.SelectboxColumn("Action", options=ACTION_OPTIONS, required=True),\n            "id_length": st.column_config.NumberColumn("ID length", min_value=1, max_value=64, step=1),\n            "static_value": st.column_config.TextColumn("Static value"),\n            "target_domain": st.column_config.TextColumn("Target domain"),\n        },\n    )\n\n\ndef render_process_section():\n    if st.session_state.rules_df is None or not st.session_state.source_files:\n        return\n\n    st.subheader("4. Process and chunk")\n    col1, col2, col3 = st.columns([1, 1, 2])\n    with col1:\n        max_chunk_mb = st.number_input(\n            "Maximum output chunk size MB",\n            min_value=1,\n            max_value=MAX_CHUNK_SIZE_MB,\n            value=DEFAULT_CHUNK_SIZE_MB,\n            step=50,\n            help="This is a hard cap per output CSV part. Use less than 2048 MB for safer downstream handling.",\n        )\n    with col2:\n        rows_per_read = st.number_input(\n            "Rows processed at a time",\n            min_value=1_000,\n            max_value=1_000_000,\n            value=DEFAULT_READ_ROWS,\n            step=10_000,\n            help="Larger values may be faster but use more memory. Start with 100,000 for huge files.",\n        )\n    with col3:\n        output_dir = st.text_input(\n            "Output folder",\n            value=str(Path.cwd() / "anonymized_output"),\n            help="The app writes anonymized chunks and a manifest here. Choose a drive with plenty of free space.",\n        )\n\n    salt = st.text_input(\n        "Anonymization salt",\n        value="local-test-salt",\n        type="password",\n        help="Same salt + same column + same original value creates the same anonymized value.",\n    )\n    st.caption(\n        "Salt controls repeatability: same salt + same column + same original value = same anonymized value. "\n        "This preserves joins across files and chunks. Change the salt to generate a different anonymized dataset; "\n        "do not use a real password or API key."\n    )\n\n    with st.expander("Performance and safety settings", expanded=True):\n        col_a, col_b, col_c, col_d = st.columns(4)\n        with col_a:\n            resource_check_every = st.number_input(\n                "Check resources every N batches",\n                min_value=1,\n                max_value=100,\n                value=DEFAULT_RESOURCE_CHECK_EVERY,\n                step=1,\n            )\n        with col_b:\n            status_update_every = st.number_input(\n                "Update status every N batches",\n                min_value=1,\n                max_value=100,\n                value=DEFAULT_STATUS_UPDATE_EVERY,\n                step=1,\n            )\n        with col_c:\n            min_free_disk_gb = st.number_input(\n                "Stop if free disk below GB",\n                min_value=1.0,\n                max_value=10_000.0,\n                value=DEFAULT_MIN_FREE_DISK_GB,\n                step=1.0,\n            )\n        with col_d:\n            min_free_memory_gb = st.number_input(\n                "Stop if memory below GB",\n                min_value=0.0,\n                max_value=1024.0,\n                value=DEFAULT_MIN_FREE_MEMORY_GB,\n                step=1.0,\n                help="Requires psutil. If psutil is not installed, disk checks still run.",\n            )\n\n        try:\n            snapshot = get_resource_snapshot(output_dir)\n            st.info("Current resources for output folder: " + format_resource_line(snapshot))\n        except Exception as exc:\n            st.warning(f"Could not check current resources yet: {exc}")\n        if psutil is None:\n            st.warning("psutil is not installed, so memory checks are disabled. Add `psutil>=5.9,<6` to requirements.txt.")\n\n    if st.button("Create anonymized chunks", type="primary", use_container_width=True):\n        Path(output_dir).mkdir(parents=True, exist_ok=True)\n        rules = selected_rules_from_df(st.session_state.rules_df)\n        active_rules = [rule for rule in rules if rule["action"] != ACTION_NONE]\n        sources = [SourceDescriptor.from_dict(item) for item in st.session_state.source_files]\n\n        manifest = {\n            "generated_at": datetime.now(timezone.utc).isoformat(),\n            "max_chunk_mb": max_chunk_mb,\n            "rows_per_read": rows_per_read,\n            "resource_check_every": resource_check_every,\n            "status_update_every": status_update_every,\n            "min_free_disk_gb": min_free_disk_gb,\n            "min_free_memory_gb": min_free_memory_gb,\n            "source_files": [source.to_dict() for source in sources],\n            "active_rules": active_rules,\n            "outputs": [],\n            "stopped_early": False,\n        }\n\n        started = time.time()\n        status = st.status("Processing files...", expanded=True)\n        try:\n            assert_resources_ok(output_dir, min_free_disk_gb, min_free_memory_gb)\n            for source in sources:\n                result = write_chunked_csv(\n                    source=source,\n                    output_dir=output_dir,\n                    rules=rules,\n                    salt=salt,\n                    max_chunk_mb=max_chunk_mb,\n                    rows_per_read=rows_per_read,\n                    progress_text=status,\n                    resource_check_every=resource_check_every,\n                    status_update_every=status_update_every,\n                    min_free_disk_gb=min_free_disk_gb,\n                    min_free_memory_gb=min_free_memory_gb,\n                )\n                manifest["outputs"].append(result)\n\n            manifest["elapsed_seconds"] = round(time.time() - started, 2)\n            manifest_path = Path(output_dir) / "anonymization_manifest.json"\n            with open(manifest_path, "w", encoding="utf-8") as handle:\n                json.dump(manifest, handle, indent=2)\n            manifest["manifest_path"] = str(manifest_path)\n            st.session_state.last_manifest = manifest\n            status.update(label="Processing complete", state="complete", expanded=False)\n            st.success(f"Created anonymized output in: {output_dir}")\n        except ResourceLimitError as exc:\n            manifest["stopped_early"] = True\n            manifest["elapsed_seconds"] = round(time.time() - started, 2)\n            if exc.partial_result:\n                manifest["outputs"].append(exc.partial_result)\n            manifest["stop_reason"] = str(exc)\n            manifest_path = Path(output_dir) / "anonymization_manifest_partial.json"\n            with open(manifest_path, "w", encoding="utf-8") as handle:\n                json.dump(manifest, handle, indent=2)\n            manifest["manifest_path"] = str(manifest_path)\n            st.session_state.last_manifest = manifest\n            status.update(label="Processing stopped safely", state="error", expanded=True)\n            st.warning(str(exc))\n            st.info(f"Completed output chunks were preserved. Partial manifest written to: {manifest_path}")\n        except Exception as exc:\n            manifest["stopped_early"] = True\n            manifest["elapsed_seconds"] = round(time.time() - started, 2)\n            manifest["stop_reason"] = str(exc)\n            manifest_path = Path(output_dir) / "anonymization_manifest_failed.json"\n            try:\n                with open(manifest_path, "w", encoding="utf-8") as handle:\n                    json.dump(manifest, handle, indent=2)\n                manifest["manifest_path"] = str(manifest_path)\n                st.session_state.last_manifest = manifest\n            except Exception:\n                pass\n            status.update(label="Processing failed", state="error", expanded=True)\n            st.error(str(exc))\n\n\ndef render_manifest_section():\n    manifest = st.session_state.last_manifest\n    if not manifest:\n        return\n\n    st.subheader("5. Output summary")\n    output_rows = []\n    for item in manifest.get("outputs", []):\n        source_info = item.get("source", {})\n        for path in item.get("output_files", []):\n            p = Path(path)\n            output_rows.append(\n                {\n                    "source": source_info.get("display_name", source_info.get("source_name", "")),\n                    "output_file": path,\n                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2) if p.exists() else "missing",\n                    "rows_from_source": item.get("rows_written", 0),\n                    "batches_read": item.get("batches_read", 0),\n                    "stopped_early": item.get("stopped_early", False),\n                }\n            )\n    if output_rows:\n        st.dataframe(pd.DataFrame(output_rows), use_container_width=True, hide_index=True)\n    if manifest.get("stopped_early"):\n        st.warning(f"Run stopped early: {manifest.get(\'stop_reason\', \'unknown reason\')}")\n    st.info(f"Manifest written to: {manifest.get(\'manifest_path\')}")\n\n\ndef render_notes():\n    with st.expander("How to use this app", expanded=True):\n        st.markdown(\n            """\n            Use this local tool before loading data into the AI Data Planner. Its job is to turn raw CSV exports into anonymized CSV chunks that are safer and easier for the planner to handle.\n\n            **Basic workflow**\n\n            1. Use **Clean up old temp files** if prior tests left large temporary folders behind.\n            2. Choose one or more CSV or ZIP files. For very large files, paste local file or folder paths instead of using browser upload.\n            3. Click **Inspect columns**. The app reads file headers and previews a small sample without fully loading the dataset into memory.\n            4. Review the suggested anonymization rules. The suggestions are based on column names, so they should always be checked before processing.\n            5. Set the output folder, chunk size, rows processed at a time, anonymization salt, and resource-safety settings.\n            6. Click **Create anonymized chunks**. The app writes anonymized CSV parts plus an `anonymization_manifest.json` report.\n\n            **CSV and ZIP inputs**\n\n            - ZIP files are streamed directly; CSV files inside them are not fully extracted before processing.\n            - If a ZIP contains multiple CSVs, each CSV source is detected and processed.\n            - For large ZIPs, make sure your computer has enough free disk space for output chunks and temporary working files.\n            - Local paths are best for multi-GB files because the app can read directly from disk instead of moving the file through the browser.\n            - Paste one local path per line. Windows paths copied with quotation marks are okay.\n\n            **How anonymization works**\n\n            - The same original value in the same column becomes the same anonymized value when the salt is unchanged.\n            - This preserves joins across files and chunks. For example, the same `UserId` can still match across users, enrollments, and activity tables after anonymization.\n            - Name misspellings, fake IDs, fake emails, URL/domain replacements, redactions, static replacements, and hash tokens are applied only to the columns you select.\n            - The manifest records which rules were used, but it does not store original-to-anonymized mappings.\n\n            **How to treat the salt**\n\n            - The salt is a private phrase used in the anonymization recipe.\n            - Use the same salt when processing files that need to relate to each other.\n            - Change the salt when you want to create a different anonymized version of the same source data.\n            - Do not use a real password, API key, or other credential as the salt.\n            - Keep the salt reasonably private. Someone with the source data, the salt, and the same rules could reproduce the same anonymized values.\n\n            **Performance and safety**\n\n            - Browser upload is still limited by Streamlit settings and local machine memory. For very large files, use local paths.\n            - Resource-aware processing checks free disk space and, when `psutil` is installed, available memory between batches.\n            - If the app stops early, it is trying to avoid exhausting the machine. Check the run time, stop reason, output folder, free disk space, and any partial output files.\n            - Smaller **Rows processed at a time** values use less memory. Larger values may run faster but can increase memory pressure.\n\n            **After processing**\n\n            - Load the anonymized output chunks into the AI Data Planner, not the original raw files.\n            - Keep the original raw files, salts, and temporary folders out of any shared workspace.\n            - Use the manifest to confirm which files were processed, which rules were applied, how long the run took, and whether the run stopped early.\n            """\n        )\n\n\ndef main():\n    ensure_state()\n    st.title(APP_TITLE)\n    st.caption("Prepare sanitized CSV chunks for the AI Data Planner")\n    render_notes()\n    render_cleanup_section()\n    render_source_section()\n    render_profile_section()\n    render_rules_section()\n    render_process_section()\n    render_manifest_section()\n\n\nif __name__ == "__main__":\n    main()\n'

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


def get_all_zips(root_dir):
    zip_files = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(".zip"):
                zip_files.append(os.path.join(root, file_name))
    return sorted(zip_files)


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


def parse_local_source_entries(raw_text):
    entries = []
    for line in (raw_text or "").splitlines():
        for piece in line.split(","):
            cleaned = piece.strip().strip('"').strip("'")
            if cleaned:
                entries.append(cleaned)
    return entries


def collect_local_source_files(source_entries, extraction_dir, status):
    csv_paths = []
    seen_csv = set()
    zip_paths = []

    for entry in source_entries:
        abs_entry = os.path.abspath(os.path.expanduser(entry))
        if not os.path.exists(abs_entry):
            raise ValueError(f"Local source not found: {entry}")

        if os.path.isdir(abs_entry):
            for csv_path in get_all_csvs(abs_entry):
                if csv_path not in seen_csv:
                    seen_csv.add(csv_path)
                    csv_paths.append(csv_path)
            for zip_path in get_all_zips(abs_entry):
                zip_paths.append(zip_path)
        elif abs_entry.lower().endswith(".csv"):
            if abs_entry not in seen_csv:
                seen_csv.add(abs_entry)
                csv_paths.append(abs_entry)
        elif abs_entry.lower().endswith(".zip"):
            zip_paths.append(abs_entry)
        else:
            raise ValueError(f"Unsupported local source type: {entry}")

    for zip_path in zip_paths:
        status.write(f"📂 Extracting local ZIP `{os.path.basename(zip_path)}`")
        safe_extract(zip_path, extraction_dir)

    for csv_path in get_all_csvs(extraction_dir):
        if csv_path not in seen_csv:
            seen_csv.add(csv_path)
            csv_paths.append(csv_path)

    return sorted(csv_paths)


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


def analyze_join_coverage(conn, metadata, artifacts_dir):
    coverage_rows = []
    table_map = metadata.get("tables", {})
    overlaps = metadata.get("table_overlaps", [])

    for item in overlaps:
        left_table = item["left_table"]
        right_table = item["right_table"]
        left_info = table_map.get(left_table, {})
        right_info = table_map.get(right_table, {})
        if not left_info or not right_info:
            continue

        left_pattern = os.path.join(artifacts_dir, left_info["file_pattern"]).replace("\\", "/")
        right_pattern = os.path.join(artifacts_dir, right_info["file_pattern"]).replace("\\", "/")

        best_result = None
        for shared in item.get("shared_key_columns", [])[:4]:
            left_col = shared["left_column"]
            right_col = shared["right_column"]
            try:
                stats = conn.execute(
                    f"""
                    WITH left_keys AS (
                        SELECT DISTINCT TRIM(CAST("{left_col}" AS VARCHAR)) AS join_key
                        FROM read_parquet('{left_pattern}')
                        WHERE "{left_col}" IS NOT NULL
                        LIMIT {JOIN_COVERAGE_SAMPLE_LIMIT}
                    ),
                    right_keys AS (
                        SELECT DISTINCT TRIM(CAST("{right_col}" AS VARCHAR)) AS join_key
                        FROM read_parquet('{right_pattern}')
                        WHERE "{right_col}" IS NOT NULL
                        LIMIT {JOIN_COVERAGE_SAMPLE_LIMIT}
                    ),
                    overlap AS (
                        SELECT COUNT(*) AS overlap_distinct
                        FROM left_keys lk
                        INNER JOIN right_keys rk ON lk.join_key = rk.join_key
                    )
                    SELECT
                        (SELECT COUNT(*) FROM left_keys) AS left_distinct,
                        (SELECT COUNT(*) FROM right_keys) AS right_distinct,
                        (SELECT overlap_distinct FROM overlap) AS overlap_distinct
                    """
                ).fetchone()
            except Exception as exc:
                logging.info(
                    "Join coverage failed for %s.%s <-> %s.%s: %s",
                    left_table,
                    left_col,
                    right_table,
                    right_col,
                    exc,
                )
                continue

            left_distinct = int(stats[0] or 0)
            right_distinct = int(stats[1] or 0)
            overlap_distinct = int(stats[2] or 0)
            if left_distinct == 0 or right_distinct == 0:
                continue

            left_coverage = round((overlap_distinct / left_distinct) * 100, 2)
            right_coverage = round((overlap_distinct / right_distinct) * 100, 2)
            avg_coverage = round((left_coverage + right_coverage) / 2, 2)
            result = {
                "left_table": left_table,
                "right_table": right_table,
                "left_column": left_col,
                "right_column": right_col,
                "left_distinct_keys": left_distinct,
                "right_distinct_keys": right_distinct,
                "overlap_distinct_keys": overlap_distinct,
                "left_coverage_pct": left_coverage,
                "right_coverage_pct": right_coverage,
                "average_coverage_pct": avg_coverage,
                "sample_limited": left_distinct >= JOIN_COVERAGE_SAMPLE_LIMIT or right_distinct >= JOIN_COVERAGE_SAMPLE_LIMIT,
            }
            if not best_result or (
                result["overlap_distinct_keys"],
                result["average_coverage_pct"],
            ) > (
                best_result["overlap_distinct_keys"],
                best_result["average_coverage_pct"],
            ):
                best_result = result

        if best_result:
            coverage_rows.append(best_result)

    coverage_rows.sort(
        key=lambda row: (
            row["overlap_distinct_keys"],
            row["average_coverage_pct"],
            row["left_coverage_pct"],
            row["right_coverage_pct"],
        ),
        reverse=True,
    )
    return coverage_rows


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
        "join_coverage": metadata.get("join_coverage", [])[:MAX_JOIN_COVERAGE_HINTS],
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


def process_input_sources(uploaded_files, local_source_text, strategy):
    if not uploaded_files and not (local_source_text or "").strip():
        raise ValueError("Please upload files or provide local CSV/ZIP paths.")

    root, bundle_dir, upload_dir, artifacts_dir = build_session_paths()
    robust_rmtree(root)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    status = st.status("🚀 Processing uploaded data...", expanded=True)
    start_time = time.time()
    conn = None
    try:
        total_uploaded_mb = 0.0
        csv_paths = []
        source_mode = "cloud_upload"

        if uploaded_files:
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
        else:
            source_mode = "local_filesystem"
            local_entries = parse_local_source_entries(local_source_text)
            csv_paths = collect_local_source_files(local_entries, upload_dir, status)

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
            "source_mode": source_mode,
            "header_map": header_map,
            "tables": result["tables"],
            "columns": result["columns"],
            "relationships": result["relationships"],
        }
        metadata["table_overlaps"] = analyze_table_overlaps(metadata)
        metadata["join_coverage"] = analyze_join_coverage(conn, metadata, artifacts_dir)
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


def strip_trailing_sql_comments(sql_text):
    if not sql_text:
        return sql_text
    cleaned = sql_text.rstrip()
    while True:
        lines = cleaned.splitlines()
        if not lines:
            return ""
        last_line = lines[-1].strip()
        if not last_line:
            cleaned = "\n".join(lines[:-1]).rstrip()
            continue
        if last_line.startswith("--"):
            cleaned = "\n".join(lines[:-1]).rstrip()
            continue
        if cleaned.endswith("*/"):
            start = cleaned.rfind("/*")
            if start != -1:
                trailer = cleaned[start:]
                if "\n" not in trailer or trailer.strip().startswith("/*"):
                    cleaned = cleaned[:start].rstrip()
                    continue
        break
    return cleaned


def split_sql_and_trailing_comments(sql_text):
    text = strip_leading_sql_comments((sql_text or "").strip())
    if not text:
        return "", ""

    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False

    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_single:
            if ch == "'" and nxt == "'":
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "-" and nxt == "-":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue
        if ch == ";":
            statement = text[:i].rstrip()
            trailing = text[i + 1 :].strip()
            return statement, trailing
        i += 1

    return text.strip(), ""


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
    table_columns = {}
    for column in metadata.get("columns", []):
        column_counts[column["table"]] = column_counts.get(column["table"], 0) + 1
        table_columns.setdefault(column["table"], []).append(column["name"])
    rows = []
    for table_name, info in metadata.get("tables", {}).items():
        cols = table_columns.get(table_name, [])
        rows.append(
            {
                "table_name": table_name,
                "row_count": info.get("total_rows", 0),
                "column_count": column_counts.get(table_name, 0),
                "table_kind": infer_table_kind(table_name, cols),
                "appears_to_represent": infer_table_description(table_name, cols),
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

    if ("join coverage" in q or "join quality" in q or "best join" in q) and metadata.get("join_coverage"):
        rows = []
        for item in metadata.get("join_coverage", []):
            rows.append(
                {
                    "left_table": item["left_table"],
                    "right_table": item["right_table"],
                    "left_column": item["left_column"],
                    "right_column": item["right_column"],
                    "overlap_distinct_keys": item["overlap_distinct_keys"],
                    "left_coverage_pct": item["left_coverage_pct"],
                    "right_coverage_pct": item["right_coverage_pct"],
                    "sample_limited": item.get("sample_limited", False),
                }
            )
        return {
            "dataframe": pd.DataFrame(rows),
            "answer": "The table below shows measured join coverage for likely key pairs across the processed tables.",
            "title": "Answered directly from join coverage analysis",
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


def get_join_coverage_hints(metadata):
    hints = []
    for item in metadata.get("join_coverage", [])[:MAX_JOIN_COVERAGE_HINTS]:
        sample_note = " (sample-limited)" if item.get("sample_limited") else ""
        hints.append(
            "- "
            f"{item['left_table']}.{item['left_column']} <-> {item['right_table']}.{item['right_column']} "
            f"| overlap keys={item['overlap_distinct_keys']} "
            f"| left coverage={item['left_coverage_pct']}% "
            f"| right coverage={item['right_coverage_pct']}%{sample_note}"
        )
    return hints


def build_relationship_context(metadata):
    relationships = [
        f"- {item['from_table']}.{item['from_column']} -> {item['to_table']}.{item['to_column']}"
        for item in metadata.get("relationships", [])
    ]
    overlap_hints = get_overlap_relationship_hints(metadata)
    coverage_hints = get_join_coverage_hints(metadata)
    blocks = []
    if relationships:
        blocks.append("KNOWN RELATIONSHIPS:\n" + "\n".join(relationships))
    if overlap_hints:
        blocks.append("OVERLAP HINTS:\n" + "\n".join(overlap_hints))
    if coverage_hints:
        blocks.append("JOIN COVERAGE HINTS:\n" + "\n".join(coverage_hints))
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
        "join_coverage": dataset_profile.get("join_coverage", [])[:MAX_JOIN_COVERAGE_HINTS],
    }
    return json.dumps(compact, indent=2)


def build_supplemental_table_review(metadata, table_names):
    if not table_names:
        return []

    profile_tables = {table["table_name"]: table for table in (metadata.get("dataset_profile", {}) or {}).get("tables", [])}
    overlap_items = metadata.get("table_overlaps", [])
    join_coverage_items = metadata.get("join_coverage", [])
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

        join_coverage = []
        for item in join_coverage_items:
            if item["left_table"] == table_name or item["right_table"] == table_name:
                other = item["right_table"] if item["left_table"] == table_name else item["left_table"]
                join_coverage.append(
                    {
                        "related_table": other,
                        "join_pair": f"{item['left_column']} <-> {item['right_column']}",
                        "overlap_distinct_keys": item.get("overlap_distinct_keys", 0),
                        "left_coverage_pct": item.get("left_coverage_pct"),
                        "right_coverage_pct": item.get("right_coverage_pct"),
                        "sample_limited": item.get("sample_limited", False),
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
                "join_coverage_hints": join_coverage[:5],
            }
        )

    return reviews


def should_include_zero_activity_rows(question):
    q = (question or "").lower()
    include_markers = [
        "summary",
        "role-level",
        "one row per",
        "show ",
        "by role",
        "posting percentage",
        "engagement",
        "all roles",
        "per role",
    ]
    exclude_markers = [
        "who posted",
        "users who posted",
        "posted at least one",
        "among posters",
        "post authors",
        "authors who posted",
        "only posters",
    ]
    return any(marker in q for marker in include_markers) and not any(marker in q for marker in exclude_markers)


def default_join_key_pairs(metadata, required_tables):
    required_set = set(required_tables or [])
    pairs = []
    for item in metadata.get("join_coverage", []):
        if item["left_table"] in required_set and item["right_table"] in required_set:
            pairs.append(
                {
                    "left_table": item["left_table"],
                    "left_column": item["left_column"],
                    "right_table": item["right_table"],
                    "right_column": item["right_column"],
                    "coverage_note": (
                        f"overlap={item['overlap_distinct_keys']}, "
                        f"left={item['left_coverage_pct']}%, right={item['right_coverage_pct']}%"
                    ),
                }
            )
    return pairs[:4]


def make_fallback_plan(question, metadata):
    explicit_tables = find_explicit_table_mentions(question, metadata)
    if not explicit_tables:
        explicit_tables = list(metadata.get("tables", {}).keys())[:2]
    include_zero_rows = should_include_zero_activity_rows(question)
    return {
        "question_type": "analysis",
        "required_tables": explicit_tables,
        "optional_tables_for_cursory_review": [],
        "likely_grain": "depends_on_question",
        "population_scope": "full_population" if include_zero_rows else "question_specific_subset",
        "include_zero_activity_rows": include_zero_rows,
        "join_key_pairs": default_join_key_pairs(metadata, explicit_tables),
        "join_strategy": "Prefer detected relationships and shared key columns such as UserId or OrgUnitId.",
        "metrics": ["Answer the user question with careful use of distinct counts where appropriate."],
        "filters": [],
        "duplication_risks": ["Joins may multiply rows; consider pre-aggregation or COUNT(DISTINCT ...)."],
        "assumptions": [],
        "reasoning_notes": [
            "Fallback plan used because planner JSON could not be parsed.",
            "Prefer join paths with stronger measured key overlap when several options are available.",
        ],
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
- population_scope
- include_zero_activity_rows
- join_key_pairs
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
3. Prefer exact shared keys, explicit relationship hints, and stronger measured join coverage over loose joins.
4. If a table has very weak practical key overlap with the main analytical path, say so and consider moving it to optional_tables_for_cursory_review.
5. Call out duplication risks whenever enrollments or event tables may multiply rows.
6. When both UserId and OrgUnitId are available between tables, consider whether UserId + OrgUnitId is a more faithful join than UserId alone.
7. Populate join_key_pairs with the exact join columns the SQL should use. Each item should include left_table, left_column, right_table, and right_column.
8. If the question implies a full summary by role or group, set population_scope to full_population and include_zero_activity_rows to true so zero-activity groups remain visible.
9. If the question is specifically about posters or active users only, you may set population_scope to filtered_subset and include_zero_activity_rows to false.
10. If the question asks for ordering by a metric, reflect that in the plan notes so the final SQL can preserve the intended ranking.
11. sufficiency_confidence should be low, medium, or high.
12. Be concrete and concise.
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
    normalized["population_scope"] = str(parsed.get("population_scope", "")).strip() or (
        "full_population" if should_include_zero_activity_rows(question) else "question_specific_subset"
    )
    include_zero_raw = parsed.get("include_zero_activity_rows")
    if isinstance(include_zero_raw, bool):
        normalized["include_zero_activity_rows"] = include_zero_raw
    elif isinstance(include_zero_raw, str):
        normalized["include_zero_activity_rows"] = include_zero_raw.strip().lower() in {"true", "yes", "1"}
    else:
        normalized["include_zero_activity_rows"] = should_include_zero_activity_rows(question)
    join_key_pairs = []
    parsed_join_pairs = parsed.get("join_key_pairs")
    if isinstance(parsed_join_pairs, list):
        for item in parsed_join_pairs:
            if not isinstance(item, dict):
                continue
            left_table = str(item.get("left_table", "")).strip()
            left_column = str(item.get("left_column", "")).strip()
            right_table = str(item.get("right_table", "")).strip()
            right_column = str(item.get("right_column", "")).strip()
            if left_table and left_column and right_table and right_column:
                join_key_pairs.append(
                    {
                        "left_table": left_table,
                        "left_column": left_column,
                        "right_table": right_table,
                        "right_column": right_column,
                    }
                )
    normalized["join_key_pairs"] = join_key_pairs or default_join_key_pairs(
        {"join_coverage": dataset_profile.get("join_coverage", [])},
        normalized["required_tables"],
    )
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
    lines.append(f"- Population scope: {plan.get('population_scope', 'n/a')}")
    lines.append(f"- Include zero-activity rows: {plan.get('include_zero_activity_rows', 'n/a')}")
    lines.append(f"- Join strategy: {plan.get('join_strategy', 'n/a')}")
    join_pairs = plan.get("join_key_pairs") or []
    if join_pairs:
        formatted_pairs = ", ".join(
            f"{item['left_table']}.{item['left_column']} -> {item['right_table']}.{item['right_column']}"
            for item in join_pairs
            if isinstance(item, dict)
        )
        if formatted_pairs:
            lines.append(f"- Join key pairs: {formatted_pairs}")
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
6. If include_zero_activity_rows is true or population_scope is full_population, preserve the base population with LEFT JOINs to aggregated activity tables so zero-activity groups remain visible.
7. Use INNER JOIN only when the question is explicitly about posters, active records, or filtered subsets only.
8. If the question asks for sorting by a metric, the final SQL ORDER BY should use that metric rather than a label column whenever possible.
9. Default to LIMIT 50 unless the query is already aggregated.
10. Output only SQL, but leading SQL comments are allowed if helpful.
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

Repair rules:
1. Preserve the intended population scope from the plan.
2. If include_zero_activity_rows is true, do not filter away zero-activity groups with an INNER JOIN to the activity table.
3. Keep the SQL DuckDB-compatible and use only the available tables and columns.
4. If the question asked for ranking by a metric, keep the final ORDER BY aligned to that metric.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return _strip_markdown_sql(response.choices[0].message.content)


def validate_sql(sql_query, artifacts_dir):
    statement_sql, trailing_sql = split_sql_and_trailing_comments(sql_query)
    trailing_sql = strip_trailing_sql_comments(trailing_sql).strip()
    if trailing_sql:
        raise ValueError("Security alert: multiple SQL statements are not allowed.")

    clean_sql = strip_trailing_sql_comments(statement_sql).strip().rstrip(";").strip()
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


def sql_join_pair_present(sql_text, left_column, right_column):
    escaped_left = re.escape(left_column)
    escaped_right = re.escape(right_column)
    pattern = re.compile(
        rf"""(?is)
        (?:\b\w+\b\.)?"?{escaped_left}"?\s*=\s*(?:\b\w+\b\.)?"?{escaped_right}"?
        |
        (?:\b\w+\b\.)?"?{escaped_right}"?\s*=\s*(?:\b\w+\b\.)?"?{escaped_left}"?
        """
    )
    return bool(pattern.search(sql_text or ""))


def missing_join_key_pairs(sql_text, plan):
    missing = []
    for item in plan.get("join_key_pairs", []) or []:
        if not isinstance(item, dict):
            continue
        left_col = item.get("left_column")
        right_col = item.get("right_column")
        if not left_col or not right_col:
            continue
        if not sql_join_pair_present(sql_text, left_col, right_col):
            missing.append(item)
    return missing


def referenced_tables_in_sql(sql_text, metadata, artifacts_dir):
    found = []
    for table_name, info in metadata.get("tables", {}).items():
        pattern = os.path.join(artifacts_dir, info["file_pattern"]).replace("\\", "/")
        if pattern in sql_text:
            found.append(table_name)
    return sorted(set(found))


def result_looks_like_poster_only_subset(df, plan):
    if df.empty or not plan.get("include_zero_activity_rows"):
        return False

    lower_cols = {col.lower(): col for col in df.columns}
    if "distinct_users_who_posted" not in lower_cols or "distinct_enrolled_users" not in lower_cols:
        return False

    posters = pd.to_numeric(df[lower_cols["distinct_users_who_posted"]], errors="coerce")
    enrolled = pd.to_numeric(df[lower_cols["distinct_enrolled_users"]], errors="coerce")
    comparable = posters.notna() & enrolled.notna() & (enrolled > 0)
    if not comparable.any():
        return False

    all_equal = (posters[comparable] == enrolled[comparable]).all()
    has_posting_metric = False
    for col in df.columns:
        if any(token in col.lower() for token in ["posting_percentage", "total_discussion_posts", "distinct_users_who_posted"]):
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                has_posting_metric = True
                break
    return bool(all_equal and has_posting_metric)


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
    if "post" in question.lower():
        posting_cols = [col for col in df.columns if any(token in col.lower() for token in ["posted", "posts", "posting"])]
        if posting_cols:
            posting_values = pd.concat([pd.to_numeric(df[col], errors="coerce") for col in posting_cols], axis=1)
            if posting_values.notna().any().any() and posting_values.fillna(0).sum().sum() == 0:
                warnings.append(
                    "Posting-related metrics are all zero in the current result. That may be real, or it may mean the chosen join path is too restrictive."
                )
    if result_looks_like_poster_only_subset(df, plan):
        warnings.append(
            "The result looks like a poster-only subset even though the plan expected zero-activity groups to remain visible."
        )

    if plan.get("required_tables"):
        warnings.append(
            "Main analysis tables: "
            + ", ".join(plan["required_tables"])
            + ". Supplemental tables, if any, were reviewed separately."
        )
    return warnings


@st.cache_data(show_spinner=False)
def build_local_package_bytes(app_source, anonymizer_app_source, requirements_text, config_text):
    readme_text = f"""# Direction 2 Local Toolkit

This package includes two local Streamlit apps:

1. `csv_anonymizer_chunker.py` prepares raw CSV or ZIP exports by anonymizing selected columns and writing smaller CSV chunks.
2. `streamlit_app.py` runs the Direction 2 AI Data Planner locally for profiling, planning, SQL analysis, critique, and summarization.

## Recommended workflow

1. Run the anonymizer/chunker locally first.
2. Use local CSV/ZIP paths for large raw files whenever possible.
3. Review the detected columns and anonymization rules carefully.
4. Create anonymized CSV chunks and review the `anonymization_manifest.json` file.
5. Use the anonymized chunks in one of two ways:
   - upload them to the hosted Streamlit Community Cloud planner, or
   - run the included local AI planner for larger datasets or local-only testing.

## Recommended Python version

- Python 3.12

## Quick start: anonymize data locally

### Windows

1. Open Command Prompt in this folder.
2. Run `run_anonymizer.bat`.

### macOS / Linux

1. Open Terminal in this folder.
2. Run:
   `chmod +x run_anonymizer.sh`
3. Then run:
   `./run_anonymizer.sh`

### Manual setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run csv_anonymizer_chunker.py --server.maxUploadSize 10000
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run csv_anonymizer_chunker.py --server.maxUploadSize 10000
```

## Optional: run the AI planner locally

Run this after your data has been anonymized, especially when the anonymized chunks are still too large for hosted upload.

### Windows

1. Open Command Prompt in this folder.
2. Run `run_local_app.bat`.

### macOS / Linux

1. Open Terminal in this folder.
2. Run:
   `chmod +x run_local_app.sh`
3. Then run:
   `./run_local_app.sh`

### Manual setup

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

On Windows PowerShell:

```powershell
.venv\\Scripts\\Activate.ps1
streamlit run streamlit_app.py
```

## Secrets file for the AI planner

Create `.streamlit/secrets.toml` based on `.streamlit/secrets.toml.example` before using the AI planner's LLM features.

The anonymizer/chunker does not require an API key.

## Salt guidance for anonymization

The anonymizer uses a salt to make replacements repeatable. The same salt, column, and original value produce the same anonymized value, which preserves joins across files and chunks.

- Use the same salt for files that need to relate to each other.
- Change the salt if you want a different anonymized version of the same source data.
- Do not use a real password, API key, or other credential as the salt.
- Keep the salt reasonably private.

## Notes

- Use the AI planner with sanitized, anonymized, or dummy data only.
- For multi-entity LMS-style exports, `Keep files as separate tables` is usually the better planner choice.
- The local package uses a higher upload setting than the hosted app.
- For multi-GB data, prefer local path input over browser upload.
- Very large jobs still depend on your machine's RAM and disk.
"""

    secrets_example = """DEEPSEEK_API_KEY = "your-api-key-here"
OPENAI_API_KEY = "your-api-key-here"
XAI_API_KEY = "your-api-key-here"
"""

    run_bat = """@echo off
if not exist .venv (
    python -m venv .venv
)
call .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
"""

    run_sh = """#!/usr/bin/env bash
set -euo pipefail
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
"""

    run_anonymizer_bat = """@echo off
if not exist .venv (
    python -m venv .venv
)
call .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run csv_anonymizer_chunker.py --server.maxUploadSize 10000
"""

    run_anonymizer_sh = """#!/usr/bin/env bash
set -euo pipefail
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run csv_anonymizer_chunker.py --server.maxUploadSize 10000
"""

    anonymizer_readme = """# CSV Anonymizer and Chunker

Run this app locally before uploading data to the Direction 2 AI Data Planner.

## What it does

- Accepts CSV files, ZIP files containing CSVs, and local file/folder paths.
- Lets you choose which columns to anonymize.
- Supports fake names, fake IDs, fake emails, Brightspace URL/domain replacement, static replacement, redaction, and hash tokens.
- Writes anonymized CSV chunks with a configurable size cap.
- Writes an `anonymization_manifest.json` report describing the run.

## Recommended workflow

1. Start the app with `run_anonymizer.bat` on Windows or `./run_anonymizer.sh` on macOS/Linux.
2. Use local paths for large files.
3. Inspect columns and review suggested rules.
4. Use the same salt for files that need to join together.
5. Write anonymized chunks.
6. Load the anonymized chunks into the hosted or local AI planner.

## Salt guidance

The salt makes anonymized values repeatable. The same salt, column, and original value produce the same anonymized output.

Use the same salt when files need to relate to each other. Change the salt when you want a different anonymized version. Do not use a real password, API key, or credential as the salt.
"""

    if "psutil" not in requirements_text.lower():
        requirements_text = requirements_text.rstrip() + "\npsutil>=5.9,<6\n"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("streamlit_app.py", app_source)
        zf.writestr("csv_anonymizer_chunker.py", anonymizer_app_source)
        zf.writestr("requirements.txt", requirements_text)
        zf.writestr(".streamlit/config.toml", config_text)
        zf.writestr(".streamlit/secrets.toml.example", secrets_example)
        zf.writestr("README.md", readme_text)
        zf.writestr("README_anonymizer_chunker.md", anonymizer_readme)
        zf.writestr("run_local_app.bat", run_bat)
        zf.writestr("run_local_app.sh", run_sh)
        zf.writestr("run_anonymizer.bat", run_anonymizer_bat)
        zf.writestr("run_anonymizer.sh", run_anonymizer_sh)
    return buffer.getvalue()


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


def select_primary_metric(question, plan, df):
    if df.empty:
        return None

    numeric_cols = [col for col in df.select_dtypes(include=["number"]).columns.tolist() if df[col].notna().any()]
    if not numeric_cols:
        return None

    lowered = question.lower()
    ordered_candidates = []
    if "posting percentage" in lowered or "percentage" in lowered or "rate" in lowered:
        ordered_candidates.extend(["posting_percentage", "engagement_percentage"])
    if "average" in lowered or "avg" in lowered:
        ordered_candidates.extend(["avg_posts_per_posting_user", "average_posts_per_posting_user"])
    if "post" in lowered:
        ordered_candidates.extend(["total_discussion_posts", "distinct_users_who_posted"])
    if "enroll" in lowered:
        ordered_candidates.append("distinct_enrolled_users")

    ordered_candidates.extend(
        [
            "posting_percentage",
            "total_discussion_posts",
            "distinct_users_who_posted",
            "distinct_enrolled_users",
        ]
    )

    seen = set()
    for col in ordered_candidates:
        if col in numeric_cols and col not in seen:
            seen.add(col)
            return col

    percent_like = [col for col in numeric_cols if "percent" in col.lower() or "rate" in col.lower()]
    if percent_like:
        return percent_like[0]

    count_like = [
        col
        for col in numeric_cols
        if any(token in col.lower() for token in ["count", "total", "posts", "users", "rows"])
    ]
    if count_like:
        return count_like[0]
    return numeric_cols[0]


def compress_duplicate_risk_note(value):
    text = str(value or "").strip()
    if not text:
        return text
    lower = text.lower()
    if "no duplication risk" in lower or "low duplication risk" in lower:
        return "Low duplication risk"
    if "duplicate" in lower or "duplication" in lower:
        if "distinct" in lower or "pre-aggreg" in lower or "mitigat" in lower:
            return "Managed duplication risk"
        return "Potential duplication risk"
    return text


def prepare_result_for_display(question, plan, df):
    if df.empty:
        return df, None, None

    display_df = df.copy()
    metric_col = select_primary_metric(question, plan, display_df)
    category_col = None

    categorical_cols = display_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if categorical_cols:
        preferred_labels = ["rolename", "table_name", "left_table", "right_table", "column_name"]
        for preferred in preferred_labels:
            for col in categorical_cols:
                if col.lower() == preferred:
                    category_col = col
                    break
            if category_col:
                break
        if not category_col:
            category_col = categorical_cols[0]

    if metric_col:
        display_df = display_df.sort_values(
            by=[metric_col] + ([category_col] if category_col and category_col in display_df.columns else []),
            ascending=[False] + ([True] if category_col and category_col in display_df.columns else []),
            kind="stable",
        ).reset_index(drop=True)

    for col in display_df.columns:
        if "duplication" in col.lower() and display_df[col].dtype == "object":
            display_df[col] = display_df[col].map(compress_duplicate_risk_note)

    return display_df, metric_col, category_col


def attempt_visualization(df, question, plan, metric_col=None, category_col=None):
    try:
        if df.empty or len(df) < 2:
            return
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if len(df.columns) <= 2 and len(df) <= 5 and numeric_cols:
            return
        metric_col = metric_col or select_primary_metric(question, plan, df)
        if metric_col and metric_col in df.columns:
            series = pd.to_numeric(df[metric_col], errors="coerce")
            if series.notna().any() and series.fillna(0).sum() == 0:
                return
        st.caption("Auto-visualization")
        if category_col and metric_col and category_col in df.columns and metric_col in df.columns:
            chart_df = df.head(20).set_index(category_col)[metric_col]
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


def render_local_download_ui():
    app_path = Path(__file__).resolve()
    app_source = app_path.read_text(encoding="utf-8")

    requirements_path = app_path.with_name("requirements.txt")
    if requirements_path.exists():
        requirements_text = requirements_path.read_text(encoding="utf-8")
    else:
        requirements_text = "streamlit>=1.45,<2\nduckdb>=1.1,<2\npandas>=2.2,<3\npyarrow>=16,<22\nopenai>=1.30,<2\n"

    config_path = app_path.parent / ".streamlit" / "config.toml"
    if config_path.exists():
        config_text = config_path.read_text(encoding="utf-8")
    else:
        config_text = "[server]\nmaxUploadSize = 2048\n"

    local_config_text = """[server]
maxUploadSize = 10240
"""

    package_bytes = build_local_package_bytes(app_source, ANONYMIZER_APP_SOURCE, requirements_text, local_config_text)

    with st.expander("Prepare data locally, then analyze here or locally", expanded=False):
        st.markdown(
            """
            Use this local toolkit when your raw CSV/ZIP exports need anonymization before analysis, or when your anonymized datasets are too large for hosted upload.

            Recommended workflow:
            1. Download the local toolkit.
            2. Run `csv_anonymizer_chunker.py` locally to anonymize raw CSV/ZIP exports and create smaller CSV chunks.
            3. Upload the anonymized chunks to this hosted planner, or run the included local planner if the anonymized data is still too large for hosted upload.

            The download includes:
            - `csv_anonymizer_chunker.py` for local anonymization and chunking
            - `streamlit_app.py` for running this AI planner locally
            - shared `requirements.txt`
            - `.streamlit/config.toml`
            - `.streamlit/secrets.toml.example`
            - `README.md`
            - `README_anonymizer_chunker.md`
            - `run_anonymizer.bat` and `run_anonymizer.sh`
            - `run_local_app.bat` and `run_local_app.sh`
            """
        )
        st.download_button(
            "Download local anonymizer and planner toolkit",
            data=package_bytes,
            file_name="direction2_local_anonymizer_and_planner_toolkit.zip",
            mime="application/zip",
            use_container_width=True,
        )


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

    render_local_download_ui()

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

    local_source_text = st.text_area(
        "Local CSV/ZIP paths or folders for local runs",
        value="",
        placeholder="/path/to/big-export-folder\n/path/to/Discussion-Posts.zip",
        help="Use this in the downloaded local app to ingest large datasets directly from disk without browser upload.",
    )

    if st.button("Process uploads", type="primary", use_container_width=True):
        process_input_sources(uploaded_files, local_source_text, strategy)
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

                missing_join_pairs = missing_join_key_pairs(sql, plan)
                if missing_join_pairs:
                    status.write("Repairing SQL to enforce planned join keys")
                    pair_labels = ", ".join(
                        f"{item['left_table']}.{item['left_column']} -> {item['right_table']}.{item['right_column']}"
                        for item in missing_join_pairs
                    )
                    sql = fix_sql_query(
                        user_input,
                        sql,
                        f"Generated SQL did not clearly use the planned join keys: {pair_labels}",
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

                if result_looks_like_poster_only_subset(df, plan):
                    status.write("Result looks poster-only. Repairing SQL to preserve zero-activity groups.")
                    sql_retry = fix_sql_query(
                        user_input,
                        sql,
                        "The result appears to exclude zero-activity groups even though the plan expects a full-population summary. Rewrite the query to preserve the base population and LEFT JOIN aggregated activity.",
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

                display_df, metric_col, category_col = prepare_result_for_display(user_input, plan, df)

                st.dataframe(display_df, use_container_width=True)
                if not display_df.empty:
                    st.download_button(
                        "Download results as CSV",
                        data=display_df.to_csv(index=False),
                        file_name="query_results.csv",
                        mime="text/csv",
                    )
                    attempt_visualization(display_df, user_input, plan, metric_col=metric_col, category_col=category_col)

                warnings = critique_result(user_input, display_df, plan)
                if warnings:
                    st.markdown("**Critique**")
                    for warning in warnings:
                        st.warning(warning)

                status.write("Summarizing result")
                if display_df.empty:
                    answer = "The query returned no rows. Try broadening the question or checking the filter assumptions."
                else:
                    answer = summarize_answer(
                        user_input,
                        plan,
                        display_df,
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
