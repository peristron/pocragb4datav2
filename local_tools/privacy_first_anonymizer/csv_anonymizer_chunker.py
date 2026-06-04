# streamlit run csv_anonymizer_chunker.py
# streamlit run csv_anonymizer_chunker.py --server.maxUploadSize 10000
# cd 
#   psutil>=5.9,<6



import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import streamlit as st

try:
    import psutil
except ImportError:  # The app still works, but memory checks are disabled.
    psutil = None


APP_TITLE = "Local CSV Anonymizer and Chunker"
DEFAULT_TARGET_DOMAIN = "acme.brightspace.com"
DEFAULT_CHUNK_SIZE_MB = 500
MAX_CHUNK_SIZE_MB = 2048
DEFAULT_READ_ROWS = 100_000
PREVIEW_ROWS = 100
DEFAULT_RESOURCE_CHECK_EVERY = 10
DEFAULT_STATUS_UPDATE_EVERY = 10
DEFAULT_MIN_FREE_DISK_GB = 20.0
DEFAULT_MIN_FREE_MEMORY_GB = 4.0

ACTION_NONE = "No change"
ACTION_NAME = "Randomized name misspelling"
ACTION_NUMERIC_ID = "Random numeric ID"
ACTION_ALPHANUMERIC_ID = "Random alphanumeric ID"
ACTION_EMAIL = "Fake email"
ACTION_URL_DOMAIN = "Replace Brightspace URL/domain"
ACTION_STATIC = "Static replacement"
ACTION_REDACT = "Redact"
ACTION_HASH = "Stable hash token"

ACTION_OPTIONS = [
    ACTION_NONE,
    ACTION_NAME,
    ACTION_NUMERIC_ID,
    ACTION_ALPHANUMERIC_ID,
    ACTION_EMAIL,
    ACTION_URL_DOMAIN,
    ACTION_STATIC,
    ACTION_REDACT,
    ACTION_HASH,
]

LIKELY_NAME_PATTERNS = [r"\bname\b", r"first.?name", r"last.?name", r"full.?name"]
LIKELY_ID_PATTERNS = [r"\bid\b", r"user.?id", r"student.?id", r"employee.?id", r"person.?id"]
LIKELY_EMAIL_PATTERNS = [r"email", r"e.?mail"]
LIKELY_URL_PATTERNS = [r"url", r"link", r"domain", r"host", r"href"]
LIKELY_REDACT_PATTERNS = [r"password", r"secret", r"token", r"ssn", r"social.?sec"]


st.set_page_config(page_title=APP_TITLE, layout="wide")


@dataclass
class SourceDescriptor:
    kind: str
    display_name: str
    source_name: str
    size_bytes: int
    path: str = ""
    zip_path: str = ""
    zip_member: str = ""
    compressed_size_bytes: int = 0

    def to_dict(self):
        return {
            "kind": self.kind,
            "display_name": self.display_name,
            "source_name": self.source_name,
            "size_bytes": self.size_bytes,
            "path": self.path,
            "zip_path": self.zip_path,
            "zip_member": self.zip_member,
            "compressed_size_bytes": self.compressed_size_bytes,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class ResourceLimitError(RuntimeError):
    def __init__(self, message, partial_result=None):
        super().__init__(message)
        self.partial_result = partial_result


def ensure_state():
    defaults = {
        "source_signature": "",
        "source_files": [],
        "profile": None,
        "rules_df": None,
        "last_manifest": None,
        "work_dir": None,
        "cleanup_candidates": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def normalize_column_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def safe_table_stem(file_name):
    stem = Path(file_name).stem
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem).strip("._-")
    return clean or "data"


def bytes_to_mb(value):
    return round(float(value or 0) / (1024 * 1024), 2)


def bytes_to_gb(value):
    return round(float(value or 0) / (1024 * 1024 * 1024), 2)


def clean_local_path_piece(value):
    cleaned = (value or "").strip()
    cleaned = cleaned.strip("\ufeff").strip()
    # Windows Explorer often copies paths wrapped in straight or smart quotes.
    quote_pairs = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")]
    changed = True
    while changed and len(cleaned) >= 2:
        changed = False
        for left, right in quote_pairs:
            if cleaned.startswith(left) and cleaned.endswith(right):
                cleaned = cleaned[1:-1].strip()
                changed = True
    return cleaned


def parse_local_paths(raw_text):
    paths = []
    for raw_line in (raw_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Prefer one path per line. Comma splitting is supported only when csv parsing is unambiguous.
        try:
            parsed = next(csv.reader([line], skipinitialspace=True))
        except Exception:
            parsed = [line]
        pieces = parsed if len(parsed) > 1 else [line]
        for piece in pieces:
            cleaned = clean_local_path_piece(piece)
            if cleaned:
                paths.append(cleaned)
    return paths


TEMP_CLEANUP_PREFIXES = ["csv_anonymizer_", "streamlit_direction2"]


def get_directory_size_bytes(path):
    total = 0
    root = Path(path)
    if not root.exists():
        return 0
    for child in root.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def discover_cleanup_candidates(include_ai_planner=True, min_age_hours=0):
    temp_root = Path(tempfile.gettempdir())
    now = time.time()
    prefixes = ["csv_anonymizer_"]
    if include_ai_planner:
        prefixes.append("streamlit_direction2")

    current_work_dir = st.session_state.get("work_dir")
    current_work_dir = str(Path(current_work_dir).resolve()) if current_work_dir else ""

    candidates = []
    if not temp_root.exists():
        return candidates

    for child in temp_root.iterdir():
        try:
            if not child.is_dir():
                continue
            if not any(child.name.startswith(prefix) for prefix in prefixes):
                continue
            child_resolved = str(child.resolve())
            if current_work_dir and child_resolved == current_work_dir:
                continue
            stat = child.stat()
            age_hours = max(0.0, (now - stat.st_mtime) / 3600.0)
            if age_hours < float(min_age_hours):
                continue
            candidates.append(
                {
                    "path": str(child),
                    "name": child.name,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "age_hours": round(age_hours, 1),
                    "size_gb": bytes_to_gb(get_directory_size_bytes(child)),
                }
            )
        except OSError:
            continue

    return sorted(candidates, key=lambda item: item["size_gb"], reverse=True)


def cleanup_candidate_dirs(candidates):
    deleted = []
    failed = []
    for item in candidates:
        path = Path(item["path"])
        try:
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
            deleted.append(str(path))
        except Exception as exc:
            failed.append({"path": str(path), "error": str(exc)})
    return deleted, failed


def ensure_safe_zip_member(member_name):
    candidate = Path(member_name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_name}")


def csv_sources_from_zip(zip_path):
    sources = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            ensure_safe_zip_member(info.filename)
            if info.is_dir() or not info.filename.lower().endswith(".csv"):
                continue
            display_name = f"{Path(zip_path).name}::{info.filename}"
            sources.append(
                SourceDescriptor(
                    kind="zip_member",
                    display_name=display_name,
                    source_name=Path(info.filename).name,
                    size_bytes=int(info.file_size or 0),
                    zip_path=str(zip_path),
                    zip_member=info.filename,
                    compressed_size_bytes=int(info.compress_size or 0),
                )
            )
    return sources


def collect_local_sources(raw_text):
    sources = []
    seen = set()

    for raw_path in parse_local_paths(raw_text):
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Local source not found: {raw_path}")

        candidates = []
        if path.is_dir():
            candidates = [child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in {".csv", ".zip"}]
        else:
            candidates = [path]

        for candidate in candidates:
            key = str(candidate)
            if candidate.suffix.lower() == ".csv":
                if key in seen:
                    continue
                seen.add(key)
                sources.append(
                    SourceDescriptor(
                        kind="csv",
                        display_name=candidate.name,
                        source_name=candidate.name,
                        size_bytes=candidate.stat().st_size,
                        path=str(candidate),
                    )
                )
            elif candidate.suffix.lower() == ".zip":
                for source in csv_sources_from_zip(candidate):
                    zip_key = f"{source.zip_path}::{source.zip_member}"
                    if zip_key not in seen:
                        seen.add(zip_key)
                        sources.append(source)
            else:
                raise ValueError(f"Unsupported source type: {raw_path}")

    return sorted(sources, key=lambda item: item.display_name.lower())


def save_uploaded_sources(uploaded_files, work_dir):
    upload_dir = Path(work_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    seen = set()

    for uploaded in uploaded_files or []:
        target_path = upload_dir / Path(uploaded.name).name
        with open(target_path, "wb") as handle:
            handle.write(uploaded.getbuffer())

        if uploaded.name.lower().endswith(".csv"):
            sources.append(
                SourceDescriptor(
                    kind="csv",
                    display_name=target_path.name,
                    source_name=target_path.name,
                    size_bytes=target_path.stat().st_size,
                    path=str(target_path),
                )
            )
        elif uploaded.name.lower().endswith(".zip"):
            for source in csv_sources_from_zip(target_path):
                key = f"{source.zip_path}::{source.zip_member}"
                if key not in seen:
                    seen.add(key)
                    sources.append(source)

    return sorted(sources, key=lambda item: item.display_name.lower())


@contextmanager
def open_source_binary(source):
    if source.kind == "csv":
        with open(source.path, "rb") as handle:
            yield handle
    elif source.kind == "zip_member":
        with zipfile.ZipFile(source.zip_path, "r") as zf:
            ensure_safe_zip_member(source.zip_member)
            with zf.open(source.zip_member, "r") as handle:
                yield handle
    else:
        raise ValueError(f"Unsupported source kind: {source.kind}")


def read_csv_preview(source, nrows):
    try:
        with open_source_binary(source) as handle:
            return pd.read_csv(handle, nrows=nrows, encoding_errors="replace", on_bad_lines="skip")
    except UnicodeDecodeError:
        with open_source_binary(source) as handle:
            return pd.read_csv(handle, nrows=nrows, encoding="latin-1", on_bad_lines="skip")


def read_csv_chunks(source, rows_per_chunk):
    try:
        with open_source_binary(source) as handle:
            yield from pd.read_csv(handle, chunksize=rows_per_chunk, encoding_errors="replace", on_bad_lines="skip")
    except UnicodeDecodeError:
        with open_source_binary(source) as handle:
            yield from pd.read_csv(handle, chunksize=rows_per_chunk, encoding="latin-1", on_bad_lines="skip")


def infer_action(column_name):
    normalized = normalize_column_name(column_name)
    pattern_groups = [
        (LIKELY_REDACT_PATTERNS, ACTION_REDACT),
        (LIKELY_EMAIL_PATTERNS, ACTION_EMAIL),
        (LIKELY_URL_PATTERNS, ACTION_URL_DOMAIN),
        (LIKELY_NAME_PATTERNS, ACTION_NAME),
        (LIKELY_ID_PATTERNS, ACTION_NUMERIC_ID),
    ]
    for patterns, action in pattern_groups:
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
            return action
    return ACTION_NONE


def profile_source(source):
    preview = read_csv_preview(source, PREVIEW_ROWS)
    columns = [str(col) for col in preview.columns]
    return {
        "source": source.to_dict(),
        "file_name": source.display_name,
        "source_name": source.source_name,
        "size_mb": bytes_to_mb(source.size_bytes),
        "compressed_size_mb": bytes_to_mb(source.compressed_size_bytes),
        "columns": columns,
        "preview": preview,
    }


def build_profile(sources):
    files = [profile_source(source) for source in sources]
    all_columns = []
    for item in files:
        for column in item["columns"]:
            if column not in all_columns:
                all_columns.append(column)
    return {"files": files, "all_columns": all_columns}


def initial_rules_df(columns):
    return pd.DataFrame(
        [
            {
                "column": column,
                "action": infer_action(column),
                "id_length": 10,
                "static_value": "",
                "target_domain": DEFAULT_TARGET_DOMAIN,
            }
            for column in columns
        ]
    )


def digest_text(value, salt, length=32):
    text = "" if pd.isna(value) else str(value)
    return hashlib.sha256(f"{salt}|{text}".encode("utf-8", errors="replace")).hexdigest()[:length]


def seeded_random(value, salt):
    seed_hex = digest_text(value, salt, length=16)
    return random.Random(int(seed_hex, 16))


def randomized_name(value, salt):
    if pd.isna(value) or str(value).strip() == "":
        return value

    rng = seeded_random(value, salt)
    parts = re.split(r"(\s+|-|'|\.)", str(value))
    out_parts = []
    vowels = "aeiou"

    for part in parts:
        if not part.isalpha() or len(part) < 3:
            out_parts.append(part)
            continue

        chars = list(part.lower())
        choice = rng.choice(["swap", "vowel", "drop", "double"])
        if choice == "swap" and len(chars) > 3:
            idx = rng.randrange(1, len(chars) - 1)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif choice == "vowel":
            vowel_positions = [idx for idx, char in enumerate(chars) if char in vowels]
            if vowel_positions:
                idx = rng.choice(vowel_positions)
                chars[idx] = rng.choice([v for v in vowels if v != chars[idx]])
        elif choice == "drop" and len(chars) > 4:
            chars.pop(rng.randrange(1, len(chars) - 1))
        elif choice == "double" and len(chars) > 3:
            idx = rng.randrange(1, len(chars) - 1)
            chars.insert(idx, chars[idx])

        new_part = "".join(chars)
        if part[0].isupper():
            new_part = new_part.capitalize()
        out_parts.append(new_part)

    return "".join(out_parts)


def numeric_id(value, salt, length):
    if pd.isna(value) or str(value).strip() == "":
        return value
    digest = digest_text(value, salt, length=64)
    digits = "".join(str(int(char, 16) % 10) for char in digest)
    digits = digits[: max(1, int(length))]
    if digits and digits[0] == "0":
        digits = "7" + digits[1:]
    return digits


def alphanumeric_id(value, salt, length):
    if pd.isna(value) or str(value).strip() == "":
        return value
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    rng = seeded_random(value, salt)
    return "".join(rng.choice(alphabet) for _ in range(max(1, int(length))))


def fake_email(value, salt):
    if pd.isna(value) or str(value).strip() == "":
        return value
    return f"user_{alphanumeric_id(value, salt, 12).lower()}@example.invalid"


def replace_brightspace_domain(value, target_domain):
    if pd.isna(value) or str(value).strip() == "":
        return value

    text = str(value)
    domain = (target_domain or DEFAULT_TARGET_DOMAIN).strip()

    if "://" in text:
        try:
            parsed = urlsplit(text)
            if parsed.netloc.lower().endswith("brightspace.com"):
                return urlunsplit((parsed.scheme, domain, parsed.path, parsed.query, parsed.fragment))
        except Exception:
            pass

    return re.sub(r"\b[a-zA-Z0-9.-]+\.brightspace\.com\b", domain, text)


def stable_hash_token(value, salt):
    if pd.isna(value) or str(value).strip() == "":
        return value
    return "anon_" + digest_text(value, salt, length=16)


def apply_rules_to_chunk(df, rules, salt):
    output = df.copy()
    for rule in rules:
        column = rule["column"]
        if column not in output.columns:
            continue

        action = rule["action"]
        id_length = int(rule.get("id_length") or 10)
        static_value = str(rule.get("static_value") or "")
        target_domain = str(rule.get("target_domain") or DEFAULT_TARGET_DOMAIN)
        column_salt = f"{salt}|{column}"

        if action == ACTION_NONE:
            continue
        if action == ACTION_NAME:
            output[column] = output[column].map(lambda value: randomized_name(value, column_salt))
        elif action == ACTION_NUMERIC_ID:
            output[column] = output[column].map(lambda value: numeric_id(value, column_salt, id_length))
        elif action == ACTION_ALPHANUMERIC_ID:
            output[column] = output[column].map(lambda value: alphanumeric_id(value, column_salt, id_length))
        elif action == ACTION_EMAIL:
            output[column] = output[column].map(lambda value: fake_email(value, column_salt))
        elif action == ACTION_URL_DOMAIN:
            output[column] = output[column].map(lambda value: replace_brightspace_domain(value, target_domain))
        elif action == ACTION_STATIC:
            output[column] = static_value
        elif action == ACTION_REDACT:
            output[column] = "[REDACTED]"
        elif action == ACTION_HASH:
            output[column] = output[column].map(lambda value: stable_hash_token(value, column_salt))

    return output


def open_new_output_file(output_dir, source_stem, part_number):
    output_path = Path(output_dir) / f"{source_stem}_anonymized_part_{part_number:03d}.csv"
    handle = open(output_path, "wb")
    return output_path, handle


def csv_bytes_for_frame(df, include_header):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=include_header, quoting=csv.QUOTE_MINIMAL)
    return buffer.getvalue().encode("utf-8")


def largest_csv_slice_that_fits(df, start_row, capacity_bytes, include_header):
    remaining = len(df) - start_row
    if remaining <= 0:
        return 0, b""

    one_row_bytes = csv_bytes_for_frame(df.iloc[start_row : start_row + 1], include_header)
    if len(one_row_bytes) > capacity_bytes:
        return 1, one_row_bytes

    low = 1
    high = remaining
    best_count = 1
    best_bytes = one_row_bytes

    while low <= high:
        mid = (low + high) // 2
        candidate_bytes = csv_bytes_for_frame(df.iloc[start_row : start_row + mid], include_header)
        if len(candidate_bytes) <= capacity_bytes:
            best_count = mid
            best_bytes = candidate_bytes
            low = mid + 1
        else:
            high = mid - 1

    return best_count, best_bytes


def get_resource_snapshot(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    disk = os.statvfs(output_path) if hasattr(os, "statvfs") else None
    if disk:
        free_disk_bytes = disk.f_bavail * disk.f_frsize
    else:
        import shutil

        free_disk_bytes = shutil.disk_usage(output_path).free

    memory_available_bytes = None
    memory_percent_used = None
    if psutil is not None:
        mem = psutil.virtual_memory()
        memory_available_bytes = int(mem.available)
        memory_percent_used = round(float(mem.percent), 1)

    return {
        "free_disk_bytes": int(free_disk_bytes),
        "free_disk_gb": bytes_to_gb(free_disk_bytes),
        "memory_available_bytes": memory_available_bytes,
        "memory_available_gb": None if memory_available_bytes is None else bytes_to_gb(memory_available_bytes),
        "memory_percent_used": memory_percent_used,
    }


def assert_resources_ok(output_dir, min_free_disk_gb, min_free_memory_gb):
    snapshot = get_resource_snapshot(output_dir)
    problems = []
    if snapshot["free_disk_gb"] < float(min_free_disk_gb):
        problems.append(
            f"free disk space is {snapshot['free_disk_gb']} GB, below the configured minimum of {min_free_disk_gb} GB"
        )
    if psutil is not None and snapshot["memory_available_gb"] is not None:
        if snapshot["memory_available_gb"] < float(min_free_memory_gb):
            problems.append(
                f"available memory is {snapshot['memory_available_gb']} GB, below the configured minimum of {min_free_memory_gb} GB"
            )
    if problems:
        raise ResourceLimitError("Processing stopped safely because " + "; ".join(problems) + ".")
    return snapshot


def format_resource_line(snapshot):
    memory = "memory unavailable"
    if snapshot.get("memory_available_gb") is not None:
        memory = f"available memory: {snapshot['memory_available_gb']} GB ({snapshot['memory_percent_used']}% used)"
    return f"free disk: {snapshot['free_disk_gb']} GB; {memory}"


def selected_rules_from_df(rules_df):
    cleaned = rules_df.copy()
    cleaned["action"] = cleaned["action"].where(cleaned["action"].isin(ACTION_OPTIONS), ACTION_NONE)
    cleaned["id_length"] = pd.to_numeric(cleaned["id_length"], errors="coerce").fillna(10).clip(1, 64).astype(int)
    return cleaned.to_dict("records")


def source_signature(uploaded_files, local_text):
    upload_bits = [f"{item.name}:{item.size}" for item in uploaded_files or []]
    return hashlib.sha256(("|".join(upload_bits) + "||" + (local_text or "")).encode("utf-8")).hexdigest()


def write_chunked_csv(
    source,
    output_dir,
    rules,
    salt,
    max_chunk_mb,
    rows_per_read,
    progress_text,
    resource_check_every,
    status_update_every,
    min_free_disk_gb,
    min_free_memory_gb,
):
    source_stem = safe_table_stem(source.source_name)
    max_bytes = int(max_chunk_mb * 1024 * 1024)
    part_number = 1
    rows_written = 0
    batches_read = 0
    output_files = []
    output_path, handle = open_new_output_file(output_dir, source_stem, part_number)

    partial_result = {
        "source": source.to_dict(),
        "rows_written": 0,
        "batches_read": 0,
        "output_files": output_files,
        "stopped_early": False,
    }

    try:
        snapshot = assert_resources_ok(output_dir, min_free_disk_gb, min_free_memory_gb)
        progress_text.write(f"Starting `{source.display_name}`. {format_resource_line(snapshot)}")

        wrote_header = False
        last_status_time = time.time()
        for chunk_idx, chunk in enumerate(read_csv_chunks(source, rows_per_read), start=1):
            batches_read = chunk_idx

            if chunk_idx == 1 or chunk_idx % max(1, int(resource_check_every)) == 0:
                try:
                    snapshot = assert_resources_ok(output_dir, min_free_disk_gb, min_free_memory_gb)
                except ResourceLimitError as exc:
                    partial_result.update(
                        {
                            "rows_written": rows_written,
                            "batches_read": batches_read,
                            "stopped_early": True,
                        }
                    )
                    if handle and not handle.closed:
                        handle.flush()
                    exc.partial_result = partial_result
                    raise

            if chunk_idx == 1 or chunk_idx % max(1, int(status_update_every)) == 0:
                progress_text.write(
                    f"Processing `{source.display_name}`: batch {chunk_idx:,}; "
                    f"rows written so far: {rows_written:,}; {format_resource_line(snapshot)}"
                )
                last_status_time = time.time()
            elif time.time() - last_status_time > 30:
                progress_text.write(
                    f"Still processing `{source.display_name}`: batch {chunk_idx:,}; rows written so far: {rows_written:,}."
                )
                last_status_time = time.time()

            transformed = apply_rules_to_chunk(chunk, rules, salt)
            row_offset = 0

            while row_offset < len(transformed):
                include_header = not wrote_header
                remaining_bytes = max_bytes - handle.tell()
                if remaining_bytes <= 0 or (not include_header and remaining_bytes < 1024):
                    handle.close()
                    output_files.append(str(output_path))
                    part_number += 1
                    snapshot = assert_resources_ok(output_dir, min_free_disk_gb, min_free_memory_gb)
                    output_path, handle = open_new_output_file(output_dir, source_stem, part_number)
                    wrote_header = False
                    include_header = True
                    remaining_bytes = max_bytes
                    progress_text.write(
                        f"Started output chunk `{output_path.name}` for `{source.display_name}`. {format_resource_line(snapshot)}"
                    )

                rows_to_write, encoded = largest_csv_slice_that_fits(
                    transformed,
                    start_row=row_offset,
                    capacity_bytes=remaining_bytes,
                    include_header=include_header,
                )
                handle.write(encoded)
                wrote_header = True
                row_offset += rows_to_write

            rows_written += len(transformed)
            partial_result.update({"rows_written": rows_written, "batches_read": batches_read})

        if handle and not handle.closed:
            handle.close()
        if output_path.exists() and output_path.stat().st_size > 0:
            output_files.append(str(output_path))
        partial_result.update(
            {
                "rows_written": rows_written,
                "batches_read": batches_read,
                "output_files": output_files,
                "stopped_early": False,
            }
        )
        return partial_result

    except ResourceLimitError:
        try:
            if handle and not handle.closed:
                handle.close()
            if output_path.exists() and output_path.stat().st_size > 0 and str(output_path) not in output_files:
                output_files.append(str(output_path))
        finally:
            partial_result.update(
                {
                    "rows_written": rows_written,
                    "batches_read": batches_read,
                    "output_files": output_files,
                    "stopped_early": True,
                }
            )
        raise
    except Exception:
        try:
            if handle and not handle.closed:
                handle.close()
        except Exception:
            pass
        raise


def render_cleanup_section():
    st.subheader("0. Clean up old temp files")
    with st.expander("Find and remove old work folders", expanded=False):
        st.caption(
            "Use this after stopping prior test runs. It looks in your system temp folder for previous "
            "csv_anonymizer_* folders, and optionally the earlier AI planner temp folder. "
            "The current active work folder is skipped."
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            include_ai_planner = st.checkbox(
                "Also include streamlit_direction2 folders",
                value=True,
                help="This was used by the earlier AI planner workflow and can hold uploaded/extracted/parquet artifacts.",
            )
        with col2:
            min_age_hours = st.number_input(
                "Only show folders older than N hours",
                min_value=0.0,
                max_value=168.0,
                value=0.0,
                step=1.0,
                help="Use 0 to show all matching old folders. Increase this if another run may still be active.",
            )

        if st.button("Scan temp folders", use_container_width=True):
            candidates = discover_cleanup_candidates(include_ai_planner=include_ai_planner, min_age_hours=min_age_hours)
            st.session_state.cleanup_candidates = candidates

        candidates = st.session_state.get("cleanup_candidates", [])
        if candidates:
            total_gb = round(sum(float(item.get("size_gb") or 0) for item in candidates), 2)
            st.info(f"Found {len(candidates)} matching folder(s), totaling about {total_gb} GB.")
            st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
            st.warning("Delete only after Streamlit/Python runs that may be using these folders are stopped.")
            confirm = st.checkbox("I understand these temp folders will be deleted permanently.")
            if st.button("Delete listed temp folders", type="primary", disabled=not confirm, use_container_width=True):
                deleted, failed = cleanup_candidate_dirs(candidates)
                st.session_state.cleanup_candidates = []
                if deleted:
                    st.success(f"Deleted {len(deleted)} temp folder(s).")
                if failed:
                    st.error("Some folders could not be deleted:")
                    st.dataframe(pd.DataFrame(failed), use_container_width=True, hide_index=True)
        else:
            st.caption(f"System temp folder: {tempfile.gettempdir()}")


def render_source_section():
    st.subheader("1. Choose input files")
    uploaded_files = st.file_uploader(
        "Upload CSV or ZIP files",
        type=["csv", "zip"],
        accept_multiple_files=True,
        help="For very large files, local paths are more reliable than browser upload.",
    )
    st.caption("For multi-GB files, prefer the local path box below so Python reads directly from disk.")
    local_source_text = st.text_area(
        "Or enter local CSV/ZIP file paths or folders",
        placeholder='C:\\Users\\oakhtar\\Downloads\\System Access Log(1).zip\n/path/to/large-file.csv\n/path/to/export-folder',
        help="Paste one path per line. Quoted Windows paths are okay; surrounding quotes will be removed.",
    )
    st.caption('Tip: paste one path per line. Paths copied as "C:\\path\\file.zip" are handled automatically.')

    if st.button("Inspect columns", type="primary", use_container_width=True):
        current_signature = source_signature(uploaded_files, local_source_text)
        if not st.session_state.work_dir:
            st.session_state.work_dir = tempfile.mkdtemp(prefix="csv_anonymizer_")
        work_dir = st.session_state.work_dir
        sources = []
        if uploaded_files:
            sources.extend(save_uploaded_sources(uploaded_files, work_dir))
        if local_source_text.strip():
            sources.extend(collect_local_sources(local_source_text))

        unique = {}
        for source in sources:
            key = f"{source.kind}|{source.path}|{source.zip_path}|{source.zip_member}"
            unique[key] = source
        sources = sorted(unique.values(), key=lambda item: item.display_name.lower())

        if not sources:
            st.error("No CSV files were found. Upload CSV/ZIP files or enter local paths.")
            return

        with st.spinner("Reading headers and previews without fully extracting ZIP files..."):
            profile = build_profile(sources)

        st.session_state.source_signature = current_signature
        st.session_state.source_files = [source.to_dict() for source in sources]
        st.session_state.profile = profile
        st.session_state.rules_df = initial_rules_df(profile["all_columns"])
        st.session_state.last_manifest = None
        st.success(f"Found {len(sources)} CSV source(s) with {len(profile['all_columns'])} unique column(s).")


def render_profile_section():
    profile = st.session_state.profile
    if not profile:
        return

    st.subheader("2. Review detected columns")
    file_rows = [
        {
            "file": item["file_name"],
            "uncompressed_size_mb": item["size_mb"],
            "compressed_size_mb": item.get("compressed_size_mb") or "",
            "columns": len(item["columns"]),
        }
        for item in profile["files"]
    ]
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)

    with st.expander("Preview files", expanded=False):
        for item in profile["files"]:
            st.markdown(f"**{item['file_name']}**")
            st.dataframe(item["preview"], use_container_width=True, height=260)


def render_rules_section():
    if st.session_state.rules_df is None:
        return

    st.subheader("3. Choose anonymization rules")
    st.caption("Suggested actions are based on column names. Review them before processing.")
    st.session_state.rules_df = st.data_editor(
        st.session_state.rules_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "column": st.column_config.TextColumn("Column", disabled=True),
            "action": st.column_config.SelectboxColumn("Action", options=ACTION_OPTIONS, required=True),
            "id_length": st.column_config.NumberColumn("ID length", min_value=1, max_value=64, step=1),
            "static_value": st.column_config.TextColumn("Static value"),
            "target_domain": st.column_config.TextColumn("Target domain"),
        },
    )


def render_process_section():
    if st.session_state.rules_df is None or not st.session_state.source_files:
        return

    st.subheader("4. Process and chunk")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        max_chunk_mb = st.number_input(
            "Maximum output chunk size MB",
            min_value=1,
            max_value=MAX_CHUNK_SIZE_MB,
            value=DEFAULT_CHUNK_SIZE_MB,
            step=50,
            help="This is a hard cap per output CSV part. Use less than 2048 MB for safer downstream handling.",
        )
    with col2:
        rows_per_read = st.number_input(
            "Rows processed at a time",
            min_value=1_000,
            max_value=1_000_000,
            value=DEFAULT_READ_ROWS,
            step=10_000,
            help="Larger values may be faster but use more memory. Start with 100,000 for huge files.",
        )
    with col3:
        output_dir = st.text_input(
            "Output folder",
            value=str(Path.cwd() / "anonymized_output"),
            help="The app writes anonymized chunks and a manifest here. Choose a drive with plenty of free space.",
        )

    salt = st.text_input(
        "Anonymization salt",
        value="local-test-salt",
        type="password",
        help="Same salt + same column + same original value creates the same anonymized value.",
    )
    st.caption(
        "Salt controls repeatability: same salt + same column + same original value = same anonymized value. "
        "This preserves joins across files and chunks. Change the salt to generate a different anonymized dataset; "
        "do not use a real password or API key."
    )

    with st.expander("Performance and safety settings", expanded=True):
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            resource_check_every = st.number_input(
                "Check resources every N batches",
                min_value=1,
                max_value=100,
                value=DEFAULT_RESOURCE_CHECK_EVERY,
                step=1,
            )
        with col_b:
            status_update_every = st.number_input(
                "Update status every N batches",
                min_value=1,
                max_value=100,
                value=DEFAULT_STATUS_UPDATE_EVERY,
                step=1,
            )
        with col_c:
            min_free_disk_gb = st.number_input(
                "Stop if free disk below GB",
                min_value=1.0,
                max_value=10_000.0,
                value=DEFAULT_MIN_FREE_DISK_GB,
                step=1.0,
            )
        with col_d:
            min_free_memory_gb = st.number_input(
                "Stop if memory below GB",
                min_value=0.0,
                max_value=1024.0,
                value=DEFAULT_MIN_FREE_MEMORY_GB,
                step=1.0,
                help="Requires psutil. If psutil is not installed, disk checks still run.",
            )

        try:
            snapshot = get_resource_snapshot(output_dir)
            st.info("Current resources for output folder: " + format_resource_line(snapshot))
        except Exception as exc:
            st.warning(f"Could not check current resources yet: {exc}")
        if psutil is None:
            st.warning("psutil is not installed, so memory checks are disabled. Add `psutil>=5.9,<6` to requirements.txt.")

    if st.button("Create anonymized chunks", type="primary", use_container_width=True):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        rules = selected_rules_from_df(st.session_state.rules_df)
        active_rules = [rule for rule in rules if rule["action"] != ACTION_NONE]
        sources = [SourceDescriptor.from_dict(item) for item in st.session_state.source_files]

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "max_chunk_mb": max_chunk_mb,
            "rows_per_read": rows_per_read,
            "resource_check_every": resource_check_every,
            "status_update_every": status_update_every,
            "min_free_disk_gb": min_free_disk_gb,
            "min_free_memory_gb": min_free_memory_gb,
            "source_files": [source.to_dict() for source in sources],
            "active_rules": active_rules,
            "outputs": [],
            "stopped_early": False,
        }

        started = time.time()
        status = st.status("Processing files...", expanded=True)
        try:
            assert_resources_ok(output_dir, min_free_disk_gb, min_free_memory_gb)
            for source in sources:
                result = write_chunked_csv(
                    source=source,
                    output_dir=output_dir,
                    rules=rules,
                    salt=salt,
                    max_chunk_mb=max_chunk_mb,
                    rows_per_read=rows_per_read,
                    progress_text=status,
                    resource_check_every=resource_check_every,
                    status_update_every=status_update_every,
                    min_free_disk_gb=min_free_disk_gb,
                    min_free_memory_gb=min_free_memory_gb,
                )
                manifest["outputs"].append(result)

            manifest["elapsed_seconds"] = round(time.time() - started, 2)
            manifest_path = Path(output_dir) / "anonymization_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
            manifest["manifest_path"] = str(manifest_path)
            st.session_state.last_manifest = manifest
            status.update(label="Processing complete", state="complete", expanded=False)
            st.success(f"Created anonymized output in: {output_dir}")
        except ResourceLimitError as exc:
            manifest["stopped_early"] = True
            manifest["elapsed_seconds"] = round(time.time() - started, 2)
            if exc.partial_result:
                manifest["outputs"].append(exc.partial_result)
            manifest["stop_reason"] = str(exc)
            manifest_path = Path(output_dir) / "anonymization_manifest_partial.json"
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
            manifest["manifest_path"] = str(manifest_path)
            st.session_state.last_manifest = manifest
            status.update(label="Processing stopped safely", state="error", expanded=True)
            st.warning(str(exc))
            st.info(f"Completed output chunks were preserved. Partial manifest written to: {manifest_path}")
        except Exception as exc:
            manifest["stopped_early"] = True
            manifest["elapsed_seconds"] = round(time.time() - started, 2)
            manifest["stop_reason"] = str(exc)
            manifest_path = Path(output_dir) / "anonymization_manifest_failed.json"
            try:
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, indent=2)
                manifest["manifest_path"] = str(manifest_path)
                st.session_state.last_manifest = manifest
            except Exception:
                pass
            status.update(label="Processing failed", state="error", expanded=True)
            st.error(str(exc))


def render_manifest_section():
    manifest = st.session_state.last_manifest
    if not manifest:
        return

    st.subheader("5. Output summary")
    output_rows = []
    for item in manifest.get("outputs", []):
        source_info = item.get("source", {})
        for path in item.get("output_files", []):
            p = Path(path)
            output_rows.append(
                {
                    "source": source_info.get("display_name", source_info.get("source_name", "")),
                    "output_file": path,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2) if p.exists() else "missing",
                    "rows_from_source": item.get("rows_written", 0),
                    "batches_read": item.get("batches_read", 0),
                    "stopped_early": item.get("stopped_early", False),
                }
            )
    if output_rows:
        st.dataframe(pd.DataFrame(output_rows), use_container_width=True, hide_index=True)
    if manifest.get("stopped_early"):
        st.warning(f"Run stopped early: {manifest.get('stop_reason', 'unknown reason')}")
    st.info(f"Manifest written to: {manifest.get('manifest_path')}")


def render_notes():
    with st.expander("How to use this app", expanded=True):
        st.markdown(
            """
            Use this local tool before loading data into the AI Data Planner. Its job is to turn raw CSV exports into anonymized CSV chunks that are safer and easier for the planner to handle.

            **Basic workflow**

            1. Use **Clean up old temp files** if prior tests left large temporary folders behind.
            2. Choose one or more CSV or ZIP files. For very large files, paste local file or folder paths instead of using browser upload.
            3. Click **Inspect columns**. The app reads file headers and previews a small sample without fully loading the dataset into memory.
            4. Review the suggested anonymization rules. The suggestions are based on column names, so they should always be checked before processing.
            5. Set the output folder, chunk size, rows processed at a time, anonymization salt, and resource-safety settings.
            6. Click **Create anonymized chunks**. The app writes anonymized CSV parts plus an `anonymization_manifest.json` report.

            **CSV and ZIP inputs**

            - ZIP files are streamed directly; CSV files inside them are not fully extracted before processing.
            - If a ZIP contains multiple CSVs, each CSV source is detected and processed.
            - For large ZIPs, make sure your computer has enough free disk space for output chunks and temporary working files.
            - Local paths are best for multi-GB files because the app can read directly from disk instead of moving the file through the browser.
            - Paste one local path per line. Windows paths copied with quotation marks are okay.

            **How anonymization works**

            - The same original value in the same column becomes the same anonymized value when the salt is unchanged.
            - This preserves joins across files and chunks. For example, the same `UserId` can still match across users, enrollments, and activity tables after anonymization.
            - Name misspellings, fake IDs, fake emails, URL/domain replacements, redactions, static replacements, and hash tokens are applied only to the columns you select.
            - The manifest records which rules were used, but it does not store original-to-anonymized mappings.

            **How to treat the salt**

            - The salt is a private phrase used in the anonymization recipe.
            - Use the same salt when processing files that need to relate to each other.
            - Change the salt when you want to create a different anonymized version of the same source data.
            - Do not use a real password, API key, or other credential as the salt.
            - Keep the salt reasonably private. Someone with the source data, the salt, and the same rules could reproduce the same anonymized values.

            **Performance and safety**

            - Browser upload is still limited by Streamlit settings and local machine memory. For very large files, use local paths.
            - Resource-aware processing checks free disk space and, when `psutil` is installed, available memory between batches.
            - If the app stops early, it is trying to avoid exhausting the machine. Check the run time, stop reason, output folder, free disk space, and any partial output files.
            - Smaller **Rows processed at a time** values use less memory. Larger values may run faster but can increase memory pressure.

            **After processing**

            - Load the anonymized output chunks into the AI Data Planner, not the original raw files.
            - Keep the original raw files, salts, and temporary folders out of any shared workspace.
            - Use the manifest to confirm which files were processed, which rules were applied, how long the run took, and whether the run stopped early.
            """
        )


def main():
    ensure_state()
    st.title(APP_TITLE)
    st.caption("Prepare sanitized CSV chunks for the AI Data Planner")
    render_notes()
    render_cleanup_section()
    render_source_section()
    render_profile_section()
    render_rules_section()
    render_process_section()
    render_manifest_section()


if __name__ == "__main__":
    main()
