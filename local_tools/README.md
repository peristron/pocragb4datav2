# local tools

this folder contains local-only utilities that support large-dataset ai analysis workflows.

these files are provided for download and local execution. they are not the entrypoint for the hosted streamlit community cloud app.

## available local workflows

### 1. privacy-first anonymizer

folder:

```text
local_tools/privacy_first_anonymizer/

use this workflow when raw csv/zip exports need to be anonymized locally before being uploaded to the hosted ai planner or analyzed locally.

main script:

csv_anonymizer_chunker.py

this tool:

runs locally in streamlit
accepts csv and zip inputs
supports local file paths for very large files
streams csv files from zip archives where possible
lets the user choose anonymization rules by column
writes anonymized csv chunks
includes disk and memory safety checks
produces a manifest describing the output files

recommended for:

privacy-first testing
preparing sanitized csv chunks
streamlit community cloud demos using anonymized data
users who want to keep raw source files local
2. robust etl artifact workflow

folder:

local_tools/robust_etl_artifact_workflow/

use this workflow when very large datasets need to be processed into durable artifacts for local or cloud-backed analysis.

main scripts:

preprocessv11.py
postprocessv8.py

this workflow:

preprocesses large csv/zip exports locally
converts data into parquet artifacts
generates metadata and relationship hints
can support artifact-style deployment workflows
provides a postprocessing/chat interface over prepared artifacts

recommended for:

larger operational workflows
durable parquet artifact generation
local or cloud-backed artifact storage
cases where datasets are too large for browser upload
teams that need a stronger preprocessing/postprocessing separation
important note

the hosted streamlit community cloud deployment should continue to use the root-level app entrypoint:

ai_planner.py

the files in local_tools/ are not intended to be launched automatically by streamlit community cloud.


---

# `local_tools/privacy_first_anonymizer/README.md`

```markdown
# privacy-first csv anonymizer and chunker

this local streamlit app prepares anonymized csv chunks for use with the ai data planner.

it is intended to run on a user's local machine before any hosted upload. raw source files should remain local.

## purpose

use this tool to:

- inspect csv files or csv files inside zip archives
- choose which columns to anonymize
- apply repeatable anonymization rules
- replace sensitive identifiers before upload
- split large outputs into smaller csv chunks
- create a manifest describing the processed output

after processing, users can either:

1. upload the anonymized csv chunks to the hosted ai data planner, or
2. run the ai data planner locally.

## files

```text
csv_anonymizer_chunker.py
requirements.local.txt
run_anonymizer.bat
run_anonymizer.sh
install

from this folder, create and activate a local python environment.

windows powershell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.local.txt

mac/linux:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.local.txt
run

windows:

.\run_anonymizer.bat

mac/linux:

chmod +x run_anonymizer.sh
./run_anonymizer.sh

or run directly:

streamlit run csv_anonymizer_chunker.py --server.maxUploadSize 10000
recommended workflow
launch the anonymizer locally.
use the local path input for large csv or zip files.
inspect detected columns.
choose anonymization rules for sensitive columns.
set output chunk size.
choose a salt.
run processing.
review the output manifest.
upload only anonymized csv chunks to the hosted planner, or use them with the local planner.
salt guidance

the app uses a salt to make anonymized replacements repeatable.

the same salt + same column + same original value produces the same anonymized value.

this is important because it preserves joins across chunks and related files.

example:

users.csv:        UserId 12345 -> 7392048156
enrollments.csv:  UserId 12345 -> 7392048156
activity.csv:     UserId 12345 -> 7392048156

use the same salt when related files need to remain joinable.

change the salt when you want to generate a different anonymized version of the same data.

do not use a real password, api key, or credential as the salt.

keep the salt reasonably private.

privacy note

this tool supports pseudonymization/anonymization workflows, but it should not be described as a formal guarantee of irreversible de-identification.

review outputs before sharing or uploading them.


---

# `local_tools/privacy_first_anonymizer/requirements.local.txt`

```txt
streamlit>=1.45,<2
pandas>=2.2,<3
psutil>=5.9,<6
