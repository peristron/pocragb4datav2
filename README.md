# direction 2: ai data planner

this repository contains a streamlit-based prototype for chatting with large structured datasets.

the main hosted app is:

```text
ai_planner.py

the goal of this project is to test a stronger ai-first workflow for large dataset analysis, where the app does more than generate sql. it profiles uploaded data, plans an answer strategy, generates sql, executes it with duckdb, critiques the result, and then summarizes the answer.

what this app is for

use this app to explore whether structured ai planning can improve reliability when asking analytical questions over large csv/zip datasets.

the app is designed for:

structured csv or zip datasets
sanitized, anonymized, or dummy data
multi-table analytical questions
testing profile-first and plan-first ai workflows
comparing hosted streamlit usage with local execution
important privacy note

the hosted streamlit community cloud version should only be used with sanitized, anonymized, or dummy datasets.

raw client, learner, user, or institution-identifiable data should be prepared locally before upload.

for that reason, this repository also includes local-only tools under:

local_tools/

these tools are intended for download and local execution. they are not the hosted streamlit app entrypoint.

main hosted workflow

the hosted app supports this flow:

sanitized csv/zip files
-> upload to ai planner
-> profile dataset
-> ask analytical questions
-> ai plans answer
-> ai generates sql
-> duckdb executes query
-> ai critiques result
-> ai summarizes answer

compared with a simple text-to-sql prototype, this version adds:

a dataset profile layer
table-kind and overlap reasoning
explicit answer planning before sql
result critique after execution
optional pii redaction before summarization
support for local run packaging
repository structure
pocragb4datav2/
├── .streamlit/
│   └── config.toml
├── ai_planner.py
├── requirements.txt
├── README.md
└── local_tools/
    ├── README.md
    ├── privacy_first_anonymizer/
    │   ├── csv_anonymizer_chunker.py
    │   ├── requirements.local.txt
    │   ├── run_anonymizer.bat
    │   ├── run_anonymizer.sh
    │   └── README.md
    └── robust_etl_artifact_workflow/
        ├── preprocessv11.py
        ├── postprocessv8.py
        ├── requirements.local.txt
        ├── run_preprocess.bat
        ├── run_preprocess.sh
        ├── run_postprocess.bat
        ├── run_postprocess.sh
        └── README.md
two supported workflow options

this repository supports two related but distinct workflows.

option 1: privacy-first anonymizer + ai planner

this is the recommended workflow when data needs to be sanitized before upload.

raw csv/zip files
-> local csv anonymizer/chunker
-> anonymized csv chunks
-> hosted ai planner or local ai planner

use this option when:

raw data should remain local
data needs to be anonymized or pseudonymized before upload
users want to prepare streamlit-friendly csv chunks
the hosted streamlit community cloud app is acceptable after anonymization
the workflow should be easy for non-infrastructure users to follow

local tool folder:

local_tools/privacy_first_anonymizer/

main script:

csv_anonymizer_chunker.py

the anonymizer supports:

csv and zip inputs
local file paths for large files
streamed zip processing
column detection
user-selected anonymization rules
randomized name misspellings
random numeric ids
random alphanumeric ids
fake emails
brightspace url/domain replacement
static replacement
redaction
stable hash tokens
chunked csv outputs
disk and memory safety checks
cleanup of old temp folders
output manifest generation
option 2: robust local etl + artifact-based chat

this is the more infrastructure-oriented workflow.

raw csv/zip files
-> local preprocessor
-> parquet artifacts + metadata
-> postprocessor chat app

use this option when:

datasets are very large
durable parquet artifacts are preferred
cloud artifact storage may be needed
preprocessing and chat should be separated
operational or repeatable data workflows are required

local tool folder:

local_tools/robust_etl_artifact_workflow/

main scripts:

preprocessv11.py
postprocessv8.py

this workflow is useful for larger-scale artifact preparation and local/cloud-backed analysis.

which workflow should i use?
need	recommended workflow
i want to use the hosted streamlit app safely	privacy-first anonymizer, then hosted ai planner
i have raw identifiable data	anonymize locally first
i have very large files	use local path input in the anonymizer or robust etl workflow
i need parquet artifacts	robust etl artifact workflow
i need cloud artifact storage	robust etl artifact workflow
i want the simplest user-facing flow	privacy-first anonymizer + ai planner
i want stronger operational separation	robust etl + postprocessor
running the hosted app locally

install dependencies:

pip install -r requirements.txt

run the main planner:

streamlit run ai_planner.py

for larger local uploads, use:

streamlit run ai_planner.py --server.maxUploadSize 10000
using the local tools

see:

local_tools/README.md

for local anonymization:

local_tools/privacy_first_anonymizer/

for the robust pre/post workflow:

local_tools/robust_etl_artifact_workflow/

local-only tools have their own requirements.local.txt files so that the hosted streamlit deployment is not affected by local-only dependencies.

streamlit community cloud deployment

the streamlit community cloud deployment should continue to point to:

ai_planner.py

adding files under local_tools/ should not affect the hosted app as long as:

the configured app entrypoint remains ai_planner.py
root requirements.txt only includes dependencies needed by the hosted app
local-only dependencies stay in requirements.local.txt
local-only scripts are not renamed to become the app entrypoint
security and privacy guidance
use the hosted app only with sanitized, anonymized, or dummy data
keep raw exports local unless approved for upload
use the anonymizer before hosted upload when data may contain identifiable values
review anonymized outputs before sharing
do not hardcode api keys or secrets
use streamlit secrets or local environment variables for credentials
remember that pseudonymization helps reduce exposure, but is not the same as a formal guarantee of irreversible de-identification
salt guidance for anonymization

the local anonymizer uses a salt to make replacements repeatable.

the same salt + same column + same original value produces the same anonymized value.

this helps preserve joins across related files and chunks.

use the same salt when related files need to remain joinable.

change the salt when you want to generate a different anonymized version.

do not use a real password, api key, or credential as the salt.

status

this is a prototype / proof of concept.

it is intended to test large-dataset ai planning, privacy-first preparation workflows, and practical local/hosted streamlit patterns.
```
