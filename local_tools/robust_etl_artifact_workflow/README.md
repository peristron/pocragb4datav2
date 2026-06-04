# robust etl artifact workflow

this folder contains the local preprocessing and postprocessing workflow for large-dataset rag/chat analysis.

this workflow is more infrastructure-oriented than the privacy-first anonymizer. it is intended for larger or more operational use cases where data is prepared into durable artifacts before chat-based analysis.

## purpose

use this workflow when the goal is to:

- process very large csv/zip exports locally
- generate parquet artifacts
- generate metadata and relationship hints
- support artifact-based or cloud-backed analysis
- separate preprocessing from the chat interface
- work with datasets that may be too large for browser upload

## files

```text
preprocessv11.py
postprocessv8.py
requirements.local.txt
run_preprocess.bat
run_preprocess.sh
run_postprocess.bat
run_postprocess.sh
workflow
raw csv/zip files
-> preprocessv11.py
-> parquet chunks + metadata/artifacts
-> postprocessv8.py
-> chat/query interface over prepared data
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
run preprocessor

windows:

.\run_preprocess.bat

mac/linux:

chmod +x run_preprocess.sh
./run_preprocess.sh

or run directly:

streamlit run preprocessv11.py --server.maxUploadSize 10000
run postprocessor

windows:

.\run_postprocess.bat

mac/linux:

chmod +x run_postprocess.sh
./run_postprocess.sh

or run directly:

streamlit run postprocessv8.py --server.maxUploadSize 10000
when to use this workflow

use this option when:

the dataset is very large
parquet artifacts are preferred
the data needs a repeatable preprocessing step
cloud artifact storage may be used
the user needs a more robust local or operational workflow
privacy and security notes

raw data should be handled locally during preprocessing.

prepared artifacts may still contain sensitive data unless anonymization or redaction has been performed.

llm usage should be reviewed carefully. in many chat workflows, the llm receives schema/context for sql generation and may receive sampled query results for summarization.

do not hardcode credentials or api keys in the scripts. use local environment variables, streamlit secrets, or another approved secret-management method.


---

# `local_tools/robust_etl_artifact_workflow/requirements.local.txt`

this may need adjustment based on the actual imports in `preprocessv11.py` and `postprocessv8.py`, but this is a reasonable starting point:

```txt
streamlit>=1.45,<2
duckdb>=1.1,<2
pandas>=2.2,<3
pyarrow>=16,<22
openai>=1.30,<2
chromadb>=0.5,<1
boto3>=1.34,<2
google-cloud-storage>=2.16,<3
psutil>=5.9,<6
plotly>=5.22,<6

if one of those scripts does not actually use chromadb, boto3, google-cloud-storage, or plotly, remove the unused packages to keep setup lighter.
