# Code Review Summarizer

A developer-focused code-review assistant that analyzes pull requests and repositories, combining static analysis, historical review patterns, and an LLM-based reviewer (NVIDIA Integrate / OpenAI-compatible) to produce structured findings, risk scores, and review guidance.

## Highlights
- Differentiates cosmetic vs substantive changes
- Runs static analysis (pylint, bandit, eslint) and surfaces results
- Uses historical commit/review patterns to prioritize findings
- Produces structured JSON output (summary, risk_assessment, security_findings, review_focus, compliance_check)
- Streamlit chat UI: accepts local repo paths, GitHub PR URLs, and git repo URLs; streams model output

## Quick start (Streamlit)

1. Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with the required environment variables (example below).

3. Run the Streamlit UI:

```powershell
# from project root
streamlit run .\streamlit_app.py
```

Open http://localhost:8501 in your browser. Use the chat input to paste a local repo path, a git repo URL (https or ssh), or a GitHub PR URL (https://github.com/owner/repo/pull/123).

## Environment variables
Create a `.env` file in the project root and set the following keys as needed:

- NVIDIA_API_KEY - required for NVIDIA integrate calls (also set x-api-key header)
- NVIDIA_API_ENDPOINT - e.g. https://integrate.api.nvidia.com/v1 (or your host)
- NVIDIA_API_MODEL - model id to call (e.g., gpt-4o-mini or an internal model)
- GITHUB_TOKEN - optional, improves GitHub API access and allows access to private repos (recommended)
- SONAR_HOST_URL - optional, if you want SonarQube integration
- SONAR_TOKEN - optional, SonarQube token

Example `.env` (do not commit this file):

```
NVIDIA_API_KEY=sk-...
NVIDIA_API_ENDPOINT=https://integrate.api.nvidia.com/v1
NVIDIA_API_MODEL=gpt-4o-mini
GITHUB_TOKEN=ghp_...
SONAR_HOST_URL=https://sonarqube.example.com
SONAR_TOKEN=...
```

## Command-line / Module usage
The project includes `main.py` which can be invoked directly for scripting analysis. Example:

```powershell
python main.py C:\path\to\repo
```

Note: the Streamlit UI is the primary integration and provides the easiest interactive workflow.

## Troubleshooting

- Shallow clone / missing commit errors: If you analyze a GitHub PR or remote repo URL, the app performs a shallow clone by default and then attempts to deepen/unshallow the clone so diff operations can find parent commits. If you still see errors like `git diff-tree ... exit code 128`, try one of:
  - Ensure the machine has network access to GitHub and sufficient permissions.
  - Provide `GITHUB_TOKEN` in `.env` so the app can fetch PR metadata and authenticated refs.
  - Manually clone the repo locally (no `--depth`) and pass the local path to the UI.

- Private repos: set `GITHUB_TOKEN` (recommended). Avoid embedding tokens in clone URLs in logs.

- Model response is not valid JSON: The app instructs the model to return a JSON object. If the response is free text, the UI will show the streamed text and attempt to parse the accumulated content. Consider increasing model strictness or adding a retry prompt.

## Developer notes & suggested improvements

I reviewed the codebase and recommend these small to medium improvements (I can implement them if you'd like):

1. run_static_analysis (in `main.py`) - bug risk: using lambda inside a loop captures the loop variable. Replace the lambda with a function or pass arguments to avoid closure capturing the final loop value.
2. Better GitHub-PR fetch: instead of unshallowing an entire repo, fetch the PR base SHA via GitHub API and `git fetch origin <sha>` — faster and avoids heavy fetches.
3. Response schema enforcement: add a retry loop that re-prompts the model if the returned content isn't valid JSON or doesn't match required keys.
4. SonarQube integration is a placeholder; use environment variables directly (not shell-style ${VAR}) and handle network/auth errors gracefully.
5. Add unit tests for the diff parsing and the FileChange dataclass behaviours (edge cases: first commit, large diffs, binary files).

## Requirements

- Python 3.8+
- Git
- Node.js (optional, for ESLint)
- Pylint, Bandit, ESLint (installable via pip/npm as needed)

## How I validated changes
- Ensured `streamlit_app.py` compiles (python -m py_compile)
- Created multi-page Streamlit views for guidelines and checklist

## Next steps I can implement for you (pick any)

1. Implement targeted PR-base fetch via GitHub API to avoid `--unshallow`.
2. Fix the closure issue in `run_static_analysis` and add small unit tests.
3. Add stricter JSON schema validator and automatic re-prompting for model responses.
4. Add authenticated clone support for private repos using `GITHUB_TOKEN` with a secure approach.

If you'd like, tell me which next step to take and I'll implement it and run the validation locally.