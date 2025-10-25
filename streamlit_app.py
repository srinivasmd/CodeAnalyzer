import streamlit as st
import time
import json
import requests
from typing import Generator
import tempfile
import shutil
import subprocess
import os
import re
from urllib.parse import urlparse

# Reuse helper functions from main.py
from main import get_pr_diff, run_static_analysis, get_historical_patterns
from main import NVIDIA_API_ENDPOINT, NVIDIA_API_KEY, NVIDIA_API_MODEL
from metrics import metrics_tracker

st.set_page_config(page_title="Code Review Summarizer Chat", layout="wide")

st.title("Code Review Summarizer — Chat Interface")
st.write("Enter a repository path (absolute) or a Git/PR URL and I'll analyze the latest commit(s) and stream a review from the LLM.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []  # list of dicts: {role: 'user'|'assistant', 'content': text}

# On startup, detect whether guideline/checklist files exist so sidebar can show status immediately
try:
    base_dir = os.path.dirname(__file__) or os.getcwd()
    _guidelines_path = os.path.join(base_dir, 'pr_review_guidelines.json')
    _checklist_path = os.path.join(base_dir, 'security-checklist.json')
    st.session_state.setdefault('guidelines_found', os.path.exists(_guidelines_path))
    st.session_state.setdefault('checklist_found', os.path.exists(_checklist_path))
    st.session_state.setdefault('guidelines_load_error', None)
except Exception:
    # If session state isn't available or os errors occur, ignore; sidebar will show 'not checked yet'
    pass


def stream_nvidia_chat(messages: list) -> Generator[str, None, None]:
    """Call NVIDIA OpenAI-compatible chat/completions endpoint in streaming mode and yield chunks.

    Best-effort: tries to stream lines from the HTTP response. If streaming isn't supported,
    will yield the full response as one chunk.
    """
    base_endpoint = (NVIDIA_API_ENDPOINT or "https://integrate.api.nvidia.com/v1").rstrip('/')
    invoke_url = f"{base_endpoint}/chat/completions"

    headers = {
        'Authorization': f'Bearer {NVIDIA_API_KEY}',
        'x-api-key': NVIDIA_API_KEY,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    payload = {
        'model': NVIDIA_API_MODEL,
        'messages': messages,
        'temperature': 0.3,
        'top_p': 0.9,
        'max_tokens': 2000,
        'stream': True
    }

    try:
        with requests.post(invoke_url, headers=headers, json=payload, stream=True, timeout=120) as resp:
            # If the server sends chunked lines (SSE-like or JSON-lines), iterate them
            if resp.status_code != 200:
                # Not successful — raise to be handled by caller
                resp.raise_for_status()

            # Try streaming by lines
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                # Many streaming endpoints prefix SSE data with 'data: '
                if line.startswith('data:'):
                    line = line[len('data:'):].strip()
                # Some lines may be keep-alives like '[DONE]'
                if line in ('[DONE]', ''):
                    continue
                # Try to decode JSON in the line
                try:
                    parsed = json.loads(line)
                    # Try to extract text from common shapes
                    if isinstance(parsed, dict):
                        # choices -> first -> delta/content or message/content
                        if 'choices' in parsed and isinstance(parsed['choices'], list) and parsed['choices']:
                            first = parsed['choices'][0]
                            # delta content (streaming token)
                            if isinstance(first, dict):
                                delta = first.get('delta') or {}
                                text = delta.get('content') or delta.get('text')
                                if text:
                                    yield text
                                    continue
                                # message style
                                msg = first.get('message', {})
                                content = msg.get('content') or first.get('text')
                                if content:
                                    yield content
                                    continue
                        # fallback: if parsed has 'output' or 'result'
                        for key in ('output', 'result', 'response'):
                            if key in parsed:
                                val = parsed[key]
                                if isinstance(val, str):
                                    yield val
                                else:
                                    yield json.dumps(val)
                                break
                    else:
                        yield str(parsed)
                except Exception:
                    # Not JSON: yield raw line
                    yield line
            # End of streaming — done
    except Exception as e:
        # If streaming fails for any reason, try a single non-streaming call as fallback
        try:
            payload.pop('stream', None)
            r2 = requests.post(invoke_url, headers=headers, json=payload, timeout=120)
            r2.raise_for_status()
            res = r2.json()
            # Try to extract a single content
            if isinstance(res, dict) and 'choices' in res and isinstance(res['choices'], list) and res['choices']:
                first = res['choices'][0]
                content = ''
                if isinstance(first, dict):
                    content = first.get('message', {}).get('content') or first.get('text') or ''
                else:
                    content = str(first)
                if content:
                    yield content
                    return
            # If not found, yield the raw JSON
            yield json.dumps(res)
        except Exception as e2:
            yield f"Error calling NVIDIA API: {e} / {e2}"


def handle_repo_input(repo_path: str):
    # Basic validation
    if not repo_path:
        return "Please provide a repository path."

    try:
        # Detect if input is a PR URL (GitHub), a git repo URL, or a local repo path
        pr_match = None
        repo_dir = None
        pr_review_comments = []
        pr_review_reviews = []
        pr_comments_error = None

        repo_url_match = None
        if repo_path.startswith('http') or repo_path.startswith('git@'):
            # parse URL-like inputs
            parsed = urlparse(repo_path if repo_path.startswith('http') else repo_path.replace(':', '/', 1).replace('git@', 'ssh://git@'))
            # GitHub PR URL detection: /owner/repo/pull/<n>
            if 'github.com' in parsed.netloc:
                m = re.search(r"/([^/]+/[^/]+)/pull/(\d+)", parsed.path)
                if m:
                    repo_full = m.group(1)  # owner/repo
                    pr_number = m.group(2)
                    pr_match = (repo_full, pr_number)
            # If not a PR, detect repo URL patterns like https://github.com/owner/repo(.git)?
            if not pr_match and 'github.com' in parsed.netloc:
                m2 = re.search(r"/([^/]+/[^/]+)(?:\.git)?/?$", parsed.path)
                if m2:
                    repo_url_match = m2.group(1)

        if pr_match:
            # Clone the repo shallowly and fetch the PR head
            owner_repo, pr_number = pr_match
            tmpdir = tempfile.mkdtemp(prefix='crs_pr_')
            repo_https = f"https://github.com/{owner_repo}.git"
            try:
                # shallow clone
                subprocess.run(["git", "clone", "--depth", "1", repo_https, tmpdir], check=True, capture_output=True)
                # fetch PR head into local branch pr-N
                subprocess.run(["git", "fetch", "origin", f"pull/{pr_number}/head:pr-{pr_number}"], cwd=tmpdir, check=True, capture_output=True)
                subprocess.run(["git", "checkout", f"pr-{pr_number}"], cwd=tmpdir, check=True, capture_output=True)
                repo_dir = tmpdir
                # Ensure we have enough history for diff operations. If shallow, try to deepen or unshallow so parent commits exist.
                try:
                    rev_parse = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=tmpdir, capture_output=True, text=True)
                    if rev_parse.returncode == 0 and rev_parse.stdout.strip() == 'true':
                        # First try to deepen a bit to include parents
                        subprocess.run(["git", "fetch", "--deepen", "50"], cwd=tmpdir, check=False, capture_output=True)
                        # As a more robust fallback, try to unshallow (may be slower)
                        subprocess.run(["git", "fetch", "--unshallow"], cwd=tmpdir, check=False, capture_output=True)
                except Exception:
                    # Non-fatal; operations below will try to work with what we have
                    pass
            except subprocess.CalledProcessError as e:
                # cleanup and raise
                shutil.rmtree(tmpdir, ignore_errors=True)
                raise RuntimeError(f"Failed to fetch PR: {e.stderr.decode(errors='ignore')}")
            # Try to fetch historical PR review comments and review events from GitHub API (public or using GITHUB_TOKEN)
            try:
                gh_token = os.environ.get('GITHUB_TOKEN')
                gh_headers = {'Accept': 'application/vnd.github.v3+json'}
                if gh_token:
                    gh_headers['Authorization'] = f'token {gh_token}'

                gh_api_base = 'https://api.github.com'
                comments_url = f"{gh_api_base}/repos/{owner_repo}/issues/{pr_number}/comments"
                reviews_url = f"{gh_api_base}/repos/{owner_repo}/pulls/{pr_number}/reviews"
                # fetch issue comments (review discussion comments)
                try:
                    r_c = requests.get(comments_url, headers=gh_headers, timeout=20)
                    if r_c.ok:
                        pr_review_comments = r_c.json()
                except Exception:
                    # non-fatal
                    pr_comments_error = 'Failed to fetch PR issue comments'

                # fetch review events (approvals, review body)
                try:
                    r_r = requests.get(reviews_url, headers=gh_headers, timeout=20)
                    if r_r.ok:
                        pr_review_reviews = r_r.json()
                except Exception:
                    pr_comments_error = (pr_comments_error or '') + '; failed to fetch PR reviews'
            except Exception as e:
                pr_comments_error = str(e)
        elif repo_url_match:
            # Clone a repo URL (owner/repo) shallowly into tmpdir
            owner_repo = repo_url_match
            tmpdir = tempfile.mkdtemp(prefix='crs_repo_')
            # build https clone URL
            repo_https = f"https://github.com/{owner_repo}.git"
            try:
                subprocess.run(["git", "clone", "--depth", "1", repo_https, tmpdir], check=True, capture_output=True)
                repo_dir = tmpdir
                # If the clone is shallow, deepen/unshallow so diffs between commits work
                try:
                    rev_parse = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=tmpdir, capture_output=True, text=True)
                    if rev_parse.returncode == 0 and rev_parse.stdout.strip() == 'true':
                        subprocess.run(["git", "fetch", "--deepen", "50"], cwd=tmpdir, check=False, capture_output=True)
                        subprocess.run(["git", "fetch", "--unshallow"], cwd=tmpdir, check=False, capture_output=True)
                except Exception:
                    pass
            except subprocess.CalledProcessError as e:
                shutil.rmtree(tmpdir, ignore_errors=True)
                raise RuntimeError(f"Failed to clone repo URL: {e.stderr.decode(errors='ignore')}")
        else:
            # treat as local repo path
            repo_dir = repo_path

        # Prepare analysis context
        changes = get_pr_diff(repo_dir, repo_path if pr_match else "")
        file_paths = [c.filename for c in changes]
        static_results = run_static_analysis(file_paths)
        historical = get_historical_patterns(repo_dir)

        # Load optional guideline and security checklist JSON files from the project root
        guidelines = {}
        checklist = {}
        guidelines_error = None
        try:
            base_dir = os.path.dirname(__file__) or os.getcwd()
            guidelines_path = os.path.join(base_dir, 'pr_review_guidelines.json')
            checklist_path = os.path.join(base_dir, 'security-checklist.json')
            if os.path.exists(guidelines_path):
                with open(guidelines_path, 'r', encoding='utf-8') as gf:
                    guidelines = json.load(gf)
            if os.path.exists(checklist_path):
                with open(checklist_path, 'r', encoding='utf-8') as sf:
                    checklist = json.load(sf)
        except Exception as e:
            guidelines_error = str(e)

        # Persist guideline/checklist load status into Streamlit session state so the UI can show indicators
        try:
            st.session_state['guidelines_found'] = bool(guidelines)
            st.session_state['checklist_found'] = bool(checklist)
            st.session_state['guidelines_load_error'] = guidelines_error
        except Exception:
            # If Streamlit session_state isn't available for some reason, ignore
            pass

        # System message: instruction and required JSON output schema
        # Be explicit: list which fields from the guidelines/checklist to consult, and require a numeric risk score.
        system_msg = {
            'role': 'system',
            'content': (
                "You are a senior developer reviewing a pull request. Use the provided PR review guidelines and the security checklist to evaluate changes. "
                "Specifically, for each finding reference which guideline or checklist item was applied (include its id/key if available). "
                "Produce the following JSON object only with these keys:\n"
                "- summary: a short human-readable summary of the change and overall recommendation.\n"
                "- risk_assessment: map of issue -> {score: number 0-10, confidence: number 0.0-1.0, rationale: string, applied_rules: [list of guideline/checklist ids]}\n"
                "- security_findings: list of {title, description, file, line, severity (low|medium|high|critical), checklist_items_applied}\n"
                "- review_focus: list of files/areas requiring manual review with reasons\n"
                "- compliance_check: map of checklist item id -> {status: pass|fail|manual-check, notes}\n"
                "Risk scoring rules: use 0 (no risk) to 10 (critical). Provide a confidence score between 0.0 and 1.0 for each assessment. Always cite which guideline/checklist entries influenced the score."
            )
        }

        # If we loaded guidelines/checklist, include them as system-level context so the model prioritizes them
        extra_system_msgs = []
        try:
            if guidelines:
                extra_system_msgs.append({'role': 'system', 'content': 'PR Review Guidelines:\n' + json.dumps(guidelines)})
            if checklist:
                extra_system_msgs.append({'role': 'system', 'content': 'Security Checklist:\n' + json.dumps(checklist)})
        except Exception:
            # If guidelines are not JSON-serializable for some reason, include as plain string
            extra_system_msgs.append({'role': 'system', 'content': f'PR Review Guidelines (raw): {str(guidelines)[:1000]}'})

        user_content = {
            'repo_path': repo_path,
            'changes_overview': {c.filename: c.lines_changed for c in changes},
            'static_results': static_results,
            'historical_patterns': historical,
            # include any fetched PR review comments/reviews to provide historical reviewer context
            'pr_review_comments': pr_review_comments if pr_match else [],
            'pr_review_reviews': pr_review_reviews if pr_match else [],
            'pr_review_comments_error': pr_comments_error if pr_match else None,
            'pr_review_guidelines': guidelines,
            'security_checklist': checklist,
            'guidelines_load_error': guidelines_error
        }
        user_msg = {'role': 'user', 'content': json.dumps(user_content)}

        # Build final messages: base system message, extra system context (guidelines/checklist), then user
        messages = [system_msg] + extra_system_msgs + [user_msg]

        # Stream the LLM response and accumulate
        assistant_text = ''
        for chunk in stream_nvidia_chat(messages):
            assistant_text += chunk
            yield chunk

    except Exception as e:
        yield f"Error preparing repo analysis: {e}"
    finally:
        # cleanup temporary clone if used
        try:
            if pr_match and repo_dir and os.path.exists(repo_dir):
                shutil.rmtree(repo_dir)
        except Exception:
            pass


# --- UI ---

with st.sidebar:
    st.markdown("## Usage\nType or paste an absolute repo path in the chat input below and press Enter.")
    st.markdown("This demo streams the model's text output as it arrives.")
    st.markdown("---")
    st.markdown("### Guidelines & Checklist status")
    # Show status if available in session_state (set when analysis starts)
    if 'guidelines_found' in st.session_state:
        if st.session_state.get('guidelines_found'):
            st.success('PR review guidelines loaded')
        else:
            st.info('PR review guidelines not found')
    else:
        st.info('PR review guidelines not checked yet')

    if 'checklist_found' in st.session_state:
        if st.session_state.get('checklist_found'):
            st.success('Security checklist loaded')
        else:
            st.info('Security checklist not found')
    else:
        st.info('Security checklist not checked yet')

    if st.session_state.get('guidelines_load_error'):
        st.warning(f"Guidelines load error: {st.session_state.get('guidelines_load_error')}")

    st.markdown("---")
    st.markdown("### View Guidelines & Checklist")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("📋 PR Review Guidelines", "http://localhost:8501/view_guidelines", use_container_width=True)
    with col2:
        st.link_button("🔒 Security Checklist", "http://localhost:8501/view_checklist", use_container_width=True)

st.write("---")

# Add metrics reports section
with st.expander("📊 Metrics Reports (click to expand)"):
    st.markdown("### Evaluation Metrics")
    try:
        reports = metrics_tracker.get_aggregated_reports()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", reports.get('total_analyses', 0))
        with col2:
            st.metric("Avg Analysis Time (s)", f"{reports.get('average_analysis_time_seconds', 0):.2f}")
        with col3:
            st.metric("Avg Lines Changed", f"{reports.get('average_lines_changed', 0):.1f}")

        st.markdown("#### False Positive Rate")
        fp_metrics = reports.get('false_positive_metrics', {})
        st.write(f"Rate: {fp_metrics.get('false_positive_rate', 0):.2%} ({fp_metrics.get('total_feedbacks', 0)} feedbacks)")

        st.markdown("#### Review Time Reduction")
        time_red = reports.get('review_time_reduction', {})
        st.write(f"Avg Reduction: {time_red.get('average_reduction_percent', 0):.1f}% over {time_red.get('total_analyses', 0)} analyses")

        # Manual feedback input
        st.markdown("#### Provide Manual Feedback")
        st.markdown("Select an analysis and provide feedback on findings to improve false positive calculations.")
        recent_analyses = metrics_tracker.get_recent_analyses(limit=5)
        if recent_analyses:
            analysis_options = {f"{a['id']}: {a['repo_path']} ({a['timestamp'][:10]})": a['id'] for a in recent_analyses}
            selected_analysis = st.selectbox("Select Analysis", list(analysis_options.keys()))
            if selected_analysis:
                analysis_id = analysis_options[selected_analysis]
                finding_title = st.text_input("Finding Title (from security_findings)")
                actual_outcome = st.selectbox("Actual Outcome", ["true_positive", "false_positive", "missed"])
                user_feedback = st.text_area("Additional Feedback (optional)")
                if st.button("Submit Feedback"):
                    metrics_tracker.log_feedback(analysis_id, finding_title, actual_outcome, user_feedback)
                    st.success("Feedback logged successfully!")
                    st.rerun()  # Refresh to update metrics
        else:
            st.info("No analyses logged yet.")

    except Exception as e:
        st.error(f"Error loading metrics: {e}")

st.write("---")

# Render chat messages
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.chat_message('user').write(msg['content'])
    else:
        # For assistant messages, try to parse as JSON and display structured
        assistant_msg = st.chat_message('assistant')
        msg_area = assistant_msg.container()
        try:
            analysis = json.loads(msg['content'])
            if isinstance(analysis, dict):
                # Raw JSON in a collapsible expander
                with msg_area.expander("🔍 Raw JSON (click to expand)"):
                    st.json(analysis)

                # Human-readable Summary
                if 'summary' in analysis:
                    col1, col2, col3 = msg_area.columns([8, 1, 1])
                    with col1:
                        msg_area.markdown("## Summary")
                    with col2:
                        if msg_area.button("👍", key=f"summary_up_{msg['content'][:10]}", help="Positive feedback"):
                            msg_area.success("Thanks for the positive feedback!")
                    with col3:
                        if msg_area.button("👎", key=f"summary_down_{msg['content'][:10]}", help="Negative feedback"):
                            msg_area.warning("Thanks for the feedback!")
                    # If summary is dict or string, render accordingly
                    if isinstance(analysis['summary'], dict):
                        for k, v in analysis['summary'].items():
                            msg_area.markdown(f"**{k}**: {v}")
                    else:
                        msg_area.write(analysis['summary'])

                # Risk Assessment
                if 'risk_assessment' in analysis:
                    col1, col2, col3 = msg_area.columns([8, 1, 1])
                    with col1:
                        msg_area.markdown("## Risk Assessment")
                    with col2:
                        if msg_area.button("👍", key=f"risk_up_{msg['content'][:10]}", help="Positive feedback"):
                            msg_area.success("Thanks for the positive feedback!")
                    with col3:
                        if msg_area.button("👎", key=f"risk_down_{msg['content'][:10]}", help="Negative feedback"):
                            msg_area.warning("Thanks for the feedback!")
                    ra = analysis['risk_assessment']
                    if isinstance(ra, dict):
                        for issue, data in ra.items():
                            with msg_area.expander(f"{issue} (click to expand)"):
                                if isinstance(data, dict):
                                    score = data.get('score')
                                    conf = data.get('confidence')
                                    rationale = data.get('rationale')
                                    applied = data.get('applied_rules')
                                    st.write(f"Score: {score} — Confidence: {conf}")
                                    if rationale:
                                        st.markdown("**Rationale**")
                                        st.write(rationale)
                                    if applied:
                                        st.markdown("**Applied Rules**")
                                        st.write(applied)
                                else:
                                    st.write(data)

                # Security Findings
                if 'security_findings' in analysis:
                    col1, col2, col3 = msg_area.columns([8, 1, 1])
                    with col1:
                        msg_area.markdown("## Security Findings")
                    with col2:
                        if msg_area.button("👍", key=f"findings_up_{msg['content'][:10]}", help="Positive feedback"):
                            msg_area.success("Thanks for the positive feedback!")
                    with col3:
                        if msg_area.button("👎", key=f"findings_down_{msg['content'][:10]}", help="Negative feedback"):
                            msg_area.warning("Thanks for the feedback!")
                    sf = analysis['security_findings']
                    if isinstance(sf, list):
                        for idx, finding in enumerate(sf, start=1):
                            title = finding.get('title') if isinstance(finding, dict) else f"Finding {idx}"
                            with msg_area.expander(f"{title}"):
                                st.json(finding)
                    else:
                        msg_area.write(sf)

                # Review Focus
                if 'review_focus' in analysis:
                    col1, col2, col3 = msg_area.columns([8, 1, 1])
                    with col1:
                        msg_area.markdown("## Review Focus")
                    with col2:
                        if msg_area.button("👍", key=f"focus_up_{msg['content'][:10]}", help="Positive feedback"):
                            msg_area.success("Thanks for the positive feedback!")
                    with col3:
                        if msg_area.button("👎", key=f"focus_down_{msg['content'][:10]}", help="Negative feedback"):
                            msg_area.warning("Thanks for the feedback!")
                    rf = analysis['review_focus']
                    if isinstance(rf, dict):
                        for k, v in rf.items():
                            msg_area.markdown(f"**{k}**")
                            msg_area.write(v)
                    else:
                        msg_area.write(rf)

                # Compliance Check
                if 'compliance_check' in analysis:
                    col1, col2, col3 = msg_area.columns([8, 1, 1])
                    with col1:
                        msg_area.markdown("## Compliance Check")
                    with col2:
                        if msg_area.button("👍", key=f"compliance_up_{msg['content'][:10]}", help="Positive feedback"):
                            msg_area.success("Thanks for the positive feedback!")
                    with col3:
                        if msg_area.button("👎", key=f"compliance_down_{msg['content'][:10]}", help="Negative feedback"):
                            msg_area.warning("Thanks for the feedback!")
                    cc = analysis['compliance_check']
                    if isinstance(cc, dict):
                        try:
                            # Render as table if mapping to statuses
                            rows = []
                            for cid, info in cc.items():
                                status = info.get('status') if isinstance(info, dict) else info
                                notes = info.get('notes') if isinstance(info, dict) else ''
                                rows.append({'check': cid, 'status': status, 'notes': notes})
                            msg_area.table(rows)
                        except Exception:
                            msg_area.write(cc)
                    else:
                        msg_area.write(cc)
            else:
                # If not a dict but valid JSON, show as string
                msg_area.write(str(analysis))
        except json.JSONDecodeError:
            # If not JSON, just write as text
            msg_area.write(msg['content'])

# Chat input
user_input = st.chat_input("Enter repo path (absolute) and press Enter")

if user_input:
    # append user message
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    # show user message immediately
    st.chat_message('user').write(user_input)

    # Create assistant placeholder
    assistant_placeholder = st.chat_message('assistant')
    message_area = assistant_placeholder.container()
    text_block = message_area.empty()

    # Stream response
    accumulated = ''
    with st.spinner('Analyzing repository and streaming response...'):
        for chunk in handle_repo_input(user_input):
            # update UI progressively
            accumulated += chunk
            # Try to parse as JSON if complete
            try:
                analysis = json.loads(accumulated)
                # If successful, clear any previous streaming text and display structured format
                text_block.markdown("")  # Clear streaming text
                if isinstance(analysis, dict):
                    # Raw JSON in a collapsible expander
                    with message_area.expander("🔍 Raw JSON (click to expand)"):
                        st.json(analysis)

                    # Human-readable Summary
                    if 'summary' in analysis:
                        message_area.markdown("## Summary")
                        # If summary is dict or string, render accordingly
                        if isinstance(analysis['summary'], dict):
                            for k, v in analysis['summary'].items():
                                message_area.markdown(f"**{k}**: {v}")
                        else:
                            message_area.write(analysis['summary'])

                    # Risk Assessment
                    if 'risk_assessment' in analysis:
                        message_area.markdown("## Risk Assessment")
                        ra = analysis['risk_assessment']
                        if isinstance(ra, dict):
                            for issue, data in ra.items():
                                with message_area.expander(f"{issue} (click to expand)"):
                                    if isinstance(data, dict):
                                        score = data.get('score')
                                        conf = data.get('confidence')
                                        rationale = data.get('rationale')
                                        applied = data.get('applied_rules')
                                        st.write(f"Score: {score} — Confidence: {conf}")
                                        if rationale:
                                            st.markdown("**Rationale**")
                                            st.write(rationale)
                                        if applied:
                                            st.markdown("**Applied Rules**")
                                            st.write(applied)
                                    else:
                                        st.write(data)

                    # Security Findings
                    if 'security_findings' in analysis:
                        message_area.markdown("## Security Findings")
                        sf = analysis['security_findings']
                        if isinstance(sf, list):
                            for idx, finding in enumerate(sf, start=1):
                                title = finding.get('title') if isinstance(finding, dict) else f"Finding {idx}"
                                with message_area.expander(f"{title}"):
                                    st.json(finding)
                        else:
                            message_area.write(sf)

                    # Review Focus
                    if 'review_focus' in analysis:
                        message_area.markdown("## Review Focus")
                        rf = analysis['review_focus']
                        if isinstance(rf, dict):
                            for k, v in rf.items():
                                message_area.markdown(f"**{k}**")
                                message_area.write(v)
                        else:
                            message_area.write(rf)

                    # Compliance Check
                    if 'compliance_check' in analysis:
                        message_area.markdown("## Compliance Check")
                        cc = analysis['compliance_check']
                        if isinstance(cc, dict):
                            try:
                                # Render as table if mapping to statuses
                                rows = []
                                for cid, info in cc.items():
                                    status = info.get('status') if isinstance(info, dict) else info
                                    notes = info.get('notes') if isinstance(info, dict) else ''
                                    rows.append({'check': cid, 'status': status, 'notes': notes})
                                message_area.table(rows)
                            except Exception:
                                message_area.write(cc)
                        else:
                            message_area.write(cc)
                else:
                    # If not a dict but valid JSON, show as string
                    message_area.write(str(analysis))
            except json.JSONDecodeError:
                # If not valid JSON yet, show as markdown (streaming text)
                text_block.markdown(accumulated)
            # small sleep to make UI smoother
            time.sleep(0.01)

    # append assistant final message to session
    st.session_state['messages'].append({'role': 'assistant', 'content': accumulated})
