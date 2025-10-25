import os
import subprocess
import json
import requests
import time
from git import Repo
from dotenv import load_dotenv
from typing import Dict, List, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from metrics import metrics_tracker

# Load environment variables
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_API_ENDPOINT = os.getenv("NVIDIA_API_ENDPOINT")
NVIDIA_API_MODEL = os.getenv("NVIDIA_API_MODEL")

# Initialize headers for NVIDIA API
HEADERS = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}


@dataclass
class FileChange:
    """Represents a change in a file"""
    filename: str
    diff: str
    is_cosmetic: bool
    lines_changed: int
    start_line: int
    end_line: int


def is_cosmetic_change(diff_content: str) -> bool:
    """Determine if a change is purely cosmetic."""
    cosmetic_indicators = [
        'whitespace', 'formatting', 'indent',
        'import order', 'newline', 'comment'
    ]
    return any(indicator in diff_content.lower() for indicator in cosmetic_indicators)


def get_pr_diff(repo_path: str, pr_url: str) -> List[FileChange]:
    """Fetch and parse PR diff from a local repo or generic diff parsing for non-Git repos, supporting large diffs by chunking."""
    changes: List[FileChange] = []

    try:
        repo = Repo(repo_path)
        # Git repo handling
        head = repo.head.commit
        parents = head.parents
        if parents:
            prev = parents[0]
        else:
            prev = None

        if prev is not None:
            diff_index = prev.diff(head)
        else:
            diff_index = head.diff(NULL_TREE:=None)

        for diff_item in diff_index:
            filename = diff_item.a_path or diff_item.b_path
            if not filename:
                continue

            try:
                if prev is not None:
                    diff_content = repo.git.diff(prev.hexsha, head.hexsha, '--', filename)
                else:
                    diff_content = repo.git.diff(head.hexsha, '--', filename)
            except Exception:
                diff_content = getattr(diff_item, 'diff', '') or getattr(diff_item, 'patch', '') or ''

            # Chunk large diffs
            if len(diff_content) > 50000:
                print(f"Warning: Large diff for {filename}, chunking for analysis.")
                chunks = [diff_content[i:i+50000] for i in range(0, len(diff_content), 50000)]
                for i, chunk in enumerate(chunks):
                    chunk_filename = f"{filename}_chunk_{i}"
                    is_cosmetic = is_cosmetic_change(chunk)
                    lines_changed = len(chunk.splitlines())
                    start_line = 0
                    changes.append(FileChange(
                        filename=chunk_filename,
                        diff=chunk,
                        is_cosmetic=is_cosmetic,
                        lines_changed=lines_changed,
                        start_line=start_line,
                        end_line=start_line + lines_changed
                    ))
            else:
                is_cosmetic = is_cosmetic_change(diff_content)
                start_line = 0
                lines_changed = len(diff_content.splitlines()) if diff_content else 0
                for line in diff_content.splitlines():
                    if line.startswith('@@'):
                        try:
                            parts = line.split(' ')
                            new_file_info = parts[2] if len(parts) >= 3 else parts[1]
                            if new_file_info.startswith('+'):
                                nums = new_file_info[1:].split(',')
                                start_line = int(nums[0])
                            else:
                                start_line = abs(int(new_file_info.split(',')[0]))
                        except Exception:
                            start_line = 0
                        break
                changes.append(FileChange(
                    filename=filename,
                    diff=diff_content,
                    is_cosmetic=is_cosmetic,
                    lines_changed=lines_changed,
                    start_line=start_line,
                    end_line=(start_line + lines_changed if start_line else lines_changed)
                ))

    except Exception as e:
        # Non-Git repo: generic diff parsing (e.g., from PR URL or file)
        print(f"Warning: Not a Git repo or Git error ({e}). Attempting generic diff parsing.")
        # Assume pr_url is a diff URL or file path; for simplicity, try to read as diff content
        try:
            if pr_url.startswith('http'):
                diff_resp = requests.get(pr_url, timeout=30)
                if diff_resp.ok:
                    diff_content = diff_resp.text
                else:
                    diff_content = ''
            else:
                with open(pr_url, 'r') as f:
                    diff_content = f.read()
        except Exception:
            diff_content = ''

        # Parse generic unified diff
        if diff_content:
            files = diff_content.split('diff --git ')
            for file_diff in files[1:]:  # Skip first empty
                lines = file_diff.splitlines()
                filename = lines[0].split(' ')[-1] if lines else 'unknown'
                # Simple chunking for large diffs
                if len(file_diff) > 50000:
                    chunks = [file_diff[i:i+50000] for i in range(0, len(file_diff), 50000)]
                    for i, chunk in enumerate(chunks):
                        chunk_filename = f"{filename}_chunk_{i}"
                        is_cosmetic = is_cosmetic_change(chunk)
                        lines_changed = len(chunk.splitlines())
                        start_line = 0
                        changes.append(FileChange(
                            filename=chunk_filename,
                            diff=chunk,
                            is_cosmetic=is_cosmetic,
                            lines_changed=lines_changed,
                            start_line=start_line,
                            end_line=start_line + lines_changed
                        ))
                else:
                    is_cosmetic = is_cosmetic_change(file_diff)
                    lines_changed = len(file_diff.splitlines())
                    start_line = 0
                    changes.append(FileChange(
                        filename=filename,
                        diff=file_diff,
                        is_cosmetic=is_cosmetic,
                        lines_changed=lines_changed,
                        start_line=start_line,
                        end_line=start_line + lines_changed
                    ))

    return changes


def run_sonarqube_analysis(file_path: str) -> Dict:
    """Run SonarQube analysis on a file if enabled in config."""
    try:
        # Load config to check if SonarQube is enabled
        with open('config.json', 'r') as f:
            config = json.load(f)
        sonar_config = config.get('static_analysis', {}).get('sonarqube', {})
        if not sonar_config.get('enabled', False):
            return {}

        host_url = sonar_config.get('host_url', 'http://localhost:9000')
        token = sonar_config.get('token', '')
        project_key = sonar_config.get('project_key', 'pr-analysis')

        sonar_scanner = subprocess.run([
            "sonar-scanner",
            f"-Dsonar.sources={file_path}",
            f"-Dsonar.projectKey={project_key}",
            f"-Dsonar.host.url={host_url}",
            f"-Dsonar.login={token}"
        ], capture_output=True, text=True)

        # Fetch results from SonarQube API
        response = requests.get(
            f"{host_url}/api/issues/search?componentKeys={project_key}",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()
    except Exception as e:
        print(f"SonarQube analysis failed: {str(e)}")
        return {}


def run_static_analysis(file_paths: List[str]) -> Dict[str, Any]:
    """Run comprehensive static analysis using multiple tools with support for more languages and graceful fallbacks."""
    results = {}

    def _run_tool(command: List[str], tool_name: str) -> str:
        """Helper to run a tool with error handling."""
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"Warning: {tool_name} failed with return code {result.returncode}: {result.stderr}")
                return ''
            return result.stdout
        except FileNotFoundError:
            print(f"Warning: {tool_name} not installed or not found in PATH. Skipping.")
            return ''
        except subprocess.TimeoutExpired:
            print(f"Warning: {tool_name} timed out. Skipping.")
            return ''
        except Exception as e:
            print(f"Warning: Error running {tool_name}: {e}. Skipping.")
            return ''

    def _run_pylint(fp: str) -> str:
        return _run_tool(["pylint", fp, "--output-format=json"], "Pylint")

    def _run_bandit(fp: str) -> str:
        return _run_tool(["bandit", "-f", "json", fp], "Bandit")

    def _run_eslint(fp: str) -> str:
        return _run_tool(["eslint", fp, "--format=json"], "ESLint")

    def _run_spotbugs(fp: str) -> str:
        # SpotBugs requires class files; assume source analysis or skip if not set up
        return _run_tool(["spotbugs", "-textui", "-effort:max", "-low", fp], "SpotBugs")

    def _run_dotnet_analyzer(fp: str) -> str:
        return _run_tool(["dotnet", "build", fp, "/p:RunAnalyzers=true", "/p:RunAnalyzersDuringBuild=true"], "DotNet Analyzer")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for file_path in file_paths:
            ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
            if ext == "py":
                pylint_future = executor.submit(_run_pylint, file_path)
                bandit_future = executor.submit(_run_bandit, file_path)
                sonar_future = executor.submit(run_sonarqube_analysis, file_path)
                futures[file_path] = (pylint_future, bandit_future, sonar_future)
            elif ext == "js":
                eslint_future = executor.submit(_run_eslint, file_path)
                sonar_future = executor.submit(run_sonarqube_analysis, file_path)
                futures[file_path] = (eslint_future, sonar_future)
            elif ext == "java":
                spotbugs_future = executor.submit(_run_spotbugs, file_path)
                sonar_future = executor.submit(run_sonarqube_analysis, file_path)
                futures[file_path] = (spotbugs_future, sonar_future)
            elif ext == "cs":
                dotnet_future = executor.submit(_run_dotnet_analyzer, file_path)
                sonar_future = executor.submit(run_sonarqube_analysis, file_path)
                futures[file_path] = (dotnet_future, sonar_future)
            else:
                # For unsupported languages, still run SonarQube if available
                sonar_future = executor.submit(run_sonarqube_analysis, file_path)
                futures[file_path] = (sonar_future,)

        # collect results
        for file_path, futs in futures.items():
            ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
            try:
                if ext == 'py':
                    pylint_out = futs[0].result()
                    bandit_out = futs[1].result()
                    sonar_out = futs[2].result()
                    results[file_path] = {
                        'pylint': json.loads(pylint_out) if pylint_out else {},
                        'bandit': json.loads(bandit_out) if bandit_out else {},
                        'sonarqube': sonar_out
                    }
                elif ext == 'js':
                    eslint_out = futs[0].result()
                    sonar_out = futs[1].result()
                    results[file_path] = {
                        'eslint': json.loads(eslint_out) if eslint_out else {},
                        'sonarqube': sonar_out
                    }
                elif ext == 'java':
                    spotbugs_out = futs[0].result()
                    sonar_out = futs[1].result()
                    results[file_path] = {
                        'spotbugs': spotbugs_out,
                        'sonarqube': sonar_out
                    }
                elif ext == 'cs':
                    dotnet_out = futs[0].result()
                    sonar_out = futs[1].result()
                    results[file_path] = {
                        'dotnet_analyzer': dotnet_out,
                        'sonarqube': sonar_out
                    }
                else:
                    sonar_out = futs[0].result()
                    results[file_path] = {'sonarqube': sonar_out}
            except Exception:
                # Non-fatal: include whatever we managed
                results.setdefault(file_path, {})

    return results


def fetch_github_pr_feedback(repo_path: str) -> Dict[str, Any]:
    """Fetch and analyze historical PR feedback from GitHub for the repo.

    Detects remote origin, fetches recent PRs' comments and reviews, analyzes patterns.
    """
    feedback = {
        "common_reviewer_comments": {},
        "approval_rates": {},
        "recurring_issues": {},
        "frequent_reviewers": []
    }

    try:
        repo = Repo(repo_path)
        # Detect remote origin
        origin = repo.remote('origin')
        origin_url = origin.url
        # Parse GitHub URL: https://github.com/owner/repo or git@github.com:owner/repo
        if 'github.com' in origin_url:
            if origin_url.startswith('https'):
                parts = origin_url.split('/')
                owner = parts[-2]
                repo_name = parts[-1].replace('.git', '')
            else:
                # SSH: git@github.com:owner/repo.git
                parts = origin_url.split(':')[1].split('/')
                owner = parts[0]
                repo_name = parts[1].replace('.git', '')

            gh_token = os.getenv('GITHUB_TOKEN')
            headers = {'Accept': 'application/vnd.github.v3+json'}
            if gh_token:
                headers['Authorization'] = f'token {gh_token}'

            # Fetch recent PRs (last 10)
            prs_url = f'https://api.github.com/repos/{owner}/{repo_name}/pulls?state=all&per_page=10'
            prs_resp = requests.get(prs_url, headers=headers, timeout=20)
            if prs_resp.ok:
                prs = prs_resp.json()
                for pr in prs:
                    pr_number = pr['number']
                    # Fetch comments
                    comments_url = f'https://api.github.com/repos/{owner}/{repo_name}/issues/{pr_number}/comments'
                    comments_resp = requests.get(comments_url, headers=headers, timeout=20)
                    if comments_resp.ok:
                        comments = comments_resp.json()
                        for comment in comments:
                            body = (comment.get('body') or '').lower()
                            user = comment.get('user', {}).get('login')
                            if user:
                                if user not in feedback['frequent_reviewers']:
                                    feedback['frequent_reviewers'].append(user)
                            # Analyze common comments
                            for keyword in ['bug', 'fix', 'security', 'refactor', 'cleanup', 'test', 'naming', 'style']:
                                if keyword in body:
                                    feedback['common_reviewer_comments'][keyword] = feedback['common_reviewer_comments'].get(keyword, 0) + 1
                            # Recurring issues
                            if 'security' in body or 'vuln' in body:
                                feedback['recurring_issues']['security'] = feedback['recurring_issues'].get('security', 0) + 1

                    # Fetch reviews
                    reviews_url = f'https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_number}/reviews'
                    reviews_resp = requests.get(reviews_url, headers=headers, timeout=20)
                    if reviews_resp.ok:
                        reviews = reviews_resp.json()
                        approvals = sum(1 for r in reviews if r.get('state') == 'APPROVED')
                        total_reviews = len(reviews)
                        if total_reviews > 0:
                            rate = approvals / total_reviews
                            feedback['approval_rates'][f'pr_{pr_number}'] = rate
    except Exception as e:
        # Non-fatal
        pass

    return feedback


def get_historical_patterns(repo_path: str) -> Dict[str, Any]:
    """Analyze historical code review patterns from recent commits and GitHub PR feedback.

    Scans commit messages and fetches PR comments/reviews for patterns.
    """
    repo = Repo(repo_path)
    patterns = {
        "high_risk_files": [],
        "common_issues": {},
        "frequent_reviewers": [],
        "pr_feedback": {}
    }

    try:
        for commit in repo.iter_commits('HEAD', max_count=100):
            # Heuristic: if commit message contains 'fix'/'security'/'vuln' treat as issue
            msg = (commit.message or '').lower()
            if 'security' in msg or 'vuln' in msg or 'fix' in msg:
                for f in commit.stats.files.keys():
                    if f not in patterns['high_risk_files']:
                        patterns['high_risk_files'].append(f)
            # simple keyword counts
            for keyword in ['bug', 'fix', 'security', 'refactor', 'cleanup', 'test']:
                if keyword in msg:
                    patterns['common_issues'][keyword] = patterns['common_issues'].get(keyword, 0) + 1
    except Exception:
        # Non-fatal; return whatever we gathered so far
        pass

    # Fetch GitHub PR feedback
    patterns['pr_feedback'] = fetch_github_pr_feedback(repo_path)

    return patterns


def analyze_with_nvidia(changes: List[FileChange], static_results: Dict, 
                        style_guide: str, security_checklist: str, 
                        historical_patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze changes using NVIDIA's OpenAI-compatible chat completions endpoint or fallback heuristics."""
    # Build a compact human prompt
    changes_by_type = {}
    total_changes = sum(c.lines_changed for c in changes) or 1
    for c in changes:
        t = 'Cleanup' if c.is_cosmetic else 'Logic Change'
        changes_by_type[t] = changes_by_type.get(t, 0) + (c.lines_changed / total_changes) * 100

    prompt = {
        'changes_overview': changes_by_type,
        'top_files': [c.filename for c in sorted(changes, key=lambda x: x.lines_changed, reverse=True)[:10]],
        'static_results_summary': {k: list(v.keys())[:5] for k, v in static_results.items()},
        'historical_patterns': historical_patterns,
        'style_guide': style_guide,
        'security_checklist': security_checklist
    }

    # Fallback if NVIDIA not configured
    if not (NVIDIA_API_KEY and NVIDIA_API_ENDPOINT and NVIDIA_API_MODEL):
        key_mods = prompt['top_files'][:5]
        return {
            'summary': {
                'changes_breakdown': changes_by_type,
                'key_modifications': key_mods,
                'architectural_impact': 'Unknown - requires model analysis'
            },
            'risk_assessment': {
                'high_risk_areas': [f for f in key_mods if any(k in f.lower() for k in ['auth', 'payment', 'db'])],
                'test_coverage': 'Unknown',
                'deployment_impact': 'Unknown'
            },
            'security_findings': {'vulnerabilities': [], 'secure_coding_violations': [], 'dependency_risks': []},
            'review_focus': {'priority_files': key_mods, 'deep_dive_links': [], 'suggested_reviewers': []},
            'compliance_check': {'style_violations': [], 'missing_requirements': [], 'documentation_needs': []}
        }

    # Call NVIDIA integrate OpenAI-compatible chat completions endpoint
    base_endpoint = (NVIDIA_API_ENDPOINT or "https://integrate.api.nvidia.com/v1").rstrip('/')
    invoke_url = f"{base_endpoint}/chat/completions"

    system_msg = {'role': 'system', 'content': 'You are an expert code reviewer focusing on security, architecture, and best practices.'}
    user_msg = {'role': 'user', 'content': json.dumps(prompt)}

    payload = {
        'model': NVIDIA_API_MODEL,
        'messages': [system_msg, user_msg],
        'temperature': 0.3,
        'top_p': 0.9,
        'max_tokens': 2000
    }

    headers = {
        'Authorization': f'Bearer {NVIDIA_API_KEY}',
        'x-api-key': NVIDIA_API_KEY,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    # Helper to extract content from a standard OpenAI-like response
    def _extract_content(resp_json: Dict) -> str:
        if isinstance(resp_json, dict) and 'choices' in resp_json and isinstance(resp_json['choices'], list) and resp_json['choices']:
            first = resp_json['choices'][0]
            if isinstance(first, dict):
                return first.get('message', {}).get('content') or first.get('text') or ''
            return str(first)
        return ''

    expected_keys = ['summary', 'risk_assessment', 'security_findings', 'review_focus', 'compliance_check']

    # Try initial call and up to 2 re-prompts to enforce JSON schema
    attempts = 0
    max_attempts = 3
    last_content = ''
    while attempts < max_attempts:
        attempts += 1
        try:
            resp = requests.post(invoke_url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            content = _extract_content(result)
            last_content = content or last_content

            if not content:
                # nothing useful returned, try again
                continue

            # Try to parse JSON
            try:
                parsed = json.loads(content)
                # validate required keys
                if isinstance(parsed, dict) and all(k in parsed for k in expected_keys):
                    return parsed
                else:
                    # If keys missing, prepare a re-prompt explaining which keys are missing
                    missing = [k for k in expected_keys if k not in parsed]
                    retry_user = {
                        'role': 'user',
                        'content': (
                            f"The previous response is missing required keys: {missing}.\n"
                            "Please respond ONLY with a single JSON object that contains the following keys: "
                            f"{expected_keys}. Each field should be properly populated (empty lists/dicts are OK). Do not include any explanatory text outside the JSON."
                        )
                    }
                    # Append the assistant's last content as context and ask for a corrected JSON
                    payload['messages'] = [system_msg, user_msg, {'role': 'assistant', 'content': content}, retry_user]
                    # continue loop to retry
                    continue
            except json.JSONDecodeError:
                # Invalid JSON, ask model to reformat strictly
                example = json.dumps({"error": "<explanation>"})
                retry_user = {
                    'role': 'user',
                    'content': (
                        "The assistant's previous reply is not valid JSON. Please reply ONLY with a valid JSON object with keys: "
                        f"{expected_keys}. If you cannot, respond with: {example}"
                    )
                }
                payload['messages'] = [system_msg, user_msg, {'role': 'assistant', 'content': content}, retry_user]
                continue

        except Exception as e:
            print(f"Error calling NVIDIA chat completions at {invoke_url}: {e}")
            # On error, break and fallback
            break

    # If we reached here, attempts exhausted or error occurred. Try to salvage last_content
    try:
        return json.loads(last_content)
    except Exception:
        # Return a structured fallback including raw model output for debugging
        return {
            'summary': {'changes_breakdown': changes_by_type, 'key_modifications': prompt['top_files'][:5], 'architectural_impact': 'Unknown'},
            'risk_assessment': {'model_output': last_content},
            'security_findings': {},
            'review_focus': {'priority_files': prompt['top_files'][:5], 'deep_dive_links': [], 'suggested_reviewers': []},
            'compliance_check': {}
        }

def main(repo_path: str):
    """Main function to analyze pull requests."""
    # Load team standards
    style_guide = """
    1. Follow PEP8 for Python, Airbnb for JavaScript
    2. Mandatory error handling for all external calls
    3. Use type hints in Python
    4. Document all public APIs
    5. Test coverage must be >80%
    """

    security_checklist = """
    1. OWASP Top 10 compliance
    2. No raw SQL queries
    3. Input validation for all user data
    4. Proper authentication checks
    5. Secure password handling
    6. Rate limiting for APIs
    7. Proper error handling without info leakage
    """

    start_time = time.time()
    try:
        # Get changes with detailed analysis
        changes = get_pr_diff(repo_path, "https://github.com/example/repo/pull/347")

        # Get file paths from changes
        file_paths = [change.filename for change in changes]

        # Run comprehensive static analysis
        static_results = run_static_analysis(file_paths)

        # Get historical patterns
        historical_patterns = get_historical_patterns(repo_path)

        # Analyze with NVIDIA (or fallback heuristics)
        analysis = analyze_with_nvidia(
            changes,
            static_results,
            style_guide,
            security_checklist,
            historical_patterns
        )

        # Generate detailed report
        report = {
            "summary": analysis["summary"],
            "risk_assessment": analysis["risk_assessment"],
            "security_findings": analysis["security_findings"],
            "review_focus": {
                **analysis["review_focus"],
                "deep_dive_links": [
                    f"vscode://file/{repo_path}/{file}#L{change.start_line}-L{change.end_line}"
                    for file, change in zip(file_paths, changes)
                    if not change.is_cosmetic
                ]
            },
            "compliance_check": analysis["compliance_check"],
            "statistics": {
                "total_files": len(file_paths),
                "total_changes": sum(change.lines_changed for change in changes),
                "cosmetic_changes": sum(change.lines_changed for change in changes if change.is_cosmetic),
                "substantive_changes": sum(change.lines_changed for change in changes if not change.is_cosmetic)
            }
        }

        # Calculate analysis time
        analysis_time = time.time() - start_time

        # Log to metrics
        total_lines = sum(change.lines_changed for change in changes)
        total_files = len(file_paths)
        log_id = metrics_tracker.log_analysis(repo_path, "https://github.com/example/repo/pull/347",
                                             report, analysis_time, total_lines, total_files)

        # Output results
        print(json.dumps(report, indent=2))

    except Exception as e:
        print(f"Error analyzing pull request: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    main(repo_path)