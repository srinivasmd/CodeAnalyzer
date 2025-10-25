import os
import json
import tempfile
from types import SimpleNamespace

import pytest

from main import run_static_analysis


class DummyCompletedProcess:
    def __init__(self, stdout=''):
        self.stdout = stdout


def test_run_static_analysis_monkeypatched(monkeypatch, tmp_path):
    # create a simple python file
    p = tmp_path / "sample.py"
    p.write_text("def foo():\n    return 1\n")

    file_path = str(p)

    # monkeypatch subprocess.run to return predictable JSON outputs for pylint and bandit
    def fake_run(cmd, capture_output=True, text=True, cwd=None):
        # return empty list JSON for diagnostics
        return SimpleNamespace(stdout='[]')

    monkeypatch.setattr('subprocess.run', fake_run)

    # monkeypatch run_sonarqube_analysis to avoid external calls
    monkeypatch.setattr('main.run_sonarqube_analysis', lambda fp: {'issues': []})

    results = run_static_analysis([file_path])

    assert file_path in results
    # pylint and bandit should parse to empty lists
    assert 'pylint' in results[file_path]
    assert isinstance(results[file_path]['pylint'], list)
    assert results[file_path]['pylint'] == []
    assert 'bandit' in results[file_path]
    assert isinstance(results[file_path]['bandit'], list)
    assert results[file_path]['bandit'] == []
    assert 'sonarqube' in results[file_path]
    assert results[file_path]['sonarqube'] == {'issues': []}
