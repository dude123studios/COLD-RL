"""
Execution-based scorer for LiveCodeBench problems.
Extracts Python code from model responses and runs against public test cases.
"""

import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def extract_code(response: str) -> str:
    """Extract Python code from a model response with ```python ... ``` blocks."""
    # Try fenced code block first
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: any fenced block
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Last resort: everything after "Solution:" or "Code:"
    m = re.search(r"(?:Solution|Code|Answer):\s*\n(.*)", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def run_code_on_test(code: str, test_input: str, expected_output: str, timeout: float = 5.0) -> bool:
    """
    Execute `code` with `test_input` on stdin, compare stdout to `expected_output`.
    Returns True if output matches (after stripping whitespace).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = Path(tmpdir) / "solution.py"
        code_file.write_text(code)
        try:
            result = subprocess.run(
                [sys.executable, str(code_file)],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            actual = result.stdout.strip()
            expected = expected_output.strip()
            return actual == expected
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


def score_code_response(response: str, test_cases: list[dict]) -> bool:
    """
    Returns True if the code in `response` passes all public test cases.
    test_cases: list of {input, output, testtype}
    """
    code = extract_code(response)
    if not code:
        return False
    # Must pass all public test cases
    for tc in test_cases:
        if tc.get("testtype") == "stdin":
            if not run_code_on_test(code, tc["input"], tc["output"]):
                return False
    return True


def score_livecodebench(
    rollouts: dict[str, list[str]],
    problems: list[dict],
) -> dict:
    """
    Compute pass@1 for LiveCodeBench using public test case execution.

    Returns dict with pass_at_1 (accuracy: fraction passing all tests).
    """
    import numpy as np

    correct_by_problem = []
    for p in problems:
        pid = p["id"]
        test_cases = p.get("public_test_cases", [])
        rolls = rollouts.get(pid, [])
        correct = [score_code_response(r, test_cases) for r in rolls]
        correct_by_problem.append(correct)

    def pass_at_k(correct_list, k):
        n = len(correct_list)
        c = sum(correct_list)
        if n < k:
            return float(c > 0)
        if n == c:
            return 1.0
        return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k) if n - c - i > 0]))

    results = {}
    for k in [1, 4, 8, 16]:
        if all(len(c) >= k for c in correct_by_problem):
            results[f"pass_at_{k}"] = float(
                np.mean([pass_at_k(c, k) for c in correct_by_problem])
            )

    return results
