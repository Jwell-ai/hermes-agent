"""Smoke tests for the bundled chemistry-reaction skill."""

from pathlib import Path
import subprocess
import sys

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "education"
    / "edu-chem-reaction"
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", *args],
        cwd=SKILL_DIR,
        check=True,
        capture_output=True,
        text=True,
    )


def test_kernel_self_check():
    result = _run("lib/reaction_kernel.py")
    assert result.returncode == 0


def test_generator_registry():
    result = _run("scripts/generate.py", "list")
    assert "combustion_ch4" in result.stdout
