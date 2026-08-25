"""Smoke tests for the bundled solid-geometry skill."""

from pathlib import Path
import subprocess
import sys

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "education"
    / "edu-solid-geometry"
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
    result = _run("lib/geometry_kernel.py")
    assert result.returncode == 0


def test_generator_default_output(tmp_path):
    output = tmp_path / "lesson.html"
    result = _run("scripts/generate.py", str(output))
    assert result.returncode == 0
    assert output.is_file()
    assert "__LESSON_DATA__" not in output.read_text(encoding="utf-8")
