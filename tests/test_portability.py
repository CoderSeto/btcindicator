"""
The suite must pass on a machine whose default text encoding isn't UTF-8.

On Windows, `open()` and `Path.read_text()` without `encoding=` decode with the
locale's code page (cp1252), and `app.py` holds UTF-8 characters. A bare
`read_text()` of it raised UnicodeDecodeError while *collecting* test_app.py, so
the whole suite stopped before running a test -- on the user's machine, not in
CI. Here the same thing is forced on any OS: a child interpreter with UTF-8 mode
off and an ASCII/C locale (on Windows LC_ALL is ignored and PYTHONUTF8=0 leaves
cp1252, which fails the same way).
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _run_pytest_without_utf8(*args):
    env = dict(os.environ, PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C", LANG="C")
    env.pop("PYTHONIOENCODING", None)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=300)


def test_the_child_really_runs_without_utf8():
    env = dict(os.environ, PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C", LANG="C")
    out = subprocess.run([sys.executable, "-c", "import locale; print(locale.getpreferredencoding(False))"],
                         env=env, capture_output=True, text=True).stdout.strip().lower()
    assert "utf" not in out, f"child still decodes as {out}; the test below would prove nothing"


def test_every_test_module_collects_without_a_utf8_locale():
    r = _run_pytest_without_utf8("--collect-only", "tests")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_tests_that_read_app_py_pass_without_a_utf8_locale():
    r = _run_pytest_without_utf8("tests/test_app.py::test_the_page_uses_no_deprecated_streamlit_apis")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
