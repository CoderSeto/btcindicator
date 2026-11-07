"""
Run from the repo root:  python -m pytest -q -s review/demo_cross_session.py -k one_session

Demonstrates (does not fix) that app.py mutates the shared `config` module, so one browser
session's sidebar choices become the next session's defaults. Uses tests/test_app.py's harness.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))
from test_app import cfg, env, run, slider  # noqa: E402,F401  (env is the pytest fixture)


def test_one_session_changes_another_sessions_defaults(env):  # noqa: F811
    a = run()
    a.sidebar.selectbox[0].select("ETH-USD").run()
    slider(a, "Alert threshold").set_value(7).run()
    b = run()                                   # a brand-new, independent session
    print("\nsession B opened with pair =", b.sidebar.selectbox[0].value,
          "| alert threshold =", slider(b, "Alert threshold").value)
    assert b.sidebar.selectbox[0].value == "ETH-USD"
