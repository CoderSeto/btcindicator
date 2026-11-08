"""
requirements-lock.txt pins the exact versions the suite last passed with (D16: every dependency
in requirements.txt is a `>=` floor, so two machines can silently run different versions). These
tests keep the two files in step: every direct requirement is pinned in the lock, and each pin
satisfies the floor requirements.txt declares.
"""
import pathlib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _lines(name):
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield line


def _lock():
    pins = {}
    for line in _lines("requirements-lock.txt"):
        req = Requirement(line)
        specs = list(req.specifier)
        assert len(specs) == 1 and specs[0].operator == "==", f"not an exact pin: {line}"
        pins[canonicalize_name(req.name)] = Version(specs[0].version)
    return pins


def test_every_lock_line_is_an_exact_pin():
    assert len(_lock()) > 0


def test_every_requirement_is_pinned_and_the_pin_meets_its_minimum():
    pins = _lock()
    for line in _lines("requirements.txt"):
        req = Requirement(line)
        name = canonicalize_name(req.name)
        assert name in pins, f"{req.name} is in requirements.txt but not pinned in requirements-lock.txt"
        assert req.specifier.contains(pins[name], prereleases=True), \
            f"{req.name}=={pins[name]} does not satisfy requirements.txt's {req.specifier}"


def test_pytest_is_pinned_too():
    assert "pytest" in _lock()          # the suite's own runner, so a fresh setup reproduces the run
