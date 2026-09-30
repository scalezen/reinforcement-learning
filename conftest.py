"""Empty on purpose.

pytest prepends the directory of every conftest.py to sys.path, so having this
file at the repo root is what makes `from gbm import ...` and
`from StoppingNet import ...` resolve when the tests live in tests/.

It also registers the `slow` marker, so `pytest -m "not slow"` (what CI runs on
every push) doesn't warn about an unknown marker.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: takes minutes (network training). Runs nightly, not on every push.",
    )