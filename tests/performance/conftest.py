"""Performance test configuration for pytest."""
from tests.utils.conftest_common import add_common_options


def pytest_addoption(parser):
    """Add command line options to pytest."""
    add_common_options(parser)
