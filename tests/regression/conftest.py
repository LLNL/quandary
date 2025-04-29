"""Regression test configuration for pytest."""

from test_utils.conftest_common import add_common_options


def pytest_addoption(parser):
    """Add command line options to pytest."""
    add_common_options(parser)

    parser.addoption(
        "--exact",
        action="store_true",
        default=False,
        help="Use exact comparison for floating point numbers"
    )
