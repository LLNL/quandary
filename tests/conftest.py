def pytest_addoption(parser):
    parser.addoption(
        "--exact",
        action="store_true",
        default=False,
        help="Use exact comparison for floating point numbers"
    )
