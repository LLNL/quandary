window.BENCHMARK_DATA = {
  "lastUpdate": 1745367299125,
  "repoUrl": "https://github.com/LLNL/quandary",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "Tara Drwenski",
            "username": "tdrwenski",
            "email": "drwenski1@llnl.gov"
          },
          "committer": {
            "name": "Tara Drwenski",
            "username": "tdrwenski",
            "email": "drwenski1@llnl.gov"
          },
          "id": "f34cfdd0f29da222167f09a2ae39187555292590",
          "message": "Add gitlab job for performance tests and github benchmark workflow for visualization",
          "timestamp": "2025-04-22T21:15:15Z",
          "url": "https://github.com/LLNL/quandary/commit/f34cfdd0f29da222167f09a2ae39187555292590"
        },
        "date": 1745367298713,
        "tool": "pytest",
        "benches": [
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_1]",
            "value": 0.12026445806261485,
            "unit": "iter/sec",
            "range": "stddev: 6.6183569422603075",
            "extra": "mean: 8.31500857451465 sec\nrounds: 10"
          },
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_4]",
            "value": 0.2595668852071951,
            "unit": "iter/sec",
            "range": "stddev: 1.704812843254234",
            "extra": "mean: 3.8525715605122968 sec\nrounds: 10"
          }
        ]
      }
    ]
  }
}