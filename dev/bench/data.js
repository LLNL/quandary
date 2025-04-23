window.BENCHMARK_DATA = {
  "lastUpdate": 1745428988731,
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
      },
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
          "id": "98669ccbd4fabdf7b61e38d16603798e02df0934",
          "message": "Add performance tests as seperate pipeline that also runs weekly",
          "timestamp": "2025-04-23T15:30:11Z",
          "url": "https://github.com/LLNL/quandary/commit/98669ccbd4fabdf7b61e38d16603798e02df0934"
        },
        "date": 1745427545182,
        "tool": "pytest",
        "benches": [
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_1]",
            "value": 0.3194566390553569,
            "unit": "iter/sec",
            "range": "stddev: 0.11251002835677128",
            "extra": "mean: 3.130315284593962 sec\nrounds: 10"
          },
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_4]",
            "value": 0.6572511182421392,
            "unit": "iter/sec",
            "range": "stddev: 0.5887897381815197",
            "extra": "mean: 1.5214884725864977 sec\nrounds: 10"
          }
        ]
      },
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
          "id": "6a65661e87e40f239f9118402b1bbfe20a3ee5c0",
          "message": "Skip build-and-test when running scheduled pipeline",
          "timestamp": "2025-04-23T17:10:30Z",
          "url": "https://github.com/LLNL/quandary/commit/6a65661e87e40f239f9118402b1bbfe20a3ee5c0"
        },
        "date": 1745428987765,
        "tool": "pytest",
        "benches": [
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_1]",
            "value": 0.3156425005530852,
            "unit": "iter/sec",
            "range": "stddev: 0.14779529653122978",
            "extra": "mean: 3.1681411668192596 sec\nrounds: 10"
          },
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_4]",
            "value": 0.7523788125407572,
            "unit": "iter/sec",
            "range": "stddev: 0.07977295655516088",
            "extra": "mean: 1.3291177041828632 sec\nrounds: 10"
          }
        ]
      }
    ]
  }
}