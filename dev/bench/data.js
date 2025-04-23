window.BENCHMARK_DATA = {
  "lastUpdate": 1745443348937,
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
          "id": "8d57ad7170a8c7c8edc1a4d047f52d8d396ce41b",
          "message": "Skip build-and-test when running scheduled pipeline",
          "timestamp": "2025-04-23T17:10:30Z",
          "url": "https://github.com/LLNL/quandary/commit/8d57ad7170a8c7c8edc1a4d047f52d8d396ce41b"
        },
        "date": 1745430261710,
        "tool": "pytest",
        "benches": [
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_1]",
            "value": 0.3190468769841041,
            "unit": "iter/sec",
            "range": "stddev: 0.11413781627056346",
            "extra": "mean: 3.1343356482684612 sec\nrounds: 10"
          },
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_4]",
            "value": 0.7553856717455474,
            "unit": "iter/sec",
            "range": "stddev: 0.18496382758088187",
            "extra": "mean: 1.3238270692760126 sec\nrounds: 10"
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
          "id": "8d57ad7170a8c7c8edc1a4d047f52d8d396ce41b",
          "message": "Skip build-and-test when running scheduled pipeline",
          "timestamp": "2025-04-23T17:10:30Z",
          "url": "https://github.com/LLNL/quandary/commit/8d57ad7170a8c7c8edc1a4d047f52d8d396ce41b"
        },
        "date": 1745431614106,
        "tool": "pytest",
        "benches": [
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_1]",
            "value": 0.3072735465715446,
            "unit": "iter/sec",
            "range": "stddev: 0.2535183503085663",
            "extra": "mean: 3.2544291923521085 sec\nrounds: 10"
          },
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_4]",
            "value": 0.7815069039532573,
            "unit": "iter/sec",
            "range": "stddev: 0.04534438189758686",
            "extra": "mean: 1.2795792269287631 sec\nrounds: 10"
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
          "id": "8d57ad7170a8c7c8edc1a4d047f52d8d396ce41b",
          "message": "Skip build-and-test when running scheduled pipeline",
          "timestamp": "2025-04-23T17:10:30Z",
          "url": "https://github.com/LLNL/quandary/commit/8d57ad7170a8c7c8edc1a4d047f52d8d396ce41b"
        },
        "date": 1745432149039,
        "tool": "pytest",
        "benches": [
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_1]",
            "value": 0.3152970256073725,
            "unit": "iter/sec",
            "range": "stddev: 0.21775956853860487",
            "extra": "mean: 3.1716125392355026 sec\nrounds: 10"
          },
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_4]",
            "value": 0.7874940727322142,
            "unit": "iter/sec",
            "range": "stddev: 0.01463007449432465",
            "extra": "mean: 1.2698508276138454 sec\nrounds: 10"
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
          "id": "292b38ef7bfc416aec792e3094c75254a4750b4b",
          "message": "WIP Make quandary slower for testing purposes",
          "timestamp": "2025-04-23T20:34:01Z",
          "url": "https://github.com/LLNL/quandary/commit/292b38ef7bfc416aec792e3094c75254a4750b4b"
        },
        "date": 1745441786963,
        "tool": "pytest",
        "benches": [
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_1]",
            "value": 0.26153997276882407,
            "unit": "iter/sec",
            "range": "stddev: 0.3621725889705462",
            "extra": "mean: 3.823507318645716 sec\nrounds: 10"
          },
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_4]",
            "value": 0.5735592484483606,
            "unit": "iter/sec",
            "range": "stddev: 0.0039271924399252285",
            "extra": "mean: 1.7434990416513756 sec\nrounds: 10"
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
          "id": "292b38ef7bfc416aec792e3094c75254a4750b4b",
          "message": "WIP Make quandary slower for testing purposes",
          "timestamp": "2025-04-23T20:34:01Z",
          "url": "https://github.com/LLNL/quandary/commit/292b38ef7bfc416aec792e3094c75254a4750b4b"
        },
        "date": 1745443348051,
        "tool": "pytest",
        "benches": [
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_1]",
            "value": 0.28026779194115436,
            "unit": "iter/sec",
            "range": "stddev: 0.11365894199137552",
            "extra": "mean: 3.5680161215597765 sec\nrounds: 10"
          },
          {
            "name": "performance_tests/performance_test.py::test_eval[config_template_4]",
            "value": 0.5696380972320826,
            "unit": "iter/sec",
            "range": "stddev: 0.03784696982869064",
            "extra": "mean: 1.7555005623027682 sec\nrounds: 10"
          }
        ]
      }
    ]
  }
}