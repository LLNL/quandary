window.BENCHMARK_DATA = {
  "lastUpdate": 1745535644309,
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
          "id": "da5a77d076f0a812a8b5694c82f72cb16df314f7",
          "message": "Revert custom pytest hook and put benchmark format conversion in own file",
          "timestamp": "2025-04-24T22:48:06Z",
          "url": "https://github.com/LLNL/quandary/commit/da5a77d076f0a812a8b5694c82f72cb16df314f7"
        },
        "date": 1745535643724,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "test_eval[config_template_1] - Time",
            "value": 3.0551612270530315,
            "unit": "seconds",
            "range": "0.10971771336277314",
            "extra": {
              "number_of_processors": 1,
              "memory_mb": 39.16
            }
          },
          {
            "name": "test_eval[config_template_1] - Memory",
            "value": 39.16,
            "unit": "MB",
            "range": "0",
            "extra": {
              "number_of_processors": 1,
              "memory_mb": 39.16
            }
          },
          {
            "name": "test_eval[config_template_4] - Time",
            "value": 1.2435194292338565,
            "unit": "seconds",
            "range": "0.009293463224632708",
            "extra": {
              "number_of_processors": 4,
              "memory_mb": 163.54
            }
          },
          {
            "name": "test_eval[config_template_4] - Memory",
            "value": 163.54,
            "unit": "MB",
            "range": "0",
            "extra": {
              "number_of_processors": 4,
              "memory_mb": 163.54
            }
          }
        ]
      }
    ]
  }
}