"""Performance test configuration for pytest."""
import copy
import datetime
from test_utils.conftest_common import add_common_options


def pytest_addoption(parser):
    """Add command line options to pytest."""
    add_common_options(parser)


def pytest_benchmark_generate_json(config, benchmarks, include_data, machine_info, commit_info):
    """Generate JSON with added memory benchmarks."""
    result = {
        "machine_info": machine_info,
        "commit_info": commit_info,
        "benchmarks": [],
        "datetime": datetime.datetime.now().isoformat(),
        "version": "5.1.0"
    }

    # First add all original benchmarks
    for benchmark in benchmarks:
        if benchmark.has_error:
            continue

        # Add original benchmark data
        benchmark_data = benchmark.as_dict()
        # Rename to indicate time measurement
        benchmark_data["name"] = f"{benchmark.name}_time"
        benchmark_data["fullname"] = f"{benchmark.name}_time"
        result["benchmarks"].append(benchmark_data)

        # Create a duplicate benchmark for memory
        mem_benchmark = copy.deepcopy(benchmark_data)
        mem_benchmark["name"] = f"{benchmark.name}_memory"
        mem_benchmark["fullname"] = f"{benchmark.name}_memory"

        # Replace stats with memory value
        memory_mb = benchmark.extra_info.get("memory_mb", 0)
        mem_benchmark["stats"] = {
            "min": memory_mb,
            "max": memory_mb,
            "mean": memory_mb,
            "stddev": 0,
            "median": memory_mb,
            "iqr": 0,
            "rounds": 1,
            "iterations": 1,
            "q1": memory_mb,
            "q3": memory_mb,
            "iqr_outliers": 0,
            "stddev_outliers": 0,
            "outliers": "0;0",
            "total": memory_mb,
            "data": [memory_mb]
        }

        # Update units in options
        mem_benchmark["options"] = copy.deepcopy(benchmark_data["options"])
        mem_benchmark["options"]["timer"] = "memory_mb"

        result["benchmarks"].append(mem_benchmark)

    return result
