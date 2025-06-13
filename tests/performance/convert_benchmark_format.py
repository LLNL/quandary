#!/usr/bin/env python3
"""
Convert pytest-benchmark JSON format to customSmallerIsBetter format
with separate entries for time and memory metrics.
"""

import argparse
import json
import sys
from typing import Dict, List, Any


def convert_to_custom_format(pytest_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert pytest-benchmark JSON data to customSmallerIsBetter format.
    Creates separate entries for time and memory metrics.
    """
    custom_benchmarks = []

    for benchmark in pytest_data.get("benchmarks", []):
        name = benchmark.get("name", "")

        # Time benchmark entry
        time_value = benchmark.get("stats", {}).get("median", 0)
        time_stddev = benchmark.get("stats", {}).get("stddev", 0)
        custom_benchmarks.append({
            "name": f"{name} - Time",
            "value": time_value,
            "unit": "seconds",
            "range": str(time_stddev),
        })

        # Memory benchmark entry (from extra_info)
        memory_mb = benchmark.get("extra_info", {}).get("memory_mb", 0)
        custom_benchmarks.append({
            "name": f"{name} - Memory",
            "value": memory_mb,
            "unit": "MB",
        })

    return custom_benchmarks


def main():
    parser = argparse.ArgumentParser(
        description="Convert pytest-benchmark JSON to customSmallerIsBetter format"
    )
    parser.add_argument("input_file", help="Input JSON file (pytest-benchmark format)")
    parser.add_argument("output_file", help="Output JSON file (customSmallerIsBetter format)")

    args = parser.parse_args()

    try:
        with open(args.input_file, "r") as f:
            pytest_data = json.load(f)

        custom_data = convert_to_custom_format(pytest_data)

        with open(args.output_file, "w") as f:
            json.dump(custom_data, f, indent=2)

        print(f"Successfully converted {args.input_file} to {args.output_file}")

    except Exception as e:
        print(f"Error converting benchmark format: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
