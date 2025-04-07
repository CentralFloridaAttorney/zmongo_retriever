import json
import time
from datetime import datetime
from pathlib import Path

def summarize_test_results(results: dict, output_path: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "timestamp": timestamp,
        "total_tests": results.get("total", 0),
        "failures": results.get("failures", 0),
        "errors": results.get("errors", 0),
        "skipped": results.get("skipped", 0),
        "passed": results.get("passed", 0),
        "duration_seconds": results.get("duration", 0),
        "modules": results.get("modules", []),
    }

    output_path.write_text(json.dumps(summary, indent=4))
    return summary

# Simulated input (can be adapted to parse from pytest/unittest or custom runner)
simulated_results = {
    "total": 92,
    "passed": 92,
    "failures": 0,
    "errors": 0,
    "skipped": 0,
    "duration": 19.04,
    "modules": [
        "test_zmongo_comparative_benchmarks",
        "test_real_db_comparative_benchmarks",
        "test_get_oid_value",
        "test_zmongo",
    ]
}

summary_path = Path.cwd() / "zmongo_test_summary.json"
summarize_test_results(simulated_results, summary_path)
