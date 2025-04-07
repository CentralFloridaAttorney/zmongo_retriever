import unittest
import time
import json
import io
import re
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout


def parse_performance_metrics(output: str) -> dict:
    metrics = {}
    unparsed_lines = []

    def parse_number(num_str):
        return float(num_str.replace(",", ""))

    patterns = {
        "insert_documents": r'Insert (\d+) documents: ([\d.]+)s \(avg ([\d.]+)s per insert\)',
        "cached_lookups": r'([\d,]+) cached lookups: ([\d.]+)s \(avg ([\d.]+)s per lookup\)',
        "cold_cache": r'Cold cache lookup: ([\d.]+)s',
        "warm_cache": r'Warm cache lookup: ([\d.]+)s',
        "mongodb_insert": r'MongoDB insert (\d+) docs: ([\d.]+)s \(avg ([\d.]+) ms per insert\)',
        "redis_set": r'Redis SET (\d+) keys: ([\d.]+)s \(avg ([\d.]+) ms per insert\)',
        "redis_get_latency": r'Redis GET latency \(cached key\): ([\d.]+) ms',
        "mongo_cached_latency": r'MongoDB query latency \(cached document\): ([\d.]+) ms',
        "mongo_bulk_insert": r'MongoDB bulk insert \(.*?\): ([\d.,]+) ops/sec',
        "bulk_write_100k": r'Bulk write 100k ops throughput: ([\d.,]+) ops/sec',
        "bulk_write_large": r'Bulk write of ([\d,]+) ops: ([\d.]+)s',
        "avg_query_latency_cached": r'Average query latency \(cached\): ([\d.]+) ms',
        "concurrent_read_test": r'Concurrent read test \(([\d,]+) ops\): ([\d.]+)s total',
        "mongo_concurrent_reads": r'MongoDB ([\d,]+) concurrent reads \(threads\): ([\d.]+)s',
        "redis_concurrent_gets": r'Redis ([\d,]+) concurrent GETs \(threads\): ([\d.]+)s',
        "cache_hit_ratio": r'Cache hit ratio: ([\d.]+)%'
    }

    for line in output.strip().splitlines():
        line = line.strip()
        matched = False

        for key, pattern in patterns.items():
            match = re.match(pattern, line)
            if not match:
                continue
            groups = match.groups()

            if key == "insert_documents":
                metrics[key] = {
                    "count": int(groups[0]),
                    "total_seconds": float(groups[1]),
                    "avg_seconds_per_insert": float(groups[2])
                }
            elif key == "cached_lookups":
                metrics[key] = {
                    "count": int(groups[0].replace(",", "")),
                    "total_seconds": float(groups[1]),
                    "avg_seconds_per_lookup": float(groups[2])
                }
            elif key in {"cold_cache", "warm_cache"}:
                metrics[f"{key}_lookup_seconds"] = float(groups[0])
            elif key in {"mongodb_insert", "redis_set"}:
                metrics[key] = {
                    "count": int(groups[0]),
                    "total_seconds": float(groups[1]),
                    "avg_ms_per_insert": float(groups[2])
                }
            elif key in {"redis_get_latency", "mongo_cached_latency", "avg_query_latency_cached"}:
                metrics[f"{key}_ms"] = float(groups[0])
            elif key in {"mongo_bulk_insert", "bulk_write_100k"}:
                metrics[f"{key}_ops_sec"] = parse_number(groups[0])
            elif key == "bulk_write_large":
                metrics[key] = {
                    "count": int(groups[0].replace(",", "")),
                    "total_seconds": float(groups[1])
                }
            elif key in {"concurrent_read_test", "mongo_concurrent_reads", "redis_concurrent_gets"}:
                metrics[key] = {
                    "count": int(groups[0].replace(",", "")),
                    "total_seconds": float(groups[1])
                }
            elif key == "cache_hit_ratio":
                metrics[f"{key}_percent"] = float(groups[0])
            matched = True
            break

        if not matched:
            unparsed_lines.append(line)

    if unparsed_lines:
        metrics["unparsed_lines"] = unparsed_lines

    return metrics


def run_all_tests_and_summarize(test_dir: Path, output_summary_path: Path) -> dict:
    start_time = time.time()

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(test_dir))

    def extract_test_cases(suite_obj):
        for item in suite_obj:
            if isinstance(item, unittest.TestSuite):
                yield from extract_test_cases(item)
            else:
                yield item

    test_cases = list(extract_test_cases(suite))

    # Capture stdout from tests
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        runner = unittest.TextTestRunner(verbosity=2, resultclass=unittest.TextTestResult)
        result = runner.run(suite)

    captured_output = buffer.getvalue()
    performance_metrics = parse_performance_metrics(captured_output)

    tested_modules = sorted({
        test.id().split('.')[0]
        for test in test_cases
        if hasattr(test, 'id') and callable(test.id)
    })

    duration = round(time.time() - start_time, 2)
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": total_tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": passed,
        "duration_seconds": duration,
        "modules": tested_modules,
        "performance_metrics": performance_metrics,
        "captured_output": captured_output.strip()
    }

    output_summary_path.write_text(json.dumps(summary, indent=4, sort_keys=True))

    # Console summary
    print("\n==== TEST SUMMARY ====")
    print(f"Total:   {total_tests}")
    print(f"Passed:  {passed}")
    print(f"Failures:{failures}")
    print(f"Errors:  {errors}")
    print(f"Skipped: {skipped}")
    print(f"Duration:{duration} seconds")
    print("Modules:")
    for mod in tested_modules:
        print(f"  - {mod}")
    print("======================\n")

    if performance_metrics:
        print("==== PERFORMANCE METRICS ====")
        print(json.dumps(performance_metrics, indent=4, sort_keys=True))
        print("=============================\n")

    return summary


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    test_directory = base_dir
    summary_output_file = base_dir / "zmongo_test_summary.json"

    run_all_tests_and_summarize(test_directory, summary_output_file)
