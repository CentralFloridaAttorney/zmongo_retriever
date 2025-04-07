import os
import unittest
import time
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

def run_all_tests_and_summarize(test_dir: Path, output_summary_path: Path) -> dict:
    start_time = time.time()

    # Load all tests in the specified directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(test_dir))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2, resultclass=unittest.TextTestResult)
    result = runner.run(suite)

    # Flatten the suite and collect test IDs
    def extract_test_cases(s):
        for item in s:
            if isinstance(item, unittest.TestSuite):
                yield from extract_test_cases(item)
            else:
                yield item

    test_cases = list(extract_test_cases(suite))
    tested_modules = sorted({
        test.id().split('.')[0]
        for test in test_cases
        if test is not None and hasattr(test, "id") and callable(test.id)
    })

    # Calculate test statistics
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
        "modules": tested_modules
    }

    # Save to JSON
    output_summary_path.write_text(json.dumps(summary, indent=4))

    # Print a clean summary
    print("\n==== TEST SUMMARY ====")
    print(f"Total:   {total_tests}")
    print(f"Passed:  {passed}")
    print(f"Failures:{failures}")
    print(f"Errors:  {errors}")
    print(f"Skipped: {skipped}")
    print(f"Duration:{duration} seconds")
    print("Modules:", ", ".join(tested_modules))
    print("======================\n")

    return summary


test_directory = Path(os.environ.get("TEST_DIR", ""))
summary_output_file = test_directory / "zmongo_test_summary.json"

run_all_tests_and_summarize(test_directory, summary_output_file)
