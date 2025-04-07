import os
import argparse


def find_large_files(start_path: str = ".", size_mb: float = 50.0):
    size_bytes = size_mb * 1024 * 1024
    print(f"\n🔍 Scanning for files larger than {size_mb} MB...\n")

    for root, dirs, files in os.walk(start_path):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                if file_size > size_bytes:
                    print(f"📦 {file_path} — {file_size / (1024 * 1024):.2f} MB")
            except Exception as e:
                print(f"⚠️  Could not access {file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find files larger than X MB.")
    parser.add_argument(
        "--path", type=str, default=".",
        help="Directory path to scan (default: current directory)"
    )
    parser.add_argument(
        "--size", type=float, default=50.0,
        help="Size threshold in MB (default: 50.0)"
    )
    args = parser.parse_args()

    find_large_files(args.path, args.size)
