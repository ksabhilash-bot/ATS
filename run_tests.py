import sys
import subprocess


def main() -> int:
    return subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
