"""
Simple test runner so you can run tests with:
  uv run --with pytest runtests.py -q

Pass any extra args after the filename and they'll be forwarded to pytest.
"""

import sys


def main():
    try:
        import pytest  # type: ignore
    except Exception:
        print("pytest is required. Run with: uv run --with pytest runtests.py")
        raise

    args = sys.argv[1:] or ["-q"]
    raise SystemExit(pytest.main(args))


if __name__ == "__main__":
    main()
