"""Allow `python -m src.eval` to run the benchmark CLI directly.

Spec calls for `python -m src.eval.run_benchmark`; we keep both forms
working so existing tooling and the documented WP6 invocation behave the
same.
"""

import sys

from src.eval.run_benchmark import main

if __name__ == "__main__":
    sys.exit(main())
