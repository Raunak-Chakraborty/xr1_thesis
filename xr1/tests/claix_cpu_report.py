#!/usr/bin/env python3
"""Run an existing harness unchanged, recording every unittest result."""
import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    suite = Path(args.suite).resolve(strict=True)
    destination = Path(args.result)
    if destination.exists():
        raise FileExistsError(destination)
    tests = []

    class Result(unittest.TextTestResult):
        def addSuccess(self, test):
            tests.append(dict(id=test.id(), status='PASS'))
            super().addSuccess(test)
        def addError(self, test, err):
            tests.append(dict(id=test.id(), status='ERROR'))
            super().addError(test, err)
        def addFailure(self, test, err):
            tests.append(dict(id=test.id(), status='FAIL'))
            super().addFailure(test, err)
        def addSkip(self, test, reason):
            tests.append(dict(id=test.id(), status='SKIP', reason=reason))
            super().addSkip(test, reason)
        def addExpectedFailure(self, test, err):
            tests.append(dict(id=test.id(), status='XFAIL'))
            super().addExpectedFailure(test, err)
        def addUnexpectedSuccess(self, test):
            tests.append(dict(id=test.id(), status='XPASS'))
            super().addUnexpectedSuccess(test)
        def addSubTest(self, test, subtest, err):
            if err is not None:
                tests.append(dict(id=subtest.id(), status='FAIL'))
            super().addSubTest(test, subtest, err)

    unittest.TextTestRunner.resultclass = Result
    code = 1
    error = None
    try:
        sys.path.insert(0, str(suite.parent))
        sys.argv = [str(suite)]  # no stub flags accepted
        runpy.run_path(str(suite), run_name='__main__')
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        with destination.open('x') as stream:
            json.dump(dict(returncode=code, tests=tests, count=len(tests), error=error,
                           counts=dict(Counter(row['status'] for row in tests))), stream, indent=2)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
