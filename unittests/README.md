# Unit tests

This folder contains all unit tests

## Setup

```
gem install ceedling
```

To support code coverage reports:

```
pip install gcovr
```

## Run

Ceedling takes care of building, running and test coverage report generation.

NOTE: all cmds below must be executed from the root directory of this git repository.

Running all tests:

```
ceedling test:all
```

Running all tests with code coverage

```
ceedling gcov:all
```

HTML report can be found at `build/artifacts/gcov/gcovr/GcovCoverageResults.html`.

Running one specific test module or test case:

```
ceedling test:test_pulp_matmul_fp32
ceedling test:test_pulp_matmul_fp32 --test-case=test_pulp_matmul_fp32_mm_M_u2_transp
```

## Debug

You can debug a test with gdb in a specific module with:

```
ceedling test:test_pulp_matmul_fp32
gdb --tui --args build/test/out/test_pulp_matmul_fp32/test_pulp_matmul_fp32.out
```

Once in the debugger set for example a breakpoint  (`b test_pulp_matmul_fp32_mm_M_u2_transp`) and hit `r`.
