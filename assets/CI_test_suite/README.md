# Test suite for continuous integration

By launching the [test suite](test_suite.py), users can verify PULP-TrainLib's primitives. 
To extend the test suite, please insert a new section in the Python suite, by following the structure of the other primitives.

The test suite is designed to create a `temp/` folder which contains all the tests that have been executed. In each test, the output is contained into its respective `log.txt` file, which is filled with the terminal's output. A summary of the execution of each test is then stored into `test_suite_results.txt`. Check for the expression `CONTAINS ERRORS` to check for tests which failed.