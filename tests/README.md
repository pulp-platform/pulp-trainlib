# Test layers

## Structure of the code and defaults

The tests are structured in the following way:

- the reference data (aka Golden Model) is generated by `utils/GM.py`, in each test folder
- inside each test, the main code for the training steps is contained into `net.c` (execution model)

Tests folder names ending without type specification refer to FP32 data, except for `test_matmul`, in which the data type is internally selected (in the `Makefile`).

In each test, the tensor sizes are specified inside the `Makefile`. Each Makefile has its own section for user mods (check the corresponding comments).

## Running the tests

To run the tests inside this folder, after sourcing `pulp-sdk/configs/pulp_open.sh`, launch:

```
make clean get_golden all run STEP='XXX'
```

where `XXX` is the step to be run, between forward (FORWARD), input grads (BACKWARD_ERROR) and weight grads (BACKWARD_GRAD). This command has to be launched from inside the folder
of the desired test (for example, from `/tests/test_linear`). 

The valid arguments are:

- Test Linear (FP32), Test Conv2D (FP32): FORWARD, BACKWARD_GRAD, BACKWARD_ERROR
- Test for Pointwise and Depthwise (FP32): DW_FORWARD, DW_BACKWARD_GRAD, DW_BACKWARD_ERROR, PW_FORWARD, PW_BACKWARD_GRAD, PW_BACKWARD_ERROR

Also, for `test_matmul`, it is possible to select the data type of the matmuls, by specifying the parameter `DATA_TYPE='XXX'` (where XXX can be one between {float, fp16}) in the make command.

If the number of cores (NUM_CORES) is changed, make sure of manually deleting the `BUILD/` folder before running the test with the new `NUM_CORES` to avoid performance or behavioural issues.
This applies also to other flags, like `DEBUG/`.

A good command to compile your tests could be:

```
rm -rf BUILD/ ; make clean get_golden all run STEP='XXX' | tee log.txt
```

## Useful mods

When compiling (`make clean get_golden all run ...`), some flags or arguments can be inserted to modify the execution (can also be modified inside the Makefile, inside the `"User Settings"` section):

- DEBUG : activates printfs to show debug features of the code
- NUM_CORES : select the number of cluster cores on which to parallelize the primitives

For other mods to the execution, please watch inside the `Makefile` of each test. Common useful parameters are inside the `"User settings"` section.

Also, in each test it can be possibile to modify the error sensitivity while checking tensors, by modifying the value of `CHECK_TOLERANCE` and `ERROR_TOLERANCE`, inside `net.h`.

Other mods to network sizes (and more) can be set by modifying the defaults inside `utils/GM.py`, which generates the golden model in each test.

## Matmul profiling

`test_matmul` is a special test to profile the implemented matmuls on any matrix size. If you need to know which matmul performs better on given matrix sizes, run:

```
make clean get_golden all run IN_CH='size1' MID_CH='size2' OUT_CH='size3' > log.txt
make clean profile_fastest
```

`size1`, `size2`, `size3` are user-defined integers (replace them with numbers of your choice). 
See `Makefile`'s "User settings" section for more info. The sorted performances are written inside the `fastest_matmul.txt` file.

## Running matmul optimizations

The baseline code for each test can be run with the commented flag: 

```
#APP_CFLAGS += -DOPTIMIZE
```

This happens regardless of `MATMUL_TYPE`. If you want to run the code with optimizations (i.e. run a specific matmul algorithm instead of the baseline's one),
you need to uncomment the `-DOPTIMIZE` flag:

```
APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=num_matmul
```

`num_matmul` is an integer number which refers to a specific optimized matmul algorithm. See the function `matmul_manager` inside `pulp_train_utils_fp32.c` or
`pulp_train_utils_fp16.c`, depending on the desired data format. 

## Running multiple simulations with different optimizations, varying sizes and matmul algorithms

To run concurrent simulations (to test a sequence of optimizations, using different matmuls), make sure to have:

```
APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=num_matmul
```

For a given data format, each matmul has a numeric code, contained into `tests/mm_manager_list.txt` (or `*_fpXX` for other data formats). You can run multiple simulations, with the command:

```
make profile_all_optim STEP='XXX' <other_options>
```

Other options could be added, like `IN_CH=...` or `OUT_CH=...`, where `...` is an integer (if you want to change the sizes of the matrices in a linear layer, via command line, for example). 
Each layer has arguments for changing the layer sizes, inside the "User setting" section of its `Makefile`. The sorted results, from the fastest to the slowest, are stored into `runs.txt` file.
Please refer to the "User settings" section of each `Makefile` to find suitable options.

Some other mods can be made by hacking `utils/profile_optimized.py`.

An example to run the evaluation of multiple matmul algorithms on the same layer (e.g. a linear) can be:

```
make profile_all_optim STEP='BACKWARD_GRAD' IN_CH=1024 OUT_CH=8 NUM_CORES=8
```

For this command, you don't need to call `rm -rf BUILD/`, since this is automatically done by the software.


## Running multiple simulations with variable sizes

If you need to launch multiple simulations with variable layer size, launch the command:

```
make profile_all_sizes STEP='XXX' <options>
```

To adjust the sequence of sizes to be launched, modify the arrays inside `utils/profile_sizes.py`, inside the desired test folder.
The sizes arrays are under the `"USER SETTINGS"` section of the code (on top of the file).

The output of the profiling is contained into `runs.txt`.

For this command, you don't need to call `rm -rf BUILD/`, since this is automatically done by the software.

## Adding new matmuls to the library

If a new matmul is added to the library, make sure to insert it in the correct function list inside the `mm_manager` function (inside `pulp_train_utils_fpxx.c`).
Also, make sure that the names and lists of all matmuls in `mm_manager_list.txt` match exactly with the lists inside `mm_manager` before performing multiple simulations using `profile_optimized.py` (or the AutoTuner)!



# PULP-TrainLib's Autotuner

PULP-TrainLib's Autotuner can be found inside `pulp-trainlib/tools/AutoTuner`. Please refer to the relative [README](../tools/README.md) for further info.

