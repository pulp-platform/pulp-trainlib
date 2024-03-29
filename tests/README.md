# Test layers

## Structure of the code and defaults

PULP-TrainLib's tests are organized as follows:

- to verify the results of the code under test, PyTorch data (the so-called Golden Model or GM) is generated by `utils/GM.py`. You can find a GM under each test folder. Each GM generates a set of .h files containing the reference data (stored in the L2 memory);
- the C code of each test is contained in `net.c`. `net.h` contains several useful definitions for each test. Furthemore, the `stats.h` file contains the macros to profile the execution of the C code. The `main.c` contains the code to launch the main task on the PULP cluster.

DNN layer tests are provided one for each data type. In case of tests related to a specific data format, the folder name ends with that specific format (e.g. `test_linear_fp32/`). Other tests may feature multiple data types. To verify this, look inside the `Makefile` and `net.c`, as well as `net.h`.

All the configuration variables in each test are specified inside the `Makefile`. Each Makefile has its own section for user mods (check the corresponding comments). To select the sizes you want to check, change these definitions as you wish.

## Running the tests

To run a test, open you terminal and first source `pulp-sdk/configs/pulp_open.sh`. Then, move to the folder of your test (e.g. `/tests/test_linear_fp32`) and, from the same terminal, launch:

```
make clean all run
```

If the test you launch has a Golden Model, launch:

```
make clean get_golden all run
```

In case of tests of layers, which feature multiple training steps, launch instead:

```
make clean get_golden all run STEP='XXX'
```

where `XXX` is the step to be run. 

The valid arguments are:

- `test_linear_fpXX/`, `test_conv2d_fpXX/`: FORWARD, BACKWARD_GRAD, BACKWARD_ERROR
- `test_conv_pw_dw_fpXX/`: DW_FORWARD, DW_BACKWARD_GRAD, DW_BACKWARD_ERROR, PW_FORWARD, PW_BACKWARD_GRAD, PW_BACKWARD_ERROR

You can see the valid arguments inside each user section of the `Makefile`. For example, certain tests, as for `test_matmul`, give the possibility to select the data type of the executed code. In this case, the parameter `DATA_TYPE='XXX'` (where XXX can be one between {float, fp16}) can be set by the user. 

If the number of cores (NUM_CORES) is changed, make sure of manually deleting the `BUILD/` folder before running the test with the new `NUM_CORES` to avoid behavioural issues.
This applies also to other flags, like `DEBUG/`. For example, run:

```
rm -rf BUILD/ ; make clean get_golden all run STEP='XXX'
```

## Useful mods

When launching a test, some flags or arguments can be set to modify the execution. They can be set as argument of the terminal command or can also be modified inside the Makefile, inside the `"User Settings"` section. Useful examples are:

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

Each baseline version of a MM-based primitive (like Conv2D ones) can be launched by commenting the `OPTIMIZE` flag:

```
#APP_CFLAGS += -DOPTIMIZE
```

If you want to run the optimized version of the same primitive, you need to uncomment the `OPTIMIZE` flag and select the optimized MM you want:

```
APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=num_matmul
```

`num_matmul` is an integer number which refers to a specific optimized matmul algorithm. The MM selection is managed by the `matmul_manager` fucntion, defined in `pulp_train_utils_fp32.c` or `pulp_train_utils_fp16.c`, for each data format. You can see the code of each matmul inside the [mm_manager_list.txt](./mm_manager_list.txt) for fp32 and [mm_manager_list_fp16.txt](./mm_manager_list_fp16.txt) for fp16.

## Running multiple simulations with different optimizations, varying sizes and matmul algorithms

To evaluate the fastest MM for your problem (layer size, step), please to refer to `utils/profile_optimized.py` under the layer test folders. This script launches multiple simulations and finds the fastest setup. To launch this, make sure to have:

```
APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=num_matmul
```

To launch the script, other options could be added, like `IN_CH=...` or `OUT_CH=...`, where `...` is the size you desire. The sorted results, from the fastest to the slowest, are stored into `runs.txt` file.

An example to evaluate the fastest setup for a fully-connected layer (`test_linear_fp32/`) is:

```
make profile_all_optim STEP='BACKWARD_GRAD' IN_CH=1024 OUT_CH=8 NUM_CORES=8
```

## Running multiple simulations with variable sizes

If you need to launch multiple simulations with variable layer size, launch the command:

```
make profile_all_sizes STEP='XXX' <options>
```

To adjust the sequence of sizes to be launched, modify the arrays inside `utils/profile_sizes.py`, inside the desired test folder. The sizes arrays are under the `"USER SETTINGS"` section of the code (on top of the file).

The output of the profiling is contained into `runs.txt`.

## Adding new matmuls to the library

If a new matmul is added to the library, make sure to insert it in the correct function list inside the `mm_manager` function (inside `pulp_train_utils_fpxx.c`).
Also, make sure that the names and lists of all matmuls in `mm_manager_list.txt` match exactly with the lists inside `mm_manager` before performing multiple simulations using `profile_optimized.py` (or the AutoTuner)!

# PULP-TrainLib's Autotuner

PULP-TrainLib's Autotuner can be found inside `pulp-trainlib/tools/AutoTuner`. Please refer to the relative [README](../tools/README.md) for further info.

