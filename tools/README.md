# Support tools for training library

# Autotuner

PULP-TrainLib's Autotuner finds the fastest (TileShape/MM) couple for a given layer and training step.

## Local execution

The AutoTuner launches a call to a tiler (either naive or based on DORY), which finds many a list of tiles, sorted by memory occupation (the higher, the earlier the solution is sorted).
After that, AutoTuner compiles and profiles the first `NUM_TILING_SOLUTIONS` layer tile sizes over a list of `NUM_STD_MATMUL` (`NUM_DW_MATMUL` in case of a Depthwise layer). The results of this exhaustive search are stored into `fastest_tiling.txt`.

To call the AutoTuner, simply launch (FROM THE `tools/` FOLDER)

```
python ./AutoTuner/autotuner.py
```

Please make sure of setting up the layer options inside the first part of this file, under the `"USER SETTING"` section of the code.


## Execution on server or other computer

The AutoTuner can generate scripts for the execution on external computers (like servers). The files to 
launch on multiple threads can be found inside `server_execution_files`. To run the threads
on another computer, you need to copy in the same folder of `run_regression.sh` the following files:
- basic.yml
- treegen.py

Before generating the files, please make sure to specify the correct location of `pulp-trainlib` folder
inside `autotuner.py`

To launch multiple parallel threads, first run:

```
python3.6 treegen.py
```

Then, launch:

```
source run_regression.sh
```

Please note that, while executing on a remote server or third device, the AutoTuner only simulates all
the tile sizes and matmul optimizations, but does not parse results. This has to be done manually.
The results can be found inside the `tests/` folder of the layer you want to optimize, inside `runs.txt`.



# Verify the memory occupation of a layer

To verify the memory occupation of the current implementation of the 
layers inside the training library, launch from here:

```
python ./memory_footprint_tool/memory_footprint_eval.py
```

If you need to set your own sizes for the layers, modify the parameters
inside the file itself, inside the `"USER SETTINGS"` section of the code.

The results are stored into `memreport.txt`.
