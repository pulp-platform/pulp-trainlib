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
python treegen.py
```

Then, launch:

```
source run_regression.sh
```

Please note that, while executing on a remote server or third device, the AutoTuner only simulates all
the tile sizes and matmul optimizations, but does not parse results. This has to be done manually.
The results can be found inside the `tests/` folder of the layer you want to optimize, inside `runs.txt`.



# PULP-TrainLib's Deployer 

PULP-TrainLib is equipped with a tool to deploy DNN training on PULP devices. This tool can generate a project folder with name and path provided by the user, which contains the Golden Model (GM) to validate the network on PULP, together with correctly setup files (`net.c`, `net.h`, `stats.h`) to be included in your final project.

For all the user settings, please refer to the `"USER SETTINGS"` section of `tools/TrainLib_Deployer/TrainLib_Deployer.py`.

To set up the network, provide a list of layers inside `TrainLinb_Deployer.py`. Be careful to write the DNN sizes as a set of lists of the same lengths, and to match the input and the output sizes of each layer.
E.g: if you have a Conv2D layer with a 3x3 kernel, 2 in channels, 4 output channels, 5x5 input size, followed by a Fully-Connected Layer with 36 inputs and 8 outputs, the input size of the Fully-Connected should have kernel sizes (hk, wk) equal to 1, as well as (hin, win). The channels, instead, need to be 36 in the Fully-Connected input and 8 in output. 

The tool is still work in progress. Its intended structure is the following:

- `TrainLib_Deployer.py`: main file, containing the call to the main functions
- `DNN_Composer.py`: this file contains all of the functions to take the tool-specific graph definition of the DNN and create the test folder for the user
- `DNN_Reader.py`: this file is still empty, but its intended use is to contain a set of function to transform the ONNX of a DNN to a list which is processable by the DNN_Composer.
- `deployment_utils.py`: biggest backend of the application, this contains all of the functions to write the files, prepare the folders, etc. If you implement new backend functions for PULP, please modify the fields of this file accordingly.
- `GM_templates.py`: this file contains the templates to create the DNN model inside the Golden Model.
- `net_templates.py`: this file contains the templates to create the DNN model in PULP.

Known issues:
- the training proccess still does not perfectly match the Golden Model. This is likely caused by non-perfect matching between PULP-TrainLib's and PyTorch optimizers.



# Verify the memory occupation of a layer

To verify the memory occupation of the current implementation of the 
layers inside the training library, launch from here:

```
python ./memory_footprint_tool/memory_footprint_eval.py
```

If you need to set your own sizes for the layers, modify the parameters
inside the file itself, inside the `"USER SETTINGS"` section of the code.

The results are stored into `memreport.txt`.
