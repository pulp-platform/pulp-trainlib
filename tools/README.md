# Support tools for training library

# PULP-TrainLib's Deployer 

PULP-TrainLib is equipped with TrainLib_Deployer, a code-generation tool to deploy DNN training on PULP devices. TrainLib_Deployer generates a project folder with name and path provided by the user, containing:
- the Golden Model (GM) to validate the training;
- the C files to launch the training task on a PULP SoC (`net.c`, `net.h`, `stats.h`). 

The user's requirements can be set by editing the `"USER SETTINGS"` section of [TrainLib_Deployer](./TrainLib_Deployer/TrainLib_Deployer.py).

The graph of the DNN model to be deployed has to be provided manually. To do so, users need to edit the lists in the `NETWORK GRAPH` section. The list of available layers is provided on top of the tool. To insert a new layer, edit the `layer_list` and all the following lists. The sizes and properties of each layer have to inserted in column - i.e. at the same index of each list. Be careful to provide the DNN sizes as a set of lists of the same lengths, and to match the input and the output sizes of each layer.
To add a Residual Connection, insert a layer called 'Skipnode' after the layer you want to take the output from, and insert a layer called 'Sumnode' where you want to compute the sum.
To add a different type of layer after a skipnode derivation is taken, simply substitute 'Skipnode' with any kind of supperted layer, and modify the lists containing the layer's informations (hin, win, cin, etc..) as you would for the selected layer.
E.g: if you have a Conv2D layer with a 3x3 kernel, 2 in channels, 4 output channels, 5x5 input size, followed by a Fully-Connected Layer with 36 inputs and 8 outputs, the input size of the Fully-Connected should have kernel sizes (hk, wk) equal to 1, as well as (hin, win). The channels, instead, need to be 36 in the Fully-Connected input and 8 in output. 

In order to select how data is stored, the variable USE_DMA can be modified, the supported modes are:
- 'NO', to load all  structures and data in L1 
- 'SB', to load only structures in L1 and keep data in L2 while using Single Buffer mode for data manipulation in L1

The structure of TrainLib_Deployer is:

- `TrainLib_Deployer.py`: main file, containing the call to the main functions
- `DNN_Composer.py`: this file contains all of the functions to take the tool-specific graph definition of the DNN and create the test folder for the user
- `DNN_Reader.py`: this file contains the functions to translate a given graph specification (e.g. in ONNX format) to TrainLib_Deployer's format. 
- `deployment_utils.py`: this file contains all of the functions to write the files, prepare the folders, etc. If you implement new backend functions for PULP, please modify the fields of this file accordingly.
- `GM_templates.py`: this file contains the templates to create the DNN model inside the Golden Model.
- `net_templates.py`: this file contains the templates to create the DNN model in PULP.



# Autotuner

PULP-TrainLib's Autotuner finds the fastest (TileShape/MM) setup for a given layer and training step specified by the user. In case of layers bigger than the available L1 memory, the AutoTuner finds the most suitable tile shapes to fit L1.

## Local execution

The AutoTuner launches a call to a tiler (either naive or user-provided), which finds a list of tiles, sorted by memory occupation (the higher, the earlier the solution is sorted). Then, AutoTuner compiles and profiles the first `NUM_TILING_SOLUTIONS` layer tile sizes over a list of `NUM_STD_MATMUL` (`NUM_DW_MATMUL` in case of a Depthwise layer). The results of this exhaustive search are stored into `fastest_tiling.txt`.

To launch the AutoTuner, launch (from `tools/` folder):

```
python ./AutoTuner/autotuner.py
```

Please make sure of setting up the layer options under the `"USER SETTING"` section of the AutoTuner before launching the program.


## Execution on server or other computer

AutoTuner can generate scripts to execute in multi-threading mode. The files to run in this mode are located under the `server_execution_files/` folder. To run in multi-threading, you need to copy in the same folder of `run_regression.sh` the following files:
- basic.yml
- treegen.py

Before launching the multi-threaded execution, please make sure to specify the correct location of the `pulp-trainlib/` repository folder inside `autotuner.py`.

When you are ready, first run:

```
python treegen.py
```

Then, launch:

```
source run_regression.sh
```

Please note that, while executing on a remote server or third device, the AutoTuner will simulate all the tile sizes and matmul optimizations, but will not parse results. This has to be done manually. The results can be found inside the `tests/` folder of the layer you want to optimize, inside `runs.txt`.



# Memory Footprint Tool 

To verify the memory occupation of the current implementation of the 
layers inside the training library, launch from here:

```
python ./memory_footprint_tool/memory_footprint_eval.py
```

If you need to set your own sizes for the layers, modify the parameters
inside the file itself, inside the `"USER SETTINGS"` section of the code.

The results are stored into `memreport.txt`.
