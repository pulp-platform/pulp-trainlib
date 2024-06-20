'''
Copyright (C) 2021-2024 ETH Zurich and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Authors: Davide Nadalini, Cristian Cioflan, Axel Vanoni
'''

"""
TrainLib Deployer: a tool to deploy DNN on-device training on MCUs

Available DNN layer names:
'linear'    -> fully-connected layer
'conv2d'    -> 2d convolution layer
'PW'        -> pointwise convolution
'DW'        -> depthwise convolution
'ReLU'      -> ReLU activation
'MaxPool'   -> max pooling layer
'AvgPool'   -> average pooling layer
'Skipnode'  -> node at which data is taken and passes forward, to add an additional layer after the skip derivation simply substitute 'Skipnode' with any kind of layer
'Sumnode'   -> node at which data from Skipnode is summed 
'InstNorm'  -> instance Normalization layer

Available losses:
'MSELoss'           -> Mean Square Error loss
'CrossEntropyLoss'  -> CrossEntropy loss

Available optimizers:
'SGD'       -> Stochastic Gradient Descent
"""

import deployer_utils.DNN_Reader     as reader
import deployer_utils.DNN_Composer   as composer

# ---------------------
# --- USER SETTINGS ---
# ---------------------

parser = argparse.ArgumentParser(
                    prog='Deployer',
                    description='Generating C code for on-device training')

parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--project_name', type=str, default="Test_CNN", help='Project name', required=True)
required.add_argument('--project_path', type=str, default="./", help='Project path', required=True)
optional.add_argument('--model_path', type=str, default=None, help='Pretrained model path')
optional.add_argument('--start_at', type=str, default=None, help='At which node to start generating')
args = parser.parse_args()


# GENERAL PROPERTIES
project_name    = args.project_name
project_path    = args.project_path
proj_folder     = project_path + project_name + '/'

# TRAINING PROPERTIES
epochs          = 5
batch_size      = 1                   # BATCHING NOT IMPLEMENTED!!
learning_rate   = 0.001
optimizer       = "SGD"                # Name of PyTorch's optimizer
loss_fn         = "MSELoss"            # Name of PyTorch's loss function

# ------- NETWORK GRAPH --------
# Manually define the list of the network (each layer in the list has its own properties in the relative index of each list)
layer_list          = [ 'conv2d', 'ReLU', 'DW', 'PW', 'ReLU', 'DW', 'PW', 'ReLU', 'DW', 'PW', 'ReLU', 'linear']
# Layer properties
sumnode_connections = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ]            # For Skipnode and Sumnode only, for each Skipnode-Sumnode couple choose a value and assign it to both, all other layer MUST HAVE 0

in_ch_list          = [ 3,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  6*6*8 ]         # Linear: size of input vector
out_ch_list         = [ 8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  2 ]            # Linear: size of output vector
hk_list             = [ 3,  1,  3,  1,  1,  3,  1,  1,  7,  1,  1,  1 ]            # Linear: = 1
wk_list             = [ 3,  1,  3,  1,  1,  3,  1,  1,  7,  1,  1,  1 ]            # Linear: = 1
# Input activations' properties
hin_list            = [ 32, 16, 16, 14, 14, 14, 12, 12, 12, 6,  6, 1 ]            # Linear: = 1
win_list            = [ 32, 16, 16, 14, 14, 14, 12, 12, 12, 6,  6, 1 ]            # Linear: = 1
# Convolutional strides
h_str_list          = [ 2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ]            # Only for conv2d, maxpool, avgpool 
w_str_list          = [ 2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ]            # Only for conv2d, maxpool, avgpool 
# Padding (bilateral, adds the specified padding to both image sides)
h_pad_list          = [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]                            # Implemented for conv2d (naive kernel), DW TO DO
w_pad_list          = [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]                            # Implemented for conv2d (naive kernel), DW TO DO
# Define the lists to call the optimized matmuls for each layer (see mm_manager_list.txt, mm_manager_list_fp16.txt or mm_manager function body)
opt_mm_fw_list      = [ 10, 0, 0, 12, 0, 0, 12, 0, 0, 12, 0, 10 ]
opt_mm_wg_list      = [ 10, 0, 0, 12, 0, 0, 12, 0, 0, 12, 0, 10 ]
opt_mm_ig_list      = [ 10, 0, 0, 12, 0, 0, 12, 0, 0, 12, 0, 10 ]
# Data type list for layer-by-layer deployment (mixed precision)
#data_type_list      = ['FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16']
data_type_list     = ['FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32']
# Placeholder for pretrained parameters
data_list           = []
# Data layout list (CHW or HWC) 
data_layout_list    = ['CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW']   # TO DO
# Sparse Update
update_layer_list   = [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ]             # Set to 1 for each layer you want to update, 0 if you want to skip weight update
# ----- END OF NETWORK GRAPH -----




# EXECUTION PROPERTIES
NUM_CORES       = 8
L1_SIZE_BYTES   = 128*(2**10)
USE_DMA = 'DB'                          # choose whether to load all structures in L1 ('NO') or in L2 and use Single Buffer mode ('SB') or Double Buffer mode ('DB') 
# BACKWARD SETTINGS
SEPARATE_BACKWARD_STEPS = True          # If True, writes separate weight and input gradient in backward step
# PROFILING OPTIONS
PROFILE_SINGLE_LAYERS = False           # If True, profiles forward and backward layer-by-layer
# CONV2D SETUPS
CONV2D_USE_IM2COL = False                # Choose if the Conv2D layers should use Im2Col or not
# PRINT TRAIN LOSS
PRINT_TRAIN_LOSS = True                 # Set to true if you want to print the train loss for each epoch
# OTHER PROPERTIES
# Select if to read the network from an external source
READ_MODEL_ARCH = False                # NOT IMPLEMENTED!!

# ---------------------------
# --- END OF USER SETTING ---
# ---------------------------




"""
BACKEND
"""

class ONNXGraphParser:
    def __init__(self, onnx_model):
        onnx.checker.check_model(onnx_model)
        self.model = shape_inference.infer_shapes(onnx_model)
        self.graph = self.model.graph
        self.value_info_lookup = {v.name: i for i, v in enumerate(self.graph.value_info)}
        self.node_lookup = {v.name: i for i, v in enumerate(self.graph.node)}
        self.input_lookup = {v.name: i for i, v in enumerate(self.graph.input)}
        self.output_lookup = {v.name: i for i, v in enumerate(self.graph.output)}
        self.init_lookup = {v.name: i for i, v in enumerate(self.graph.initializer)}
        self.get_precision() # make sure precision is fine

    def _get_type(self, node_name):
        if node_name in self.value_info_lookup:
            return self.graph.value_info[self.value_info_lookup[node_name]].type
        elif node_name in self.input_lookup:
            return self.graph.input[self.input_lookup[node_name]].type
        elif node_name in self.output_lookup:
            return self.graph.output[self.output_lookup[node_name]].type
        else:
            raise KeyError(f"Node {node_name} not found")

    def _get_node_attr(self, node_name, attr):
        for a in self.graph.node[self.node_lookup[node_name]].attribute:
            if a.name == attr:
                return a
        else:
            raise ValueError(f"Node {node_name} has no {attr} attribute")

    def is_pointwise(self, node_name):
        hk, wk = self.get_kernel_size(node_name)
        return hk == wk == 1

    def is_depthwise(self, node_name):
        node = self.graph.node[self.node_lookup[node_name]]
        try:
            groups = self._get_node_attr(node_name, "group").i
        except ValueError:
            return False
        in_ch = self.get_channel_count(node.input[0])
        out_ch = self.get_channel_count(node.output[0])
        if groups <= 1:
            return False
        assert in_ch == out_ch == groups, "For depthwise convolutions, input and output channels must be the same as groups"
        return True

    def get_channel_count(self, node_name):
        tensor_type = self._get_type(node_name)
        # Data layout is B, C, H, W
        return tensor_type.tensor_type.shape.dim[1].dim_value

    def get_hw(self, node_name):
        tensor_type = self._get_type(node_name)
        shape = tensor_type.tensor_type.shape.dim
        return shape[2].dim_value, shape[3].dim_value

    def get_activation_size(self, node_name):
        tensor_type = self._get_type(node_name)
        dims = tensor_type.tensor_type.shape.dim
        # Data layout is B, C, H, W
        return dims[2].dim_value, dims[3].dim_value

    def get_init(self, node_name):
        index = self.init_lookup.get(node_name, -1)
        if index == -1:
            raise KeyError(f"Node {node_name} has no initializer")
        init = self.graph.initializer[index]
        return numpy_helper.to_array(init)

    def get_kernel_size(self, node_name):
        ksize = self._get_node_attr(node_name, "kernel_shape")
        return ksize.ints[0], ksize.ints[1]

    def get_stride(self, node_name):
        stride = self._get_node_attr(node_name, "strides")
        return stride.ints[0], stride.ints[1]

    def get_pad(self, node_name):
        pad = self._get_node_attr(node_name, "pads").ints
        assert pad[0] == pad[2] and pad[1] == pad[3], "Only symmetric padding is supported."
        return pad[0], pad[1]

    def get_precision(self):
        elem_type = self.graph.value_info[0].type.tensor_type.elem_type
        if elem_type == onnx.TensorProto.FLOAT:
            return "FP32"
        elif elem_type == onnx.TensorProto.FLOAT16:
            return "FP16"
        elif elem_type == onnx.TensorProto.BFLOAT16:
            raise NotImplementedError("Numpy does not support bfloat16 and converts it to FP32. We need to change how we save and load weights.")
        else:
            raise ValueError("Only FP32 and FP16 are supported")


# Call the DNN Reader and then the DNN Composer 
if READ_MODEL_ARCH :


    layer_list = []
    in_ch_list = []
    out_ch_list = []
    hk_list = []
    wk_list = []
    hin_list = []
    win_list = []
    h_pad_list = []
    w_pad_list = []
    opt_mm_fw_list = []
    opt_mm_wg_list = []
    opt_mm_ig_list = []
    data_type_list = []
    data_layout_list = []
    
    if (args.model_path.split('.')[-1] == "onnx"):
        onnx_model = onnx.load(args.model_path)
        onnx.checker.check_model(onnx_model)
        graph = ONNXGraphParser(onnx_model)
        found_start = args.start_at is None

        if args.start_at is not None:
            node_names = [n.name for n in graph.graph.node if n.op_type != 'Constant']
            assert args.start_at in node_names, f"{args.start_at} is not a valid layer name. Layer names are: {node_names}"

        for onnx_node in graph.graph.node:
            if not found_start:
                if onnx_node.name != args.start_at:
                    continue
                else:
                    found_start = True

            if (onnx_node.op_type == 'Gemm') or (onnx_node.op_type == 'MatMul'):
                in_ch_list.append(graph.get_channel_count(onnx_node.input[0]))
                out_ch_list.append(graph.get_channel_count(onnx_node.output[0]))
                layer_list.append('linear')
                hk_list.append(1)
                wk_list.append(1)
                hin_list.append(1)
                win_list.append(1)
                h_str_list.append(1)
                w_str_list.append(1)
                h_pad_list.append(0)
                w_pad_list.append(0)
                opt_mm_fw_list.append(0)
                opt_mm_wg_list.append(0)
                opt_mm_ig_list.append(0)
                # TODO: Read from file
                data_type_list.append(graph.get_precision())
                # TODO: Read from file
                # Note that this also determines the read position for in_ch_list and out_ch_list
                data_layout_list.append('CHW')
                weight_init = graph.get_init(onnx_node.input[1])
                # Gemm node does y = x*B, but torch uses y = A*x, so transpose B to get A back
                # This also aligns with how trainlib does things
                weight_init = weight_init.transpose(1,0)
                try:
                    bias_init = graph.get_init(onnx_node.input[2])
                    raise NotImplementedError("Biases are not implemented in trainlib")
                except (KeyError, IndexError):
                    bias_init = []
                data_list.append((weight_init, bias_init))
                sumnode_connections.append(0)
            elif onnx_node.op_type == 'AveragePool':
                in_ch_list.append(graph.get_channel_count(onnx_node.input[0]))
                out_ch_list.append(graph.get_channel_count(onnx_node.output[0]))
                layer_list.append('AvgPool')
                (hk, wk) = graph.get_kernel_size(onnx_node.name)
                hk_list.append(hk)
                wk_list.append(wk)
                (hin, win) = graph.get_activation_size(onnx_node.input[0])
                hin_list.append(hin)
                win_list.append(win)
                (hstr, wstr) = graph.get_stride(onnx_node.name)
                h_str_list.append(hstr)
                w_str_list.append(wstr)
                (hpad, wpad) = graph.get_pad(onnx_node.name)
                h_pad_list.append(hpad)
                w_pad_list.append(wpad)
                opt_mm_fw_list.append(0)
                opt_mm_wg_list.append(0)
                opt_mm_ig_list.append(0)
                data_type_list.append(graph.get_precision())
                data_layout_list.append('CHW')
                data_list.append(([], [])) # kernels
                sumnode_connections.append(0)
            elif onnx_node.op_type == 'GlobalAveragePool':
                hk, wk = graph.get_hw(onnx_node.input[0])
                if hk == 1 and wk == 1:
                    # There is nothing to average, skip this node
                    continue
                in_ch_list.append(graph.get_channel_count(onnx_node.input[0]))
                out_ch_list.append(graph.get_channel_count(onnx_node.output[0]))
                layer_list.append('AvgPool')
                hk_list.append(hk)
                wk_list.append(wk)
                (hin, win) = graph.get_activation_size(onnx_node.input[0])
                hin_list.append(hin)
                win_list.append(win)
                h_str_list.append(1)
                w_str_list.append(1)
                h_pad_list.append(0)
                w_pad_list.append(0)
                opt_mm_fw_list.append(0)
                opt_mm_wg_list.append(0)
                opt_mm_ig_list.append(0)
                data_type_list.append(graph.get_precision())
                data_layout_list.append('CHW')
                data_list.append(([], [])) # kernels
                sumnode_connections.append(0)
            elif onnx_node.op_type == 'Conv':
                in_ch_list.append(graph.get_channel_count(onnx_node.input[0]))
                out_ch_list.append(graph.get_channel_count(onnx_node.output[0]))
                if graph.is_pointwise(onnx_node.name):
                    ty = "PW"
                elif graph.is_depthwise(onnx_node.name):
                    ty = "DW"
                else:
                    ty = "conv2d"
                layer_list.append(ty)
                (hk, wk) = graph.get_kernel_size(onnx_node.name)
                hk_list.append(hk)
                wk_list.append(wk)
                (hin, win) = graph.get_activation_size(onnx_node.input[0])
                hin_list.append(hin)
                win_list.append(win)
                (hstr, wstr) = graph.get_stride(onnx_node.name)
                h_str_list.append(hstr)
                w_str_list.append(wstr)
                (hpad, wpad) = graph.get_pad(onnx_node.name)
                h_pad_list.append(hpad)
                w_pad_list.append(wpad)
                opt_mm_fw_list.append(0)
                opt_mm_wg_list.append(0)
                opt_mm_ig_list.append(0)
                data_type_list.append(graph.get_precision())
                # TODO: Read from file
                # Note that this also determines the read position for in_ch_list and out_ch_list
                data_layout_list.append('CHW')
                weight_init = graph.get_init(onnx_node.input[1])
                try:
                    bias_init = graph.get_init(onnx_node.input[2])
                    raise NotImplementedError("Biases are not implemented in trainlib")
                except (KeyError, IndexError):
                    # Ignore missing bias
                    bias_init = []
                    pass
                data_list.append((weight_init, bias_init)) # kernels
                sumnode_connections.append(0)
            elif onnx_node.op_type == 'Clip':
                # This does not handle ReLU6, as it is not supported by trainlib
                layer_list.append('ReLU')
                in_ch_list.append(graph.get_channel_count(onnx_node.input[0]))
                out_ch_list.append(graph.get_channel_count(onnx_node.output[0]))
                hk_list.append(1)
                wk_list.append(1)
                hin_list.append(1)
                win_list.append(1)
                h_str_list.append(1)
                w_str_list.append(1)
                h_pad_list.append(0)
                w_pad_list.append(0)
                opt_mm_fw_list.append(0)
                opt_mm_wg_list.append(0)
                opt_mm_ig_list.append(0)
                data_layout_list.append('CHW')
                data_type_list.append(graph.get_precision())
                data_list.append(([], []))
                sumnode_connections.append(0)
    else:
        raise NotImplementedError("Model format not supported.")

    data_dir = proj_folder+'data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for i, (weight_init, bias_init) in enumerate(data_list):
        np.save(data_dir+f"l{i}w.npy", np.array(data_list[i][0], dtype=("float32" if data_type_list[i] == "FP32" else "float16")))
        np.save(data_dir+f"l{i}b.npy", np.array(data_list[i][1], dtype=("float32" if data_type_list[i] == "FP32" else "float16")))

    print("Generating project at location "+proj_folder)

    # Check if Residual Connections are valid
    sumnode_connections = composer.AdjustResConnList(sumnode_connections)

    composer.CheckResConn(layer_list, in_ch_list, out_ch_list, hin_list, win_list, sumnode_connections, update_layer_list) 

    # Check if the network training fits L1
    memocc = composer.DNN_Size_Checker(layer_list, in_ch_list, out_ch_list, hk_list, wk_list, hin_list, win_list, 
                                h_str_list, w_str_list, h_pad_list, w_pad_list,
                                data_type_list, update_layer_list, L1_SIZE_BYTES, USE_DMA, CONV2D_USE_IM2COL)

    print("DNN memory occupation: {} bytes of {} available L1 bytes ({}%).".format(memocc, L1_SIZE_BYTES, (memocc/L1_SIZE_BYTES)*100))

    # Call DNN Composer on the user-provided graph
    composer.DNN_Composer(proj_folder, project_name, 
                            layer_list, in_ch_list, out_ch_list, hk_list, wk_list, 
                            hin_list, win_list, h_str_list, w_str_list, h_pad_list, w_pad_list,
                            epochs, batch_size, learning_rate, optimizer, loss_fn,
                            NUM_CORES, data_type_list, data_list, update_layer_list, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list, sumnode_connections,
                            USE_DMA, PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS, CONV2D_USE_IM2COL, PRINT_TRAIN_LOSS)

    print("PULP project generation successful!")


else:

    print("Generating project at location "+proj_folder)

    # Check if Residual Connections are valid
    
    sumnode_connections = composer.AdjustResConnList(sumnode_connections)

    composer.CheckResConn(layer_list, in_ch_list, out_ch_list, hin_list, win_list, sumnode_connections, update_layer_list) 

    # Check if the network training fits L1
    memocc = composer.DNN_Size_Checker(layer_list, in_ch_list, out_ch_list, hk_list, wk_list, hin_list, win_list, 
                                h_str_list, w_str_list, h_pad_list, w_pad_list,
                                data_type_list, update_layer_list, 
                                L1_SIZE_BYTES, USE_DMA, CONV2D_USE_IM2COL)

    print("DNN memory occupation: {} bytes of {} available L1 bytes ({}%).".format(memocc, L1_SIZE_BYTES, (memocc/L1_SIZE_BYTES)*100))

    # Call DNN Composer on the user-provided graph
    composer.DNN_Composer(proj_folder, project_name, 
                            layer_list, in_ch_list, out_ch_list, hk_list, wk_list, 
                            hin_list, win_list, h_str_list, w_str_list, h_pad_list, w_pad_list,
                            epochs, batch_size, learning_rate, optimizer, loss_fn,
                            NUM_CORES, data_type_list, data_list, update_layer_list, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list, 
                            sumnode_connections, USE_DMA, PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS, CONV2D_USE_IM2COL, PRINT_TRAIN_LOSS)

    print("PULP project generation successful!")

    pass
