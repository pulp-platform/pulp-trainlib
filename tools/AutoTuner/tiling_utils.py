'''
Copyright (C) 2021-2022 ETH Zurich and University of Bologna

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
Authors: Davide Nadalini
'''

import math
import os
from ortools.constraint_solver import pywrapcp

# Frontend function for the computation of the tiles
def get_tiling (DW,
                filter_size1,
                filter_size2,
                stride,
                padding_top,padding_bottom,padding_left,padding_right,
                groups,
                BN,
                in_channels,
                out_channels,
                x_shape,
                y_shape,
                buffer_size,
                multiple_buffering_factor=2,
                name='conv',
                BitIn=32,
                BitW=32,
                BitActivation=32,
                BitOut=32,
                NUM_RESULTS=10,
                layer_type='CONV2D',
                use_bias=1,
                NAIVE=True,
                NUM_CORES=8,
                IGNORE_IN_GRAD=False): 

    # Output arrays
    N_input = []
    N_output = []
    W_input = []
    W_output = []
    H_input = []
    H_output = []
    Obj_values = []
    NUM_FOUND_SOLUTIONS = 0

    # NAIVE TILER
    if NAIVE == True:

        if IGNORE_IN_GRAD == True:
            ignore_in_grads = True
        elif layer_type == 'CONV2D':
            ignore_in_grads = True
        else:
            ignore_in_grads = False

        N_input, N_output, H_input, H_output, W_input, W_output, Obj_values, NUM_FOUND_SOLUTIONS = steven_the_tiler(
                                                                            DW=DW,
                                                                            filter_size1=filter_size1,
                                                                            filter_size2=filter_size2,
                                                                            stride=stride,
                                                                            padding_top=padding_top,
                                                                            padding_bottom=padding_bottom,
                                                                            padding_left=padding_left,
                                                                            padding_right=padding_right,
                                                                            groups=groups,
                                                                            BN=BN,
                                                                            in_channels=in_channels,
                                                                            out_channels=out_channels,
                                                                            x_shape=x_shape,
                                                                            y_shape=y_shape,
                                                                            buffer_size=buffer_size,
                                                                            multiple_buffering_factor=multiple_buffering_factor,
                                                                            name=name,
                                                                            BitIn=BitIn,
                                                                            BitW=BitW,
                                                                            BitActivation=BitActivation,
                                                                            BitOut=BitOut,
                                                                            NUM_RESULTS=NUM_RESULTS,
                                                                            Solution_idx=0,
                                                                            layer_type=layer_type,
                                                                            use_bias=use_bias,
                                                                            ignore_in_grads=ignore_in_grads,
                                                                            NUM_CORES=NUM_CORES
                                                                            )        

        return (N_input, N_output, H_input, H_output, W_input, W_output, Obj_values, NUM_FOUND_SOLUTIONS)



    # DORY-BASED TILER
    else: 
        # TILER FOR LINEAR AND PW
        if DW==0 :
            # Find the number of solutions
            C_in, C_out, H_in, H_out, W_in, W_out, Obj, NUM_FOUND_SOLUTIONS = get_tiling_conv2d_like(
                                                                                DW=DW,
                                                                                filter_size1=filter_size1,
                                                                                filter_size2=filter_size2,
                                                                                stride=stride,
                                                                                padding_top=padding_top,
                                                                                padding_bottom=padding_bottom,
                                                                                padding_left=padding_left,
                                                                                padding_right=padding_right,
                                                                                groups=groups,
                                                                                BN=BN,
                                                                                in_channels=in_channels,
                                                                                out_channels=out_channels,
                                                                                x_shape=x_shape,
                                                                                y_shape=y_shape,
                                                                                buffer_size=buffer_size,
                                                                                multiple_buffering_factor=multiple_buffering_factor,
                                                                                name=name,
                                                                                BitIn=BitIn,
                                                                                BitW=BitW,
                                                                                BitActivation=BitActivation,
                                                                                BitOut=BitOut,
                                                                                NUM_RESULTS=NUM_RESULTS,
                                                                                Solution_idx=0
                                                                                )

            # Now the number of solutions is known, iterate and find all
            for sol_idx in range(NUM_FOUND_SOLUTIONS):
                C_in, C_out, H_in, H_out, W_in, W_out, Obj, NUM_FOUND_SOLUTIONS = get_tiling_conv2d_like(
                                                                                    DW=DW,
                                                                                    filter_size1=filter_size1,
                                                                                    filter_size2=filter_size2,
                                                                                    stride=stride,
                                                                                    padding_top=padding_top,
                                                                                    padding_bottom=padding_bottom,
                                                                                    padding_left=padding_left,
                                                                                    padding_right=padding_right,
                                                                                    groups=groups,
                                                                                    BN=BN,
                                                                                    in_channels=in_channels,
                                                                                    out_channels=out_channels,
                                                                                    x_shape=x_shape,
                                                                                    y_shape=y_shape,
                                                                                    buffer_size=buffer_size,
                                                                                    multiple_buffering_factor=multiple_buffering_factor,
                                                                                    name=name,
                                                                                    BitIn=BitIn,
                                                                                    BitW=BitW,
                                                                                    BitActivation=BitActivation,
                                                                                    BitOut=BitOut,
                                                                                    NUM_RESULTS=NUM_FOUND_SOLUTIONS,
                                                                                    Solution_idx=(NUM_FOUND_SOLUTIONS-sol_idx-1)
                                                                                    )    
                N_input.append(int(C_in))
                N_output.append(int(C_out))
                H_input.append(int(H_in))
                H_output.append(int(H_out))
                W_input.append(int(W_in))
                W_output.append(int(W_out))
                Obj_values.append(int(Obj))    
                

            return (N_input, N_output, H_input, H_output, W_input, W_output, Obj_values, NUM_FOUND_SOLUTIONS)

        # TILER FOR DW CONV
        if DW==1 :
            # Find the number of solutions
            C_in, C_out, H_in, H_out, W_in, W_out, Obj, NUM_FOUND_SOLUTIONS = get_tiling_dw_like(
                                                                                DW=DW,
                                                                                filter_size1=filter_size1,
                                                                                filter_size2=filter_size2,
                                                                                stride=stride,
                                                                                padding_top=padding_top,
                                                                                padding_bottom=padding_bottom,
                                                                                padding_left=padding_left,
                                                                                padding_right=padding_right,
                                                                                groups=groups,
                                                                                BN=BN,
                                                                                in_channels=in_channels,
                                                                                out_channels=out_channels,
                                                                                x_shape=x_shape,
                                                                                y_shape=y_shape,
                                                                                buffer_size=buffer_size,
                                                                                multiple_buffering_factor=multiple_buffering_factor,
                                                                                name=name,
                                                                                BitIn=BitIn,
                                                                                BitW=BitW,
                                                                                BitActivation=BitActivation,
                                                                                BitOut=BitOut,
                                                                                NUM_RESULTS=NUM_RESULTS,
                                                                                Solution_idx=0
                                                                                )

            # Now the number of solutions is known, iterate and find all
            for sol_idx in range(NUM_FOUND_SOLUTIONS):
                C_in, C_out, H_in, H_out, W_in, W_out, Obj, NUM_FOUND_SOLUTIONS = get_tiling_dw_like(
                                                                                    DW=DW,
                                                                                    filter_size1=filter_size1,
                                                                                    filter_size2=filter_size2,
                                                                                    stride=stride,
                                                                                    padding_top=padding_top,
                                                                                    padding_bottom=padding_bottom,
                                                                                    padding_left=padding_left,
                                                                                    padding_right=padding_right,
                                                                                    groups=groups,
                                                                                    BN=BN,
                                                                                    in_channels=in_channels,
                                                                                    out_channels=out_channels,
                                                                                    x_shape=x_shape,
                                                                                    y_shape=y_shape,
                                                                                    buffer_size=buffer_size,
                                                                                    multiple_buffering_factor=multiple_buffering_factor,
                                                                                    name=name,
                                                                                    BitIn=BitIn,
                                                                                    BitW=BitW,
                                                                                    BitActivation=BitActivation,
                                                                                    BitOut=BitOut,
                                                                                    NUM_RESULTS=NUM_FOUND_SOLUTIONS,
                                                                                    Solution_idx=(NUM_FOUND_SOLUTIONS-sol_idx-1)
                                                                                    )    
                N_input.append(int(C_in))
                N_output.append(int(C_out))
                H_input.append(int(H_in))
                H_output.append(int(H_out))
                W_input.append(int(W_in))
                W_output.append(int(W_out))
                Obj_values.append(int(Obj))    
                

            return (N_input, N_output, H_input, H_output, W_input, W_output, Obj_values, NUM_FOUND_SOLUTIONS)




# Core function to compute the optimal solutions to occupy L1
def get_tiling_conv2d_like(DW,
                           filter_size1,
                           filter_size2,
                           stride,
                           padding_top,padding_bottom,padding_left,padding_right,
                           groups,
                           BN,
                           in_channels,
                           out_channels,
                           x_shape,
                           y_shape,
                           buffer_size,
                           multiple_buffering_factor=2,
                           name='conv',
                           BitIn=32,
                           BitW=32,
                           BitActivation=32,
                           BitOut=32,
                           NUM_RESULTS=10,
                           Solution_idx=0): 

    # This function is used to create the tiling parameters for a conv2d like operation.
    ## initial parameters
    fs1 = filter_size1
    fs2 = filter_size2
    s = stride
    g = groups
    n_in = in_channels * g
    n_out = out_channels
    h_in = x_shape + padding_top + padding_bottom #x_shape[-2] + padding_top + padding_bottom
    w_in = x_shape + padding_left + padding_right #x_shape[-1] + padding_left + padding_right
    h_out = y_shape #y_shape[-2]
    w_out = y_shape #y_shape[-1]
    h_in = x_shape #x_shape[-2]
    w_in = x_shape #x_shape[-1]
    max_tile_n_out = n_out
    max_tile_n_in = n_in
    min_tile_w_in = fs2
    min_tile_h_in = fs1
    min_tile_w_out = 1
    min_tile_h_out = 1
    # Output arrays
    N_input = 0
    N_output = 0
    W_input = 0
    W_output = 0
    H_input = 0
    H_output = 0
    Obj_values = 0
    # this is to renormalize all costs
    max_obj_value = buffer_size * 8 * 32 * 10000
    # constraints
    input_dim = BitIn * n_in * h_in * w_in
    output_dim = BitOut * n_out * h_out * w_out
    if DW == 0:
        weight_dim = BitW * n_in * n_out * fs1 * fs2
    else:
        weight_dim = BitW * n_out * fs1 * fs2
    if DW == 0:
        im2col_dim = 8 * 2 * 8 * fs1 * fs2 * n_in #always 8 since im2col contains unpacked data
    else:
        #im2col_dim = 32 * fs1 * fs2 * h_out * w_out * n_in
        im2col_dim = 8 * 8 * (fs1 * (h_in + padding_top + padding_bottom) + fs1) * int(32 / min(BitIn, BitOut, BitW)) 
        weight_full_prec_dim = 8 * 8 * fs1 * fs2 * int(32 / min(BitIn, BitOut, BitW))
        if BitW==8:
             weight_full_prec_dim = 0
    if 'MatMul' in name or 'Gemm' in name or 'PW' in name:
        im2col_dim = 0
    bn_dim = BitActivation * n_out * 2
    buffer_total = input_dim + output_dim + weight_dim + im2col_dim + bn_dim

    if DW == 1:
        buffer_total+= weight_full_prec_dim
    if BN == 0:
        buffer_total -= bn_dim   
    # return immediatly if the memory fits the L1   
    if buffer_total <= buffer_size * 8:
        if fs2 == h_in and h_out == 1:
            h_in = h_in - padding_bottom
        if fs1 == w_in and w_out == 1:
            w_in = w_in - padding_right
        return (n_in, n_out, h_in, h_out, w_in, w_out)
    else:
        db = multiple_buffering_factor
    # searching for tiling parameters
    parameters = pywrapcp.Solver.DefaultSolverParameters()
    solver = pywrapcp.Solver("simple_CP", parameters)
    tile_n_in = solver.IntVar(1, max_tile_n_in, 'tile_n_in')
    tile_n_out = solver.IntVar(1, max_tile_n_out, 'tile_n_out')
    tile_h_in = solver.IntVar(min_tile_h_in, h_in, 'tile_h_in')
    if h_in<min_tile_h_in:
        tile_h_in = solver.IntVar(min_tile_h_in, min_tile_h_in, 'tile_h_in')
    tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out')
    tile_w_in = solver.IntVar(min_tile_w_in, w_in, 'tile_w_in')
    if w_in<min_tile_w_in:
        tile_w_in = solver.IntVar(min_tile_w_in, min_tile_w_in, 'tile_w_in')
    tile_w_out = solver.IntVar(min_tile_w_out, w_out, 'tile_w_out')
    zero_variable = solver.IntVar(0, 0, 'zero_variable')
    one_variable = solver.IntVar(1, 1, 'one_variable')

    # scaling is used to ensure datasize is integer
    ds_x_scale = int(math.floor(32 * BitIn))
    ds_y_scale = int(math.floor(32 * BitOut))
    ds_W_scale = int(math.floor(32 * BitW))
    ds_bn_scale = int(math.floor(32 * BitActivation))
    if DW != 1 or (h_in > 32 and w_in > 32):
#        pass
        solver.Add(0 == (tile_h_in - fs1) % s)
        #solver.Add(0 == (tile_w_in - fs2) % s)
    if DW == 1:
        solver.Add(tile_n_in == tile_n_out)
    if DW == 1:
        if h_in <= 32 and w_in <= 32:
            solver.Add(tile_h_in == h_in)
            solver.Add(tile_w_in == w_in)
            solver.Add(tile_h_out == h_out)
            solver.Add(tile_w_out == w_out)
        elif h_in > 32 or w_in > 32:
            solver.Add(tile_h_out * s == (tile_h_in - (fs1 - 1) + ((tile_h_in % h_in) == 0) * (padding_top + padding_bottom) + (s - 1)))
            #solver.Add(tile_w_out * s == (tile_w_in - (fs2 - 1) + ((tile_w_in % w_in) == 0) * (padding_left + padding_right) + (s - 1)))
            solver.Add(tile_w_in == w_in)
            solver.Add(tile_w_out == w_out)
    elif DW == 0:
        solver.Add(tile_h_out * s ==(tile_h_in - (fs1 - 1) + (s - 1)))
        solver.Add(tile_w_out * s ==(tile_w_in - (fs2 - 1) + (s - 1)))
    solver.Add((n_out*one_variable) % tile_n_out == 0)
    # Input tile cannot be smaller than filter kernel!!
    solver.Add(tile_h_in >= fs1)
    solver.Add(tile_w_in >= fs2)
    # constraints of border tile. It can't be smaller than filter size
    solver.Add(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - padding_top)), 0) % (tile_h_in - fs1 + 1) + abs(solver.Min(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - padding_bottom)), 0) % (tile_h_in - fs1 + 1), 1) - 1) * fs1 >= fs1)
    solver.Add(solver.Max((w_in - tile_w_in - (tile_w_in - fs2 + 1 - padding_left)), 0) % (tile_w_in - fs2 + 1) + abs(solver.Min(solver.Max((w_in - tile_w_in - (tile_w_in - fs2 + 1 - padding_right)), 0) % (tile_w_in - fs2 + 1), 1) - 1) * fs2 >= fs2)
    constr_in = db * ds_x_scale * tile_n_in * tile_h_in * tile_w_in
    constr_out = db * ds_y_scale * tile_n_out * tile_h_out * tile_w_out
    if DW == 0:
        constr_weight = db * ds_W_scale * tile_n_in * tile_n_out * fs1 * fs2
        constr_im2col = 32 * 8 * 2 * 8 * fs1 * fs2 * tile_n_in
    else:
        constr_weight = db * ds_W_scale * tile_n_in * fs1 * fs2
        constr_im2col = 32 * 8 * 8 * ( fs1 * (tile_h_in + padding_top + padding_bottom) + fs1) * int(32 / min(BitIn, BitOut, BitW))
        constr_weight_full_prec = db * 32 * 8 * 8 * fs1 * fs2 * int(32 / min(BitIn, BitOut, BitW))
        if BitW==8:
            constr_weight_full_prec = 0
    if 'MatMul' in name or 'Gemm' in name or 'PW' in name:
        constr_im2col = 0
    constr_bn = ds_bn_scale * tile_n_out * 2 * db
    constraint_all = constr_in + constr_out + constr_weight + constr_bn + constr_im2col + 20 
    if DW == 1:
        constraint_all += constr_weight_full_prec
    if BN == 0:
        constraint_all -= constr_bn
    solver.Add(constraint_all <= 32 * buffer_size * 8)
    if DW == 0:
        solver.Add(tile_n_in == n_in)
    # constraint for future mixed
    if DW == 1: 
        solver.Add(tile_n_in % (int(32/min(BitIn, BitOut, BitW)))==0)
    # solver.Add(tile_n_out % (int(32/min(BitIn, BitOut, BitW)))==0)
    # obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
    # added some constraints for border tiles:     
    # 1. TILE_N_OUT / 4 LOWER IMPORTANCE THAN W / 2 and H / 8
    # 2. same constraintN_inputexpr == (64 * 10000 * tile_n_out
                                # + constraint_all
                                # + 64 * 2000000 * ((tile_h_out - 1) % 8)
                                # + 64 * 3000000 * ((tile_w_out - 1) % 2)
                                # + 64 * 1000000 * ((tile_n_out - 1) % 4) 
                                # + 64 * 1000000 * (tile_w_out * tile_h_out >= 16)
                                # + 64 * 10000 * ((n_out-zero_variable) % (tile_n_out+1))
                                # + 64 * 10000 * (((n_out-zero_variable) % (tile_n_out+1)) % 4)
                                # + 64 * 20000 * (((h_out-zero_variable) % (tile_h_out+1)) % 8)
                                # + 64 * 30000 * (((w_out-zero_variable) % (tile_w_out+1)) % 2) ))
    # else:
        # solver.Add(obj_expr == (constraint_all
                                # + 32 * 1000 * tile_w_out
                                # + 32 * 1000 * tile_h_out
                                # + 32 * 10000 * ((tile_n_out > 7))
                                # + 64 * 10000 * ((tile_n_out - 1) % int(8*8/min(BitIn, BitOut, BitW)))
                                # + 32 * 10000 * ((tile_h_out % 4) == 0)
                                # + 32 * 100 * (((n_out-zero_variable) % (tile_n_out+1)) > 7)
                                # + 32 * 100 * (((h_out-zero_variable) % (tile_h_out+1)))
                                # + 32 * 100 * (((h_out-zero_variable) % (tile_h_out+1)) % 4)
                                # + 32 * 100 * (((w_out-zero_variable) % (tile_w_out+1)))))

    objective = solver.Maximize(constraint_all, 1)

    decision_builder = solver.Phase([tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out],
                                    #solver.CHOOSE_HIGHEST_MAX, 
                                    #solver.CHOOSE_MAX_SIZE, 
                                    solver.CHOOSE_FIRST_UNBOUND, # solver chooses the next unbound variable to tie in the order of the list above
                                    #solver.CHOOSE_RANDOM, # solver chooses the next unbound variable to tie randomly
                                    solver.ASSIGN_RANDOM_VALUE) # solver assigns values to unbound variables randomly
                                    
    # Create a solution collector.
    #collector = solver.LastSolutionCollector()  # MODIFY TO GET SUCCESSIVE SOLUTIONS
    collector = solver.AllSolutionCollector() 
    #print(collector)
    # Add the decision variables.
    collector.Add(tile_n_in)
    collector.Add(tile_n_out)
    collector.Add(tile_h_in)
    collector.Add(tile_h_out)
    collector.Add(tile_w_in)
    collector.Add(tile_w_out)
    # Add the objective.
    collector.AddObjective(constraint_all)
    solver.Solve(decision_builder, [objective, collector])
    #import IPython; IPython.embed()
    NUM_FOUND_SOLUTIONS = 0
    if collector.SolutionCount() > 0:
        if Solution_idx == 0:
            print('Found {} solutions.'.format(collector.SolutionCount()))
        NUM_FOUND_SOLUTIONS = min(NUM_RESULTS, collector.SolutionCount())

        # TO BE DEBUGGED
        single_sol = collector.Solution(collector.SolutionCount() - 1 - Solution_idx)
        obj_value = single_sol.ObjectiveValue()
        tile_n_in = single_sol.Value(tile_n_in) 
        tile_n_out = single_sol.Value(tile_n_out) 
        tile_h_in = single_sol.Value(tile_h_in) 
        tile_h_out = single_sol.Value(tile_h_out) 
        tile_w_in = single_sol.Value(tile_w_in) 
        tile_w_out = single_sol.Value(tile_w_out) 
        if tile_h_in >= h_in:
            tile_h_in = h_in
            tile_h_out = int((tile_h_in -(fs1 - 1) + (padding_top + padding_bottom) + (s - 1))/s)
        if tile_w_in >= w_in:
            tile_w_in = w_in
            tile_w_out = int((tile_w_in -(fs2 - 1) + (padding_left + padding_right) + (s - 1))/s)

        ## QUICKFIX
        #if (tile_h_out <= 0):
        #    print("Avoided H_out={} by setting H_out=(H_in-ker_H+1)={}".format(tile_h_out, tile_h_in-fs1+1))
        #    tile_h_out = tile_h_in-fs1+1

        ## QUICKFIX
        if tile_h_in < fs1:
            tile_h_in = fs1
            tile_h_out = tile_h_in - fs1 + 1
        if tile_w_in < fs2:
            tile_w_in = fs2
            tile_w_out = tile_w_in - fs2 + 1
            

        N_input = tile_n_in #N_input.append(tile_n_in) 
        N_output = tile_n_out #N_output.append(tile_n_out) 
        H_input = tile_h_in #H_input.append(tile_h_in) 
        H_output = tile_h_out #H_output.append(tile_h_out) 
        W_input = tile_w_in #W_input.append(tile_w_in)
        W_output = tile_w_out #W_output.append(tile_w_out)
        Obj_values = obj_value #Obj_values.append(obj_value)

    return (N_input, N_output, H_input, H_output, W_input, W_output, Obj_values, NUM_FOUND_SOLUTIONS)
    print("  Conv2d ERROR: no L2-L1 tiling found. Exiting...")
    os._exit(0)




# Core function to compute the optimal solutions to occupy L1
def get_tiling_dw_like(DW,
                           filter_size1,
                           filter_size2,
                           stride,
                           padding_top,padding_bottom,padding_left,padding_right,
                           groups,
                           BN,
                           in_channels,
                           out_channels,
                           x_shape,
                           y_shape,
                           buffer_size,
                           multiple_buffering_factor=2,
                           name='conv',
                           BitIn=32,
                           BitW=32,
                           BitActivation=32,
                           BitOut=32,
                           NUM_RESULTS=10,
                           Solution_idx=0): 

    # This function is used to create the tiling parameters for a conv2d like operation.
    ## initial parameters
    fs1 = filter_size1
    fs2 = filter_size2
    s = stride
    g = groups
    n_in = in_channels * g
    n_out = out_channels
    h_in = y_shape + padding_top + padding_bottom #x_shape[-2] + padding_top + padding_bottom
    w_in = x_shape + padding_left + padding_right #x_shape[-1] + padding_left + padding_right
    h_out = y_shape - fs1 + 1 #y_shape[-2]
    w_out = x_shape - fs2 + 1 #y_shape[-1]
    #h_in = x_shape #x_shape[-2]
    #w_in = x_shape #x_shape[-1]
    max_tile_n_out = n_out
    max_tile_n_in = n_in
    min_tile_w_in = fs2
    min_tile_h_in = fs1
    min_tile_w_out = 1
    min_tile_h_out = 1
    # Output arrays
    N_input = 0
    N_output = 0
    W_input = 0
    W_output = 0
    H_input = 0
    H_output = 0
    Obj_values = 0
    # this is to renormalize all costs
    max_obj_value = buffer_size * 8 * 32 * 10000
    # constraints
    input_dim = BitIn * n_in * h_in * w_in
    output_dim = BitOut * n_out * h_out * w_out
    if DW == 0:
        weight_dim = BitW * n_in * n_out * fs1 * fs2
    else:
        weight_dim = BitW * n_out * fs1 * fs2
    if DW == 0:
        im2col_dim = 8 * 2 * 8 * fs1 * fs2 * n_in #always 8 since im2col contains unpacked data
    else:
        im2col_dim = 32 * fs1 * fs2 * h_out * w_out * n_in
        #im2col_dim = 8 * 8 * (fs1 * (h_in + padding_top + padding_bottom) + fs1) * int(32 / min(BitIn, BitOut, BitW)) 
        weight_full_prec_dim = 8 * 8 * fs1 * fs2 * int(32 / min(BitIn, BitOut, BitW))
        if BitW==8:
             weight_full_prec_dim = 0
    if 'MatMul' in name or 'Gemm' in name or 'PW' in name:
        im2col_dim = 0
    bn_dim = BitActivation * n_out * 2
    buffer_total = input_dim + output_dim + weight_dim + im2col_dim + bn_dim

    if DW == 1:
        buffer_total+= weight_full_prec_dim
    if BN == 0:
        buffer_total -= bn_dim   
    # return immediatly if the memory fits the L1   
    if buffer_total <= buffer_size * 8:
        if fs2 == h_in and h_out == 1:
            h_in = h_in - padding_bottom
        if fs1 == w_in and w_out == 1:
            w_in = w_in - padding_right
        return (n_in, n_out, h_in, h_out, w_in, w_out)
    else:
        db = multiple_buffering_factor
    # searching for tiling parameters
    parameters = pywrapcp.Solver.DefaultSolverParameters()
    solver = pywrapcp.Solver("simple_CP", parameters)
    tile_n_in = solver.IntVar(1, max_tile_n_in, 'tile_n_in')
    tile_n_out = solver.IntVar(1, max_tile_n_out, 'tile_n_out')
    tile_h_in = solver.IntVar(min_tile_h_in, h_in, 'tile_h_in')
    if h_in<min_tile_h_in:
        tile_h_in = solver.IntVar(min_tile_h_in, min_tile_h_in, 'tile_h_in')
    tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out')
    tile_w_in = solver.IntVar(min_tile_w_in, w_in, 'tile_w_in')
    if w_in<min_tile_w_in:
        tile_w_in = solver.IntVar(min_tile_w_in, min_tile_w_in, 'tile_w_in')
    tile_w_out = solver.IntVar(min_tile_w_out, w_out, 'tile_w_out')
    zero_variable = solver.IntVar(0, 0, 'zero_variable')
    one_variable = solver.IntVar(1, 1, 'one_variable')

    # scaling is used to ensure datasize is integer
    ds_x_scale = int(math.floor(32 * BitIn))
    ds_y_scale = int(math.floor(32 * BitOut))
    ds_W_scale = int(math.floor(32 * BitW))
    ds_bn_scale = int(math.floor(32 * BitActivation))
    if DW != 1 or (h_in > 32 and w_in > 32):
#        pass
        solver.Add(0 == (tile_h_in - fs1) % s)
        #solver.Add(0 == (tile_w_in - fs2) % s)
    if DW == 1:
        solver.Add(tile_n_in == tile_n_out)
    if DW == 1:
        if h_in <= 100 and w_in <= 100: #if h_in <= 32 and w_in <= 32:
            solver.Add(tile_h_in == tile_w_in)
            solver.Add(tile_h_out == tile_w_out)
            solver.Add(tile_h_out == tile_h_in - fs1 + 1)
            solver.Add(tile_w_out == tile_w_in - fs2 + 1)
            #solver.Add(tile_h_in == h_in)
            #solver.Add(tile_w_in == w_in)
            #solver.Add(tile_h_out == h_out)
            #solver.Add(tile_w_out == w_out)
        elif h_in > 32 or w_in > 32:
            solver.Add(tile_h_out * s == (tile_h_in - (fs1 - 1) + ((tile_h_in % h_in) == 0) * (padding_top + padding_bottom) + (s - 1)))
            #solver.Add(tile_w_out * s == (tile_w_in - (fs2 - 1) + ((tile_w_in % w_in) == 0) * (padding_left + padding_right) + (s - 1)))
            solver.Add(tile_w_in == w_in)
            solver.Add(tile_w_out == w_out)
    elif DW == 0:
        solver.Add(tile_h_out * s ==(tile_h_in - (fs1 - 1) + (s - 1)))
        solver.Add(tile_w_out * s ==(tile_w_in - (fs2 - 1) + (s - 1)))
    solver.Add((n_out*one_variable) % tile_n_out == 0)
    # Avoid tiles with 0 sizes
    solver.Add(tile_h_in > 0)
    solver.Add(tile_w_in > 0)
    solver.Add(tile_h_out > 0)
    solver.Add(tile_w_out > 0)
    # constraints of border tile. It can't be smaller than filter size
    solver.Add(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - padding_top)), 0) % (tile_h_in - fs1 + 1) + abs(solver.Min(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - padding_bottom)), 0) % (tile_h_in - fs1 + 1), 1) - 1) * fs1 >= fs1)
    solver.Add(solver.Max((w_in - tile_w_in - (tile_w_in - fs2 + 1 - padding_left)), 0) % (tile_w_in - fs2 + 1) + abs(solver.Min(solver.Max((w_in - tile_w_in - (tile_w_in - fs2 + 1 - padding_right)), 0) % (tile_w_in - fs2 + 1), 1) - 1) * fs2 >= fs2)
    constr_in = db * ds_x_scale * tile_n_in * tile_h_in * tile_w_in
    constr_out = db * ds_y_scale * tile_n_out * tile_h_out * tile_w_out
    if DW == 0:
        constr_weight = db * ds_W_scale * tile_n_in * tile_n_out * fs1 * fs2
        constr_im2col = 32 * 8 * 2 * 8 * fs1 * fs2 * tile_n_in
    else:
        constr_weight = db * ds_W_scale * tile_n_in * fs1 * fs2
        constr_im2col = 32 * 8 * 8 * ( fs1 * (tile_h_in + padding_top + padding_bottom) + fs1) * int(32 / min(BitIn, BitOut, BitW))
        constr_weight_full_prec = db * 32 * 8 * 8 * fs1 * fs2 * int(32 / min(BitIn, BitOut, BitW))
        if BitW==8:
            constr_weight_full_prec = 0
    if 'MatMul' in name or 'Gemm' in name or 'PW' in name:
        constr_im2col = 0
    constr_bn = ds_bn_scale * tile_n_out * 2 * db
    constraint_all = constr_in + constr_out + constr_weight + constr_bn + constr_im2col + 20 
    if DW == 1:
        constraint_all += constr_weight_full_prec
    if BN == 0:
        constraint_all -= constr_bn
    solver.Add(constraint_all <= 32 * buffer_size * 8)
    if DW == 0:
        solver.Add(tile_n_in == n_in)
    # constraint for future mixed
    if DW == 1: 
        solver.Add(tile_n_in % (int(32/min(BitIn, BitOut, BitW)))==0)
    # solver.Add(tile_n_out % (int(32/min(BitIn, BitOut, BitW)))==0)
    # obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
    # added some constraints for border tiles:     
    # 1. TILE_N_OUT / 4 LOWER IMPORTANCE THAN W / 2 and H / 8
    # 2. same constraintN_inputexpr == (64 * 10000 * tile_n_out
                                # + constraint_all
                                # + 64 * 2000000 * ((tile_h_out - 1) % 8)
                                # + 64 * 3000000 * ((tile_w_out - 1) % 2)
                                # + 64 * 1000000 * ((tile_n_out - 1) % 4) 
                                # + 64 * 1000000 * (tile_w_out * tile_h_out >= 16)
                                # + 64 * 10000 * ((n_out-zero_variable) % (tile_n_out+1))
                                # + 64 * 10000 * (((n_out-zero_variable) % (tile_n_out+1)) % 4)
                                # + 64 * 20000 * (((h_out-zero_variable) % (tile_h_out+1)) % 8)
                                # + 64 * 30000 * (((w_out-zero_variable) % (tile_w_out+1)) % 2) ))
    # else:
        # solver.Add(obj_expr == (constraint_all
                                # + 32 * 1000 * tile_w_out
                                # + 32 * 1000 * tile_h_out
                                # + 32 * 10000 * ((tile_n_out > 7))
                                # + 64 * 10000 * ((tile_n_out - 1) % int(8*8/min(BitIn, BitOut, BitW)))
                                # + 32 * 10000 * ((tile_h_out % 4) == 0)
                                # + 32 * 100 * (((n_out-zero_variable) % (tile_n_out+1)) > 7)
                                # + 32 * 100 * (((h_out-zero_variable) % (tile_h_out+1)))
                                # + 32 * 100 * (((h_out-zero_variable) % (tile_h_out+1)) % 4)
                                # + 32 * 100 * (((w_out-zero_variable) % (tile_w_out+1)))))

    objective = solver.Maximize(constraint_all, 1)

    decision_builder = solver.Phase([tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out],
                                    #solver.CHOOSE_HIGHEST_MAX, 
                                    #solver.CHOOSE_MAX_SIZE, 
                                    #solver.CHOOSE_FIRST_UNBOUND, # solver chooses the next unbound variable to tie in the order of the list above
                                    solver.CHOOSE_RANDOM, # solver chooses the next unbound variable to tie randomly
                                    solver.ASSIGN_RANDOM_VALUE) # solver assigns values to unbound variables randomly
                                    
    # Create a solution collector.
    #collector = solver.LastSolutionCollector()  # MODIFY TO GET SUCCESSIVE SOLUTIONS
    collector = solver.AllSolutionCollector() 
    #print(collector)
    # Add the decision variables.
    collector.Add(tile_n_in)
    collector.Add(tile_n_out)
    collector.Add(tile_h_in)
    collector.Add(tile_h_out)
    collector.Add(tile_w_in)
    collector.Add(tile_w_out)
    # Add the objective.
    collector.AddObjective(constraint_all)
    solver.Solve(decision_builder, [objective, collector])
    #import IPython; IPython.embed()
    NUM_FOUND_SOLUTIONS = 0
    if collector.SolutionCount() > 0:
        if Solution_idx == 0:
            print('Found {} solutions.'.format(collector.SolutionCount()))
        NUM_FOUND_SOLUTIONS = min(NUM_RESULTS, collector.SolutionCount())

        # TO BE DEBUGGED
        single_sol = collector.Solution(collector.SolutionCount() - 1 - Solution_idx)
        obj_value = single_sol.ObjectiveValue()
        tile_n_in = single_sol.Value(tile_n_in) 
        tile_n_out = single_sol.Value(tile_n_out) 
        tile_h_in = single_sol.Value(tile_h_in) 
        tile_h_out = single_sol.Value(tile_h_out) 
        tile_w_in = single_sol.Value(tile_w_in) 
        tile_w_out = single_sol.Value(tile_w_out) 
        if tile_h_in >= h_in:
            tile_h_in = h_in
            tile_h_out = int((tile_h_in -(fs1 - 1) + (padding_top + padding_bottom) + (s - 1))/s)
        if tile_w_in >= w_in:
            tile_w_in = w_in
            tile_w_out = int((tile_w_in -(fs2 - 1) + (padding_left + padding_right) + (s - 1))/s)
        N_input = tile_n_in #N_input.append(tile_n_in) 
        N_output = tile_n_out #N_output.append(tile_n_out) 
        H_input = tile_h_in #H_input.append(tile_h_in) 
        H_output = tile_h_out #H_output.append(tile_h_out) 
        W_input = tile_w_in #W_input.append(tile_w_in)
        W_output = tile_w_out #W_output.append(tile_w_out)
        Obj_values = obj_value #Obj_values.append(obj_value)

    return (N_input, N_output, H_input, H_output, W_input, W_output, Obj_values, NUM_FOUND_SOLUTIONS)
    print("  Conv2d ERROR: no L2-L1 tiling found. Exiting...")
    os._exit(0)



# Naive tiler for DAC paper
def steven_the_tiler (DW=0,
                      filter_size1=4,
                      filter_size2=10,
                      stride=1,
                      padding_top=0,padding_bottom=0,padding_left=0,padding_right=0,
                      groups=1,
                      BN=0,
                      in_channels=1,
                      out_channels=64,
                      x_shape=8,
                      y_shape=34,
                      buffer_size=64*1024,
                      multiple_buffering_factor=2,
                      name='conv',
                      BitIn=32,
                      BitW=32,
                      BitActivation=32,
                      BitOut=32,
                      NUM_RESULTS=10,
                      Solution_idx=0,
                      layer_type='CONV2D',
                      use_bias=1,
                      ignore_in_grads=False,
                      NUM_CORES=8
                      ):

    # Size of the divider array to define the number of tiling elements
    divider_list = range(1000)
    # Lists of the subdivided layer sizes
    C_list = []
    H_list = []
    W_list = []
    Cout_list = []
    # Tile configuration and size variable
    tile_list = []
    max_memocc_list = []
    # Output variables
    NUM_FOUND_SOLUTIONS = 0
    N_input = []
    N_output = []
    H_input = []
    H_output = []
    W_input = []
    W_output = []
    Obj_values = []


    # TIle convolutions
    if layer_type == 'CONV2D' or layer_type == 'DW':
        for divider in range(len(divider_list)):
            divider += 1
            # In Channels
            if in_channels % divider == 0:
                C_list.append(int(in_channels/divider))
            # Height
            if divider == y_shape:
                H_list.append(int(divider))
            elif divider == filter_size1:
                H_list.append(int(divider))
            elif divider >= filter_size1 and divider < y_shape:
                dynasize = divider
                remainder = 0
                iterator = 0
                while dynasize <= y_shape:
                    dynasize = divider
                    dynasize += iterator * (divider - filter_size1 + 1)
                    remainder = y_shape - dynasize
                    iterator += 1
                if remainder == 0:
                    H_list.append(int(divider))
            # Width
            if divider == x_shape:
                W_list.append(int(divider))
            elif divider == filter_size2:
                W_list.append(int(divider))
            elif divider >= filter_size2 and divider <= x_shape:
                dynasize = divider
                remainder = 0
                iterator = 0
                while dynasize <= x_shape:
                    dynasize = divider
                    dynasize += iterator * (divider - filter_size2 + 1)
                    remainder = x_shape - dynasize
                    iterator += 1
                if remainder == 0:
                    W_list.append(int(divider))
            # Out Channels
            if out_channels % divider == 0:
                Cout_list.append(int(out_channels/divider))
            pass

    # Tile other layers
    else :
        # Find the list of dividers (without remainder to avoid border tiles)
        for divider in range(len(divider_list)):
            divider += 1
            if in_channels % divider == 0:
                C_list.append(int(in_channels/divider))
            if y_shape % divider == 0 and int(y_shape/divider) > filter_size1:
                H_list.append(int(y_shape/divider))
            if x_shape % divider == 0 and int(x_shape/divider) > filter_size2:
                W_list.append(int(x_shape/divider))
            if out_channels % divider == 0:
                Cout_list.append(int(out_channels/divider))

    # Avoid empty lists 
    if len(C_list) == 0:
        C_list.append(in_channels) 
    if len(H_list) == 0:
        H_list.append(y_shape)
    if len(W_list) == 0:
        W_list.append(x_shape)
    if len(Cout_list) == 0:
        Cout_list.append(out_channels)

    print("Raw lists..\n")
    print("C_list: " + str(C_list))
    print("H_list: " + str(H_list))
    print("W_list: " + str(W_list))
    print("Cout_list: " + str(Cout_list))

    # Build all the possible tile configurations by combining the tile sizes
    # General case
    if DW == 0:
        for chin in C_list:
            for height in H_list:
                for width in W_list:
                    for chout in Cout_list:
                        tile_config = []
                        tile_config.append(chin)
                        tile_config.append(height)
                        tile_config.append(width)
                        tile_config.append(chout)
                        # Compute memocc for the new tile
                        memocc_raw = compute_memory_footprint(layer_type=layer_type, C_in=chin, H_in=height, 
                                                    W_in=width, C_out=chout, H_out=height-filter_size1+1, W_out=width-filter_size2+1,
                                                    IN_BYTES=int(BitIn/8), KER_BYTES=int(BitW/8), OUT_BYTES=int(BitOut/8), USE_BIAS=use_bias)
                        if ignore_in_grads:
                            max_memocc = max(memocc_raw[0], memocc_raw[1])
                        else:
                            max_memocc = max(memocc_raw[0], memocc_raw[1], memocc_raw[2])
                        # Add new configuration to the lists
                        tile_list.append(tile_config)
                        max_memocc_list.append(max_memocc)
    # Special case for DW conv
    else:
        for chin in C_list:
            for height in H_list:
                for width in W_list:
                        tile_config = []
                        # TEMPORARY FIX DUE TO BROKEN PARALLELIZATION IF (ch % NUM_CORES != 0)
                        if chin % NUM_CORES == 0:
                        # END OF TEMPORARY FIX
                            tile_config.append(chin)
                            tile_config.append(height)
                            tile_config.append(width)
                            tile_config.append(chin)
                            # Compute memocc for the new tile
                            memocc_raw = compute_memory_footprint(layer_type, chin, height, width, chin, 
                                                    height-filter_size1+1, width-filter_size2+1, 
                                                    int(BitIn/8), int(BitW/8), int(BitOut/8))
                            if ignore_in_grads:
                                max_memocc = max(memocc_raw[0], memocc_raw[1])
                            else:
                                max_memocc = max(memocc_raw[0], memocc_raw[1], memocc_raw[2])
                            # Add new configuration to the lists
                            tile_list.append(tile_config)
                            max_memocc_list.append(max_memocc)

    # Report that N solutions were found
    num_found_solutions_print = len(max_memocc_list)
    print("\nFound {} solutions..\n".format(num_found_solutions_print))
    
    # Create dictionary to find best fitting solutions
    zip_iter = zip(tile_list, max_memocc_list)
    def take_largest (e):
        return e[-1]
    # Sort results by the highest memocc
    sorted_tiles = sorted(zip_iter, key=take_largest, reverse=True)
    
    # Extract the first best results which fit the memory
    for tile_t, memocc_t in sorted_tiles:
        if NUM_FOUND_SOLUTIONS < NUM_RESULTS and memocc_t < buffer_size:
            NUM_FOUND_SOLUTIONS += 1
            N_input.append(tile_t[0])
            H_input.append(tile_t[1])
            W_input.append(tile_t[2])
            N_output.append(tile_t[3])
            H_output.append(tile_t[1]-filter_size1+1)
            W_output.append(tile_t[2]-filter_size2+1)
            memocc_reread = compute_memory_footprint(layer_type, tile_t[0], tile_t[1], tile_t[2], tile_t[3],
                                                    tile_t[1]-filter_size1+1, tile_t[2]-filter_size2+1, 4, 4, 4, USE_BIAS=use_bias)
            print("Solution " + str(NUM_FOUND_SOLUTIONS) + " has a memocc of " + str(memocc_reread[0]) + ", " + str(memocc_reread[1]) + ", "
                                + str(memocc_reread[2]) + " bytes.")
            Obj_values.append(memocc_t)

    return (N_input, N_output, H_input, H_output, W_input, W_output, Obj_values, NUM_FOUND_SOLUTIONS)




# Finds the best matmul and its performance relatively to an experiment
def find_best_perf(source_file, tiling_test_idx, num_cores):

    # Open file to be read
    source_f = open(source_file, 'r')
    Lines = source_f.readlines()
    # Phrase to be found
    perf_header = '=====> BEST TO WORST <====='
    mm_cyc_premise = ' => '
    mm_cyc_suffix = ' cycles'
    # Variables
    correct_entry = False
    # Entries
    num_errors = 0
    broken_mm_name = []
    matmul_name = []
    matmul_cycles = []
    # Find performances
    for idx, line in enumerate(Lines):
        if (line.find(perf_header) != -1):
            matmul_name.append(str(Lines[idx+2]))
            raw_line = Lines[idx+3]
            matmul_cycles.append(int(raw_line[len(mm_cyc_premise) : (len(raw_line)-len(mm_cyc_suffix))]))
            while correct_entry == False:
                srch_idx = 0
                if (matmul_cycles[-1] == 0):
                    # Notify error
                    srch_idx += 1
                    num_errors += 1
                    # Save broken matmul name and take a new best one
                    broken_mm_name.append(matmul_name[-1])
                    matmul_name.pop(-1)
                    matmul_cycles.pop(-1)
                    matmul_name.append(str(Lines[idx+2+3*srch_idx]))
                    raw_line = Lines[idx+3+3*srch_idx]
                    matmul_cycles.append(int(raw_line[len(mm_cyc_premise) : (len(raw_line)-len(mm_cyc_suffix))]))
                else:
                    correct_entry = True
    source_f.close()

    return tiling_test_idx, matmul_name, matmul_cycles, num_cores, num_errors, broken_mm_name


# Write raw data from single experiments into the raw result file
def write_raw_file(source_file, raw_result_file):

    with open(source_file, 'r') as f:
        with open(raw_result_file, 'a') as raw_f:
            for line in f:
                raw_f.write(line)
                
    return


# Writes broken matmuls on the error file
def write_error_file (err_log_file, layer_step, tiling_idx, errors, broken_mm):

    # Open log file to append results
    f = open(err_log_file, 'a')
    if errors > 1:
        f.write("TILING EXAMPLE {} contains {} errors in {} step.\n".format(tiling_idx, errors, layer_step))
        f.write("   Broken matmuls are: ")
        for idx in range(errors):
            f.write("{}, ".format(broken_mm[idx]))
        f.write("\n\n")
    elif errors == 1:
        f.write("TILING EXAMPLE {} contains {} error in {} step.\n".format(tiling_idx, errors, layer_step))
        f.write("   Broken matmul: {}".format(broken_mm[-1]))
        f.write("\n\n")
    f.close()

    return 



# Sort the tiling results in the final log file selecting the best one
def sort_results (sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list):

    # GENERAL CHOICE
    #if (len(tiling_idx_list) == 1):
    #    take_all = 1
    #else:
    #    take_all = 0

    # TAKE ALWAYS ALL SOLUTIONS
    take_all = 1

    if take_all == 0:
    # Take the lists and create dictionary to sort them by the number of cycles (descending)
        zip_iter = zip(tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)
        def take_perf (e):
            return e[2]
        sorted_results = sorted(zip_iter, key=take_perf, reverse=True)
        # Take the first result, as it is the best performing one
        best_result = sorted_results[0]
        # Write the result to the sim_result_file
        f = open(sim_result_file, 'a')
        b_idx = best_result[0]
        b_mm = best_result[1]
        b_cyc = best_result[2] 
        b_cores = best_result[3]
        b_step = best_result[4]
        f.write("TILING TEST {} ({} pass):\t{} with {} cycles on {} cores\n".format(b_idx, b_step, b_mm, b_cyc, b_cores))           
        f.close()
    
    else:
        # Open destination file
        f = open(sim_result_file, 'a')
        for idx in range(len(tiling_idx_list)):
            f.write("TILING TEST {} ({} pass):\t{} with {} cycles on {} cores\n".format(tiling_idx_list[idx], passes_list[idx], matmul_names_list[idx], matmul_cycles_list[idx], num_cores_list[idx]))
        f.close()

    return



# Compute the memory footprint of a layer (RETURNS RESULT IN BYTE)
def compute_memory_footprint (layer_type, C_in, H_in, W_in, C_out, H_out, W_out, IN_BYTES, KER_BYTES, OUT_BYTES, USE_BIAS):

    # Internal variables
    if isinstance(C_in, list):
        Cin = C_in[0]
        Hin = H_in[0]
        Win = W_in[0]
        Cout = C_out[0]
        Hout = H_out[0]
        Wout = W_out[0]
    else:
        Cin = C_in
        Hin = H_in
        Win = W_in
        Cout = C_out
        Hout = H_out
        Wout = W_out

    # Memory occupation
    memory_size_bytes = []

    if layer_type == 'DW':
        # FW
        dw_ker_H = Hin - Hout + 1
        dw_ker_W = Win - Wout + 1
        in_act  = Hin * Win * Cin
        ker     = dw_ker_H * dw_ker_W * Cin
        im2colF = dw_ker_H * dw_ker_W * Cin * (Hin-dw_ker_H+1) * (Win-dw_ker_W+1)
        out_act = Cin * (Hin-dw_ker_H+1) * (Win-dw_ker_W+1)
        tot_FW  = in_act * IN_BYTES + ker * KER_BYTES + im2colF * IN_BYTES + out_act * OUT_BYTES
        memory_size_bytes.append(tot_FW)
        # WGT_G
        in_act  = Hin * Win * Cin
        ker     = dw_ker_H * dw_ker_W * Cin
        im2colW  = dw_ker_H * dw_ker_W * Cin * (Hin-dw_ker_H+1) * (Win-dw_ker_W+1) 
        out_act = Cin * (Hin-dw_ker_H+1) * (Win-dw_ker_W+1)
        tot_WGT = in_act * IN_BYTES + ker * KER_BYTES + im2colW * IN_BYTES + out_act * OUT_BYTES
        memory_size_bytes.append(tot_WGT)
        # IN_G
        in_act  = Hin * Win * Cin
        ker     = dw_ker_H * dw_ker_W * Cin
        im2colI = Hin * Win * Cin * dw_ker_H * dw_ker_W
        out_act = Cin * (Hin-dw_ker_H+1) * (Win-dw_ker_W+1)
        tot_ING = in_act * IN_BYTES + ker * KER_BYTES + im2colI * IN_BYTES + out_act * OUT_BYTES
        memory_size_bytes.append(tot_ING)

    elif layer_type == 'PW':
        # FW
        in_act  = Hin * Win * Cin
        ker     = Cin * Cout
        out_act = Hin * Win * Cout
        tot_FW  = in_act * IN_BYTES + ker * KER_BYTES + out_act * OUT_BYTES
        memory_size_bytes.append(tot_FW)
        # WGT_G
        in_act  = Hin * Win * Cin
        ker     = Cin * Cout
        out_act = Hin * Win * Cout
        tot_WGT = in_act * IN_BYTES + ker * KER_BYTES + out_act * OUT_BYTES
        memory_size_bytes.append(tot_WGT)
        # IN_G
        in_act  = Hin * Win * Cin
        ker     = Cin * Cout
        out_act = Hin * Win * Cout
        tot_ING = in_act * IN_BYTES + ker * KER_BYTES + out_act * OUT_BYTES
        memory_size_bytes.append(tot_ING)

    elif layer_type == 'LINEAR':
        # FW
        in_act  = Cin
        ker     = Cin * Cout
        out_act = Cout
        bias    = Cout
        tot_FW  = in_act * IN_BYTES + ker * KER_BYTES + out_act * OUT_BYTES + bias * KER_BYTES
        memory_size_bytes.append(tot_FW)
        # WGT_G
        in_act  = Cin
        ker     = Cin * Cout
        out_act = Cout
        bias    = Cout
        tot_WGT = in_act * IN_BYTES + ker * KER_BYTES + out_act * OUT_BYTES + bias * KER_BYTES
        memory_size_bytes.append(tot_WGT)
        # IN_G
        in_act  = Cin
        ker     = Cin * Cout
        out_act = Cout
        tot_ING = in_act * IN_BYTES + ker * KER_BYTES + out_act * OUT_BYTES
        memory_size_bytes.append(tot_ING)

    elif layer_type == 'CONV2D':
        # FW
        conv2d_ker_H = Hin - Hout + 1
        conv2d_ker_W = Win - Wout + 1
        in_act  = Cin * Hin * Win
        ker     = conv2d_ker_H * conv2d_ker_W * Cin * Cout
        bias    = Cout
        im2colF = Cin * conv2d_ker_H * conv2d_ker_W * (Win-conv2d_ker_W+1) * (Hin-conv2d_ker_H+1) 
        out_act = (Win-conv2d_ker_W+1) * (Hin-conv2d_ker_H+1) * Cout
        tot_FW  = in_act * IN_BYTES + ker * KER_BYTES + bias * KER_BYTES + im2colF * IN_BYTES + out_act * OUT_BYTES
        memory_size_bytes.append(tot_FW)
        # WGT_G
        in_act  = Cin * Hin * Win
        ker     = conv2d_ker_H * conv2d_ker_W * Cin * Cout
        bias    = Cout
        im2colW = Cin * conv2d_ker_H * conv2d_ker_W * (Win-conv2d_ker_W+1) * (Hin-conv2d_ker_H+1)
        # im2colW = Cout * conv2d_ker_H * conv2d_ker_W * (Win-conv2d_ker_W+1) * (Hin-conv2d_ker_H+1)
        out_act = (Win-conv2d_ker_W+1) * (Hin-conv2d_ker_H+1) * Cout
        tot_WGT = in_act * IN_BYTES + ker * KER_BYTES + bias * KER_BYTES + im2colW * IN_BYTES + out_act * OUT_BYTES
        memory_size_bytes.append(tot_WGT)
        # IN_G
        in_act  = Cin * Hin * Win
        ker     = conv2d_ker_H * conv2d_ker_W * Cin * Cout
        im2colI = Cout * conv2d_ker_H * conv2d_ker_W * Hin * Win
        out_act = (Win-conv2d_ker_W+1) * (Hin-conv2d_ker_H+1) * Cout
        tot_ING = in_act * IN_BYTES + ker * KER_BYTES + im2colI * IN_BYTES + out_act * OUT_BYTES
        memory_size_bytes.append(tot_ING)

    else:
        print("Invalid layer entry for memocc calculation!!")

    if USE_BIAS == 1 and layer_type not in ['CONV2D', 'LINEAR']:
        print("Bias not handled for selected layer type and will be ignored.")

    return memory_size_bytes