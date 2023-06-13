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

import os
import subprocess
import profile_utils as prof
import ci_utils as ci

"""
USER CONSTRAINTS
"""
timeout                     = 120       # Sets the timeout for each process
Conv2D_Opt_MM_FP32          = 10        # Selects the optimized MM for the Conv2D layers (FP32)
PointWise_Opt_MM_FP32       = 10        # Selects the optimized MM for the PW layers (FP32)
Linear_Opt_MM_FP32          = 0         # Selects the optimized MM for the Fully Conneced layer (FP32)
Conv2D_Opt_MM_FP16          = 3         # Selects the optimized MM for the Conv2D layers (FP16)
PointWise_Opt_MM_FP16       = 3         # Selects the optimized MM for the PW layers (FP16)
Linear_Opt_MM_FP16          = 0         # Selects the optimized MM for the Fully Conneced layer (FP16)


"""
BACKEND
"""
ci_cwd = os.getcwd()
test_cwd = os.getcwd()
trainlib_cwd = os.getcwd() + "/../../lib"
results_file = ci_cwd + "/test_suite_results.txt"
checkpoint = ci_cwd + "/checkpoint.txt"

with open(results_file, 'w') as f:

    print("<<< ENTERING TEST SEQUENCE FOR CONTINUOUS INTEGRATION >>>")

    # Create the temp folder
    if not os.path.exists(ci_cwd+"/temp"):
        os.mkdir(ci_cwd+"/temp")   
    if not os.path.exists(ci_cwd+"/temp/tests"):
        os.mkdir(ci_cwd+"/temp/tests")
    if not os.path.exists(ci_cwd+"/temp/lib"):
        os.mkdir(ci_cwd+"/temp/lib")
    
    # Go to the test folder
    os.chdir(ci_cwd+"/../../tests/")
    test_cwd = os.getcwd()

    print("CI Suite Folder: "+ci_cwd)
    print("Test Folder: "+test_cwd)
    print("TrainLib Folder: "+trainlib_cwd)

    # Copy PULP-TrainLib in the right position
    ci.copy_trainlib_ci(ci_cwd, trainlib_cwd)



    """
    START TEST SEQUENCE
    """
    test_sequence_iterator = 0

    print("\n=====> ENTERING TEST SEQUENCE FOR ACTIVATIONS.. <=====\n")

    # Test settings
    current_test_source_folder = test_cwd + "/test_act"
    cmd = "rm -rf BUILD/; make clean get_golden all run DATA_TYPE='FP32' > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Activations (FP32): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_act"
    cmd = "rm -rf BUILD/; make clean get_golden all run DATA_TYPE='FP16' > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Activations (FP16): ", 0, results_file)
    test_sequence_iterator += 1


    # TO DO
    #print("\n=====> ENTERING TEST SEQUENCE FOR BLOCKTRANSPOSE.. <=====\n")


    print("\n=====> ENTERING TEST SEQUENCE FOR FP16 DEPTWHISE AND POINTWISE.. <=====\n")

    """
    DEPTHWISE CONVOLUTION
    """

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_FORWARD' HWC_layout=0 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP16, FW, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_BACKWARD_GRAD' HWC_layout=0 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP16, WG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_BACKWARD_ERROR' HWC_layout=0 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP16, IG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_FORWARD' HWC_layout=1 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP16, FW, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_BACKWARD_GRAD' HWC_layout=1 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP16, WG, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_BACKWARD_ERROR' HWC_layout=1 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP16, IG, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    """
    POINTWISE CONVOLUTION
    """

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_FORWARD' HWC_layout=0 MATMUL_TYPE="+str(PointWise_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP16, FW, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_BACKWARD_GRAD' HWC_layout=0 MATMUL_TYPE="+str(PointWise_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP16, WG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_BACKWARD_ERROR' HWC_layout=0 MATMUL_TYPE="+str(PointWise_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP16, IG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_FORWARD' HWC_layout=1 MATMUL_TYPE="+str(PointWise_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP16, FW, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_BACKWARD_GRAD' HWC_layout=1 MATMUL_TYPE="+str(PointWise_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP16, WG, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_BACKWARD_ERROR' HWC_layout=1 MATMUL_TYPE="+str(PointWise_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP16, IG, HWC): ", 0, results_file)
    test_sequence_iterator += 1


    print("\n=====> ENTERING TEST SEQUENCE FOR FP32 DEPTWHISE AND POINTWISE.. <=====\n")

    """
    DEPTHWISE CONVOLUTION
    """

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_FORWARD' HWC_layout=0 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP32, FW, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_BACKWARD_GRAD' HWC_layout=0 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP32, WG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_BACKWARD_ERROR' HWC_layout=0 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP32, IG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_FORWARD' HWC_layout=1 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP32, FW, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_BACKWARD_GRAD' HWC_layout=1 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP32, WG, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='DW_BACKWARD_ERROR' HWC_layout=1 > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Depthwise (FP32, IG, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    """
    POINTWISE CONVOLUTION
    """

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_FORWARD' HWC_layout=0 MATMUL_TYPE="+str(PointWise_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP32, FW, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_BACKWARD_GRAD' HWC_layout=0 MATMUL_TYPE="+str(PointWise_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP32, WG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_BACKWARD_ERROR' HWC_layout=0 MATMUL_TYPE="+str(PointWise_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP32, IG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_FORWARD' HWC_layout=1 MATMUL_TYPE="+str(PointWise_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP32, FW, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_BACKWARD_GRAD' HWC_layout=1 MATMUL_TYPE="+str(PointWise_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP32, WG, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv_pw_dw_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='PW_BACKWARD_ERROR' HWC_layout=1 MATMUL_TYPE="+str(PointWise_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Pointwise (FP32, IG, HWC): ", 0, results_file)
    test_sequence_iterator += 1


    print("\n=====> ENTERING TEST SEQUENCE FOR FP16 CONV2D.. <=====\n")

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='FORWARD' HWC_LAYOUT=0 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP16, FW, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_GRAD' HWC_LAYOUT=0 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP16, WG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_ERROR' HWC_LAYOUT=0 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP16, IG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='FORWARD' HWC_LAYOUT=1 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP16, FW, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_GRAD' HWC_LAYOUT=1 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP16, WG, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp16"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_ERROR' HWC_LAYOUT=1 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP16)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP16, IG, HWC): ", 0, results_file)
    test_sequence_iterator += 1


    print("\n=====> ENTERING TEST SEQUENCE FOR FP32 CONV2D.. <=====\n")

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='FORWARD' HWC_LAYOUT=0 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP32, FW, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_GRAD' HWC_LAYOUT=0 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP32, WG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_ERROR' HWC_LAYOUT=0 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP32, IG, CHW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='FORWARD' HWC_LAYOUT=1 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP32, FW, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_GRAD' HWC_LAYOUT=1 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP32, WG, HWC): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_conv2d_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_ERROR' HWC_LAYOUT=1 MATMUL_TYPE="+str(Conv2D_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Conv2D (FP32, IG, HWC): ", 0, results_file)
    test_sequence_iterator += 1


    # TO DO
    #print("\n=====> ENTERING TEST SEQUENCE FOR DMA TRANSFER.. <=====\n")

    # TO DO
    # print("\n=====> ENTERING TEST SEQUENCE FOR IM2COL.. <=====\n")

    # TO DO
    #print("\n=====> ENTERING TEST SEQUENCE FOR LAYOUT CHANGE.. <=====\n")


    # print("\n=====> ENTERING TEST SEQUENCE FOR FP16 FULLY-CONNECTED.. <=====\n")

    # # Test settings
    # current_test_source_folder = test_cwd + "/test_linear_fp16"
    # cmd = "rm -rf BUILD/; make clean get_golden all run STEP='FORWARD' > log.txt"
    # # Automatic test sequence
    # ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    # os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    # p = subprocess.call(cmd, shell=True, timeout=timeout)
    # prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Linear (FP16, FW): ", 0, results_file)
    # test_sequence_iterator += 1

    # # Test settings
    # current_test_source_folder = test_cwd + "/test_linear_fp16"
    # cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_GRAD' > log.txt"
    # # Automatic test sequence
    # ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    # os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    # p = subprocess.call(cmd, shell=True, timeout=timeout)
    # prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Linear (FP16, WG): ", 0, results_file)
    # test_sequence_iterator += 1

    # # Test settings
    # current_test_source_folder = test_cwd + "/test_linear_fp16"
    # cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_ERROR' > log.txt"
    # # Automatic test sequence
    # ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    # os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    # p = subprocess.call(cmd, shell=True, timeout=timeout)
    # prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Linear (FP16, IG): ", 0, results_file)
    # test_sequence_iterator += 1


    print("\n=====> ENTERING TEST SEQUENCE FOR FP32 FULLY-CONNECTED.. <=====\n")

    # Test settings
    current_test_source_folder = test_cwd + "/test_linear_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='FORWARD' MATMUL_TYPE="+str(Linear_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Linear (FP32, FW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_linear_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_GRAD' MATMUL_TYPE="+str(Linear_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Linear (FP32, WG): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_linear_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD_ERROR' MATMUL_TYPE="+str(Linear_Opt_MM_FP32)+" > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") Linear (FP32, IG): ", 0, results_file)
    test_sequence_iterator += 1


    # TO DO
    #print("\n=====> ENTERING TEST SEQUENCE FOR LOSSES.. <=====\n")


    # TO DO
    #print("\n=====> ENTERING TEST SEQUENCE FOR MATMUL.. <=====\n")


    print("\n=====> ENTERING TEST SEQUENCE FOR FP32 MHSA.. <=====\n")

    # Test settings
    current_test_source_folder = test_cwd + "/test_mhsa_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='FORWARD' > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") MHSA (FP32, FW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_mhsa_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD' > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") MHSA (FP32, BW): ", 0, results_file)
    test_sequence_iterator += 1


    print("\n=====> ENTERING TEST SEQUENCE FOR FP32 RNN.. <=====\n")

    # Test settings
    current_test_source_folder = test_cwd + "/test_rnn_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='FORWARD' > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") RNN (FP32, FW): ", 0, results_file)
    test_sequence_iterator += 1

    # Test settings
    current_test_source_folder = test_cwd + "/test_rnn_fp32"
    cmd = "rm -rf BUILD/; make clean get_golden all run STEP='BACKWARD' > log.txt"
    # Automatic test sequence
    ci.copy_test_folder_ci(test_sequence_iterator, ci_cwd, current_test_source_folder)
    os.chdir(ci_cwd+"/temp/tests/ci_test_"+str(test_sequence_iterator))
    p = subprocess.call(cmd, shell=True, timeout=timeout)
    prof.extract_performance("\n\nTest ("+str(test_sequence_iterator)+") RNN (FP32, BW): ", 0, results_file)
    test_sequence_iterator += 1
    