#!/bin/bash
MYHOME=/home/lanmei
# export PATH=/usr/pack/gcc-9.2.0-af/linux-x64/bin:$PATH #??
# export
# LD_LIBRARY_PATH=/usr/pack/gcc-9.2.0-af/linux-x64/lib64/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/pack/gcc-9.2.0-af/linux-x64/lib/:$LD_LIBRARY_PATH
# export
# LD_LIBRARY_PATH=/usr/pack/graphicsmagick-1.3.36-kgf/lib/:$LD_LIBRARY_PATH
export CC=gcc #-11
export CXX=g++ #-11
export GAP_RISCV_GCC_TOOLCHAIN=$MYHOME/gap_riscv_toolchain
#cd $MYHOME/gap_sdk_private
source $MYHOME/gap_sdk_private/configs/gap9_evk_audio.sh
# Only if you wanna build the entire sdk
# At least cmake 3.19
# make clean VERBOSE=1 all
# make openocd.all -j8
# Hello world test
cd $MYHOME/pulp-trainlib_ori/tests/test_linear_fp32 #&& make all run platform=gvsoc
#cd $MYHOME/pulp-trainlib/tests/new_gap_test_fcn_mibmi_fp32
#cmake -B build
#cmake --build build -t run
#make clean get_golden all run platform=gvsoc # -j 10
make clean profile_all_optim all run platform=gvsoc
#make all run platform=gvsoc -j 10
