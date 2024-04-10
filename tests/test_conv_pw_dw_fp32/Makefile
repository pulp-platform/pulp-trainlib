APP = dw_pw_fp32

# User settings
# Depthwise
IMAGE_W?=8
IMAGE_H?=8
DW_KER_W?=3
DW_KER_H?=3
DW_IN_CH?=8
# Pointwise
PW_OUT_CH?=8
# Depthwise Padding and stride
LPAD?=0
RPAD?=0
UPAD?=0
DPAD?=0
HSTR?=1
WSTR?=1
# Others
NUM_CORES?=1
STEP?='DW_FORWARD' 	# Steps: 'DW_FORWARD', 'DW_BACKWARD_GRAD', 'DW_BACKWARD_ERROR', 'PW_FORWARD', 'PW_BACKWARD_GRAD', 'PW_BACKWARD_ERROR'
HWC_layout?=0				# Choose if data layout is CHW (=0) or HWC (=1)
# Optimization
APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=0
NUM_MATMULS?=24	# When profiling with multiple matmul algorithms
NUM_SIZES?=3	# When profiling multiple sizes of the network
APP_CFLAGS += -DCHECK_PRINT
#APP_CFLAGS += -DDEBUG_DW
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

DATA_TYPE?='fp32'
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3 -mno-memcpy
APP_CFLAGS += -DFABRIC
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DMEMOCC_COMP
APP_CFLAGS += -DMATMUL_TYPE=${MATMUL_TYPE}
APP_CFLAGS += -mhwloopalign
APP_CFLAGS += -DLPAD=$(LPAD)
APP_CFLAGS += -DRPAD=$(RPAD)
APP_CFLAGS += -DUPAD=$(UPAD)
APP_CFLAGS += -DDPAD=$(DPAD)
APP_CFLAGS += -DHWC_LAYOUT=$(HWC_layout)
APP_LDFLAGS += -lm

# STATISTICS
APP_CFLAGS += -DSTATS

# Sources
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_dw_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_naive_fp32.c


get_golden:
	python3 ./utils/GM.py --step ${STEP} --image_width ${IMAGE_W} --image_height ${IMAGE_H} --ker_width ${DW_KER_W} --ker_height ${DW_KER_H} --ch_in_dw ${DW_IN_CH} --ch_out_pw ${PW_OUT_CH} --pad_h ${UPAD} --pad_w ${LPAD} --HWC_layout ${HWC_layout}

profile_all_optim:
	python3 ./utils/profile_optimized.py --num_matmuls ${NUM_MATMULS} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --image_width ${IMAGE_W} --image_height ${IMAGE_H} --ker_width ${DW_KER_W} --ker_height ${DW_KER_H} --ch_in_dw ${DW_IN_CH} --ch_out_pw ${PW_OUT_CH} 

profile_all_sizes:
	python3 ./utils/profile_sizes.py --num_sizes ${NUM_SIZES} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --matmul_type ${MATMUL_TYPE}

include $(RULES_DIR)/pmsis_rules.mk
