APP = dw_pw_fp16

# User settings
# Depthwise
IMAGE_W?=5
IMAGE_H?=25
DW_KER_W?=3
DW_KER_H?=3
DW_IN_CH?=16
# Pointwise
PW_OUT_CH?=1
# Depthwise Padding and stride
LPAD?=1
RPAD?=1
UPAD?=1
DPAD?=1
HSTR?=1
WSTR?=1
# Others
NUM_CORES?=8
STEP?='DW_FORWARD' # Steps: 'DW_FORWARD', 'DW_BACKWARD_GRAD', 'DW_BACKWARD_ERROR', 'PW_FORWARD', 'PW_BACKWARD_GRAD', 'PW_BACKWARD_ERROR'
#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=0
NUM_MATMULS?=6	# When profiling with multiple matmul algorithms
NUM_SIZES?=3	# When profiling multiple sizes of the network
#APP_CFLAGS += -DCHECK_PRINT
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

BYPASS?=1
DATA_TYPE?='fp16'
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
APP_LDFLAGS += -lm

# STATISTICS
APP_CFLAGS += -DSTATS

# Sources
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_dw_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c

get_golden:
	python3.6 ./utils/GM.py --step ${STEP} --image_width ${IMAGE_W} --image_height ${IMAGE_H} --ker_width ${DW_KER_W} --ker_height ${DW_KER_H} --ch_in_dw ${DW_IN_CH} --ch_out_pw ${PW_OUT_CH} --bypass_size_automation ${BYPASS}

profile_all_optim:
	python3.6 ./utils/profile_optimized.py --num_matmuls ${NUM_MATMULS} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --image_width ${IMAGE_W} --image_height ${IMAGE_H} --ker_width ${DW_KER_W} --ker_height ${DW_KER_H} --ch_in_dw ${DW_IN_CH} --ch_out_pw ${PW_OUT_CH} 

profile_all_sizes:
	python3.6 ./utils/profile_sizes.py --num_sizes ${NUM_SIZES} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --matmul_type ${MATMUL_TYPE}

include $(RULES_DIR)/pmsis_rules.mk
