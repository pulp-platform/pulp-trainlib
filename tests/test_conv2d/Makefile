APP = conv2d_fp32

# User settings
IMAGE_H?=34
IMAGE_W?=8
KER_H?=10
KER_W?=4
IN_CH?=1
OUT_CH?=4
NUM_CORES?=8
STEP?='FORWARD' # options: // FORWARD, BACKWARD_GRAD, BACKWARD_ERROR
#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=0
NUM_MATMULS?=24		# When profiling with multiple matmul algorithms
NUM_SIZES?=3		# When profiling multiple sizes of the network
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

DATA_TYPE?='fp32'
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3
APP_CFLAGS += -DFABRIC
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -mhwloopalign
APP_CFLAGS += -DMATMUL_TYPE=${MATMUL_TYPE}
APP_CFLAGS += -DMEMOCC_COMP
APP_LDFLAGS += -lm


# STATISTICS
APP_CFLAGS += -DSTATS

# Sources
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c

get_golden:
	python3.6 ./utils/GM.py --step ${STEP} --image_width ${IMAGE_W} --image_height ${IMAGE_H} --ker_width ${KER_W} --ker_height ${KER_H} --ch_in ${IN_CH} --ch_out ${OUT_CH}

profile_all_optim:
	python3.6 ./utils/profile_optimized.py --num_matmuls ${NUM_MATMULS} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --image_width ${IMAGE_W} --image_height ${IMAGE_H} --ker_width ${KER_W} --ker_height ${KER_H} --ch_in ${IN_CH} --ch_out ${OUT_CH}

profile_all_sizes:
	python3.6 ./utils/profile_sizes.py --num_sizes ${NUM_SIZES} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --matmul_type ${MATMUL_TYPE}

include $(RULES_DIR)/pmsis_rules.mk
