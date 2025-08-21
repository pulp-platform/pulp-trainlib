APP = conv2d_fp16

# User settings
IMAGE_H?=10
IMAGE_W?=10
KER_H?=3
KER_W?=3
IN_CH?=8
OUT_CH?=8
PAD_L?=0
PAD_R?=0
PAD_U?=0
PAD_D?=0
STRIDE_H?=1
STRIDE_W?=1
NUM_CORES?=8
STEP?='FORWARD' # options: // FORWARD, BACKWARD_GRAD, BACKWARD_ERROR
#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -DOPTIMIZE
MATMUL_TYPE?=0
NUM_MATMULS?=7		# When profiling with multiple matmul algorithms
NUM_SIZES?=3		# When profiling multiple sizes of the network
IM2COL?=1			# Selects to use or not the im2col+matmul (0=don't, 1=use)
DMA?=0				# In case IM2COL+MM are used, select to manage IM2COL using DMA (input data/output gradient need to be in L2, im2col buffer in L1)
HWC_LAYOUT?=0		# Choose if data layout is CHW (=0) or HWC (=1)
USE_BIASES?=0		# Choose whether to include biases or not
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

DATA_TYPE?='fp16'
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3
APP_CFLAGS += -DFABRIC
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
#APP_CFLAGS += -mhwloopalign
APP_CFLAGS += -DMATMUL_TYPE=${MATMUL_TYPE}
APP_CFLAGS += -DMEMOCC_COMP
APP_CFLAGS += -DIM2COL=$(IM2COL)
APP_CFLAGS += -DUSE_BIAS=$(USE_BIASES)
APP_CFLAGS += -DPAD_R=$(PAD_R)
APP_CFLAGS += -DPAD_L=$(PAD_L)
APP_CFLAGS += -DPAD_U=$(PAD_U)
APP_CFLAGS += -DPAD_D=$(PAD_D)
APP_CFLAGS += -DSTRIDE_H=$(STRIDE_H)
APP_CFLAGS += -DSTRIDE_W=$(STRIDE_W)
APP_CFLAGS += -DDMA=$(DMA)
APP_CFLAGS += -DHWC_LAYOUT=$(HWC_LAYOUT)
APP_LDFLAGS += -lm


# STATISTICS
APP_CFLAGS += -DSTATS

# Sources
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_naive_fp16.c

get_golden:
	python3 ./utils/GM.py --step ${STEP} --image_width ${IMAGE_W} --image_height ${IMAGE_H} --ker_width ${KER_W} --ker_height ${KER_H} --ch_in ${IN_CH} --ch_out ${OUT_CH} --w_pad ${PAD_L} --h_pad ${PAD_U} --h_str ${STRIDE_H} --w_str ${STRIDE_W} --HWC ${HWC_LAYOUT} --USE_BIASES ${USE_BIASES}

profile_all_optim:
	python3 ./utils/profile_optimized.py --num_matmuls ${NUM_MATMULS} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --image_width ${IMAGE_W} --image_height ${IMAGE_H} --ker_width ${KER_W} --ker_height ${KER_H} --ch_in ${IN_CH} --ch_out ${OUT_CH} --USE_BIASES ${USE_BIASES}

profile_all_sizes:
	python3 ./utils/profile_sizes.py --num_sizes ${NUM_SIZES} --step ${STEP} --cores ${NUM_CORES} --data_type ${DATA_TYPE} --matmul_type ${MATMUL_TYPE} --use_biases ${USE_BIASES}

include $(RULES_DIR)/pmsis_rules.mk
