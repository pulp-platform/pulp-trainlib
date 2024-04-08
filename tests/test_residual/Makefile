APP = test_residual

CI?=64
HI?=56
WI?=56
KER?=1
NUM_CORES?=8
HWC?=0
DEBUG_INFO?=0
STEP?='BACKWARD'
DATA_TYPE?='FLOAT32'
USE_IM2COL?=1
USE_DMA?=0
MATMUL_TYPE?=0

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS += main.c net.c


APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_residual_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_residual_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp16.c


APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -DCLUSTER -DFABRIC -O3 -g3
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DOPTIMIZE



APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	python3 ./utils/GM.py -CI ${CI} -HI ${HI} -WI ${WI} -KER ${KER} -NUM_CORES ${NUM_CORES} -HWC ${HWC} -DEBUG_INFO ${DEBUG_INFO} -STEP ${STEP} -DATA_TYPE ${DATA_TYPE} -USE_IM2COL ${USE_IM2COL} -USE_DMA ${USE_DMA} -MATMUL_TYPE ${MATMUL_TYPE}

include $(RULES_DIR)/pmsis_rules.mk


