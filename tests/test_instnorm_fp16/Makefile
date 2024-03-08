APP = test_instnorm_fp16

CI?=16
HI?=8
WI?=8
KER?=1
NUM_CORES?=8
HWC?=0
DEBUG_INFO?=0
STEP?='FORWARD'			# 'FORWARD' or 'BACKWARD_GRAD' or 'BACKWARD_ERROR'
DATA_TYPE?='FLOAT16'
EPOCHS?=0

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS += main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_instnorm_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_instnorm_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_optimizers_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_optimizers_fp16.c

APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -DCLUSTER -DFABRIC -O3 -g3
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DOPTIMIZE



APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	python3 ./utils/GM.py -CI ${CI} -HI ${HI} -WI ${WI} -NUM_CORES ${NUM_CORES} -STEP ${STEP} -EPOCHS ${EPOCHS}

include $(RULES_DIR)/pmsis_rules.mk


