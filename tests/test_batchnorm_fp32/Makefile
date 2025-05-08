APP = test_batchnorm_fp32

# User Section
CI?=4
HI?=4
WI?=4

BATCH_SIZE?=2
NUM_CORES?=2
HWC?=0

STEP?='FORWARD'			# 'FORWARD' or 'BACKWARD_GRAD' or 'BACKWARD_ERROR'
# End of User Section

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_batchnorm_fp32.c
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
APP_CFLAGS += -DDEBUG

APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	python3 ./utils/GM.py -CI ${CI} -HI ${HI} -WI ${WI} -NUM_CORES ${NUM_CORES} -STEP ${STEP} -BATCH_SIZE ${BATCH_SIZE}

include $(RULES_DIR)/pmsis_rules.mk
