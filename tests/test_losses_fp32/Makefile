APP = test_loss

# User settings
# Standard matmul arguments
OUT_SIZE?=10
VALUE?=0.5
LOSS_FN?=3		# Available options: 0='L1Loss', 1='MSE', 2='CrossEntropy', 3='berHuLoss'
# General arguments
NUM_CORES?=1
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS += main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -DCLUSTER -DFABRIC -O3 -g3
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DWGT_SIZE=$(OUT_SIZE)
APP_CFLAGS += -DLOSS_FN=$(LOSS_FN)

APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	python3 ./utils/GM.py --out_size $(OUT_SIZE) --value $(VALUE) --loss_fn $(LOSS_FN)

include $(RULES_DIR)/pmsis_rules.mk
