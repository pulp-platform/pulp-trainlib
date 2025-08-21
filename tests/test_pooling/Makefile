APP = test_pooling

# User settings
# Standard matmul arguments
IN_H?=6
IN_W?=6
IN_C?=1
KER_H?=6
KER_W?=6
H_STR?=1
W_STR?=1
VALUE?=0.5
# General arguments
NUM_CORES?=8
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS += main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_dw_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_optimizers_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_pooling_fp32.c

APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -DCLUSTER -DFABRIC -O3 -g3
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DIN_H=$(IN_H)
APP_CFLAGS += -DIN_W=$(IN_W)
APP_CFLAGS += -DIN_C=$(IN_C)
APP_CFLAGS += -DKER_H=$(KER_H)
APP_CFLAGS += -DKER_W=$(KER_W)
APP_CFLAGS += -DH_STR=$(H_STR)
APP_CFLAGS += -DW_STR=$(W_STR)
APP_CFLAGS += -DVALUE=$(VALUE)

APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	python3 ./utils/GM.py --in_c $(IN_C) --in_h $(IN_H) --in_w $(IN_W) --ker_h $(KER_H) --ker_w $(KER_W) --stride_h $(H_STR) --stride_w $(W_STR) --value $(VALUE)

include $(RULES_DIR)/pmsis_rules.mk
