APP = blocktransposer

# User code
NUM_CORES?=8
CH_IN?=2
CH_OUT?=3
HK?=2
WK?=2
HWC_LAY?=1		# =0 use CHW layout, =1 use HWC layout for the weights
APP_CFLAGS += -DPRINT_MATS
# End of user code


TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c
#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3 -mno-memcpy
APP_CFLAGS += -DFABRIC
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -mhwloopalign
APP_CFLAGS += -DTin_Cout=$(CH_OUT)
APP_CFLAGS += -DTin_Cin=$(CH_IN)
APP_CFLAGS += -DTin_Hk=$(HK)
APP_CFLAGS += -DTin_Wk=$(WK)
APP_CFLAGS += -DHWC_LAYOUT=$(HWC_LAY)
APP_LDFLAGS += -lm

# STATISTICS
APP_CFLAGS += -DSTATS

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c

include $(RULES_DIR)/pmsis_rules.mk
