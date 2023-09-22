APP = padder

# User code
NUM_CORES?=8
CH_IN?=2
H_IN?=3
W_IN?=3
LPAD?=1
RPAD?=1
UPAD?=1
DPAD?=1
HWC_LAY?=0		# =0 use CHW layout, =1 use HWC layout for the weights
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
APP_CFLAGS += -DTin_C=$(CH_IN)
APP_CFLAGS += -DTin_H=$(H_IN)
APP_CFLAGS += -DTin_W=$(W_IN)
APP_CFLAGS += -DLPAD=$(LPAD)
APP_CFLAGS += -DRPAD=$(RPAD)
APP_CFLAGS += -DUPAD=$(UPAD)
APP_CFLAGS += -DDPAD=$(DPAD)
APP_CFLAGS += -DHWC_LAYOUT=$(HWC_LAY)
APP_LDFLAGS += -lm

# STATISTICS
APP_CFLAGS += -DSTATS

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c

include $(RULES_DIR)/pmsis_rules.mk
