APP = transposer

# User code
NUM_CORES?=1
IN_N?=125
IN_M?=16
#APP_CFLAGS += -DPRINT_MATS
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
APP_CFLAGS += -DTin_N=$(IN_N)
APP_CFLAGS += -DTin_M=$(IN_M)
APP_LDFLAGS += -lm

# STATISTICS
APP_CFLAGS += -DSTATS

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c

include $(RULES_DIR)/pmsis_rules.mk
