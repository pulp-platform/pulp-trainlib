APP = rng

# User code
NUM_CORES?=8
PROBABILITY?=0.23
SEED?=0
SIZE?=100
USE_MASK?=0
DATA_TYPE?=float		# 'float' or 'fp16'
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
APP_CFLAGS += -DPROBABILITY=$(PROBABILITY)
APP_CFLAGS += -DSEED=$(SEED) #sneed
APP_CFLAGS += -DSIZE=$(SIZE)
APP_CFLAGS += -DUSE_MASK=$(USE_MASK)
APP_CFLAGS += -DDATA_TYPE=$(DATA_TYPE)
APP_LDFLAGS += -lm


# STATISTICS
APP_CFLAGS += -DSTATS

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_random.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_dropout_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_dropout_fp16.c

include $(RULES_DIR)/pmsis_rules.mk

get_golden:
	python3 utils/GM.py --in_size $(SIZE) --type $(DATA_TYPE) --prob $(PROBABILITY)
