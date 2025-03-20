APP = transposer

# User code
#DIMS_1 = 10 1 30
#DIMS_2 = 20 1

DIMS_1 = 2 1 5
DIMS_2 = 3 1

NUM_CORES = 1
DATA_TYPE = 32

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
APP_CFLAGS += -DDATA_TYPE=$(DATA_TYPE)
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

get_golden:
	rm -rf BUILD/
	python3 utils/GM.py --dims_1 $(DIMS_1) --dims_2 $(DIMS_2) --dtype $(DATA_TYPE)
