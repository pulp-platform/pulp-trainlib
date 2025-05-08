APP = test_reduce_mean

# User settings
# Standard matmul arguments
DIMS = 1 7 800 2
REDUCE_AXIS = 2

NUM_CORES = 8
DATA_TYPE = 32 	# 32 for fp32, 16 for fp16
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS += main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c

APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -DCLUSTER -DFABRIC -O3 -g3
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DDATA_TYPE=$(DATA_TYPE)
APP_CFLAGS += -DPROF_NET

APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	rm -rf BUILD/
	python3 utils/GM.py --input_dims $(DIMS) --reduce_axis $(REDUCE_AXIS) --dtype $(DATA_TYPE)

include $(RULES_DIR)/pmsis_rules.mk
