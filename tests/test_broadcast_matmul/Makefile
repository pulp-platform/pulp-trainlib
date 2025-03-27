APP = test_broadcast_matmul

# User settings
# Standard matmul arguments
DIMS_1 = 2 1 3 2 5
DIMS_2 = 5 1 5 3

NUM_CORES = 8
DATA_TYPE = 16 	# 32 for fp32, 16 for fp16
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS += main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c

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
	python3 utils/GM.py --dims_1 $(DIMS_1) --dims_2 $(DIMS_2) --dtype $(DATA_TYPE)

include $(RULES_DIR)/pmsis_rules.mk
