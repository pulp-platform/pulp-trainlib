APP = layernorm_fp32

# User code
NUM_CORES?=8
DATA_TYPE?=fp32		# 'fp32'

INPUT_WIDTH?=16
INPUT_HEIGHT?=16
# End of user code

TASK_NAME=sst-2
TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3 -mno-memcpy
APP_CFLAGS += -DFABRIC
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DN_HEADS=$(N_HEADS)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -mhwloopalign
APP_LDFLAGS += -lm

APP_CFLAGS += -DTILE_H=$(TILE_H)
APP_CFLAGS += -DTILE_W=$(TILE_W)
APP_CFLAGS += -DTILE_DIM=$(TILE_DIM)

APP_CFLAGS += -DOPTIMIZE
APP_CFLAGS += -DMATMUL_TYPE=${MATMUL_TYPE}

# STATISTICS
APP_CFLAGS += -DSTATS

# =============== SOURCES ===============
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_layernorm_fp32.c

include $(RULES_DIR)/pmsis_rules.mk

get_golden:
	rm -rf BUILD/
	python3 utils/GM.py --data_type $(DATA_TYPE) --input_shape_height $(INPUT_HEIGHT) --input_shape_width $(INPUT_WIDTH)
