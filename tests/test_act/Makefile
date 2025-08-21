APP = test_act

# ~~~~~~~~~~ User settings ~~~~~~~~~~
# Standard matmul arguments
IN_H?=4
IN_W?=4
IN_C?=8
VALUE?=0.05

# General arguments
DATA_TYPE?='FP16'	# FP32 or FP16
NUM_CORES?=8

# ~~~~~~~~~~ End of user settings ~~~~~~~~~~

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS += main.c net.c

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp16.c

APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -DCLUSTER -DFABRIC -O3 -g3
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DIN_H=$(IN_H)
APP_CFLAGS += -DIN_W=$(IN_W)
APP_CFLAGS += -DIN_C=$(IN_C)
APP_CFLAGS += -DVALUE=$(VALUE)
APP_CFLAGS += -DDATA_TYPE=$(DATA_TYPE)
#APP_CFLAGS += -DDEBUG

APP_LDFLAGS += -lm 

# STATISTICS
APP_CFLAGS += -DSTATS

get_golden:
	rm -rf BUILD/
	python3 ./utils/GM.py --in_c $(IN_C) --in_h $(IN_H) --in_w $(IN_W) --value $(VALUE) --data_type $(DATA_TYPE)

include $(RULES_DIR)/pmsis_rules.mk
