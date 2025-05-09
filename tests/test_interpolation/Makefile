APP = interpolation_fp32

# User code
NUM_CORES?=1
DATA_TYPE?='float'	# 'float' or 'bfloat16'
INTP_TYPE?=0 		# 0='NEAREST', 1='BILINEAR' 
CH?=1
IN_H?=8
IN_W?=8
OUT_H?=48
OUT_W?=48
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
APP_CFLAGS += -DCH=$(CH)
APP_CFLAGS += -DIN_H=$(IN_H)
APP_CFLAGS += -DIN_W=$(IN_W)
APP_CFLAGS += -DOUT_H=$(OUT_H)
APP_CFLAGS += -DOUT_W=$(OUT_W)
APP_CFLAGS += -DINTP_TYPE=$(INTP_TYPE)
APP_LDFLAGS += -lm

# STATISTICS
APP_CFLAGS += -DSTATS

APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_interpolation_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_interpolation_fp16.c

include $(RULES_DIR)/pmsis_rules.mk

get_golden:
	python3 utils/GM.py --in_ch $(CH) --in_height $(IN_H) --in_width $(IN_W) --out_height $(OUT_H) --out_width $(OUT_W) --type $(DATA_TYPE)
