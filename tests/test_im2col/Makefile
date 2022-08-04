APP = im2col_fp32

# User settings
DATA_BITS?=16				# 32=FP32, 16=FP16
NUM_CORES?=1
MOD?=1 						# 0=Forward/WeightGrad, 1=InGrad
DMA_ENABLE?=1					# Use Cluster DMA to move data or not
# Layer padding and stride
LPAD?=0
RPAD?=0
UPAD?=0
DPAD?=0
HSTR?=1
WSTR?=1
# Layer sizes
Tin_H_l1?=8
Tin_W_l1?=8
Tin_C_l1?=16
Tker_W_l1?=3
Tout_C_l1?=16
#APP_CFLAGS += -DPRINT_OUTPUT
#STEP?='DW_FORWARD' # Steps: 'DW_FORWARD', 'DW_BACKWARD_GRAD', 'DW_BACKWARD_ERROR', 'PW_FORWARD', 'PW_BACKWARD_GRAD', 'PW_BACKWARD_ERROR'
# End of user settings

TRAIN_LIB=../../lib
TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources
APP_SRCS = main.c net.c

APP_CFLAGS += -I. -I$(TRAIN_LIB)/include
APP_CFLAGS += -O3 -g3 -mno-memcpy
APP_CFLAGS += -DFABRIC
APP_CFLAGS += -DCLUSTER
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
APP_CFLAGS += -DPROF_NET
APP_CFLAGS += -DDATA_BITS=$(DATA_BITS)
APP_CFLAGS += -DMOD=$(MOD)
APP_CFLAGS += -DDMA_ENABLE=$(DMA_ENABLE)
APP_CFLAGS += -DLPAD=$(LPAD)
APP_CFLAGS += -DRPAD=$(RPAD)
APP_CFLAGS += -DUPAD=$(UPAD)
APP_CFLAGS += -DDPAD=$(DPAD)
APP_CFLAGS += -DHSTR=$(HSTR)
APP_CFLAGS += -DWSTR=$(WSTR)
APP_CFLAGS += -DTin_H_l1=$(Tin_H_l1)
APP_CFLAGS += -DTin_W_l1=$(Tin_W_l1)
APP_CFLAGS += -DTin_C_l1=$(Tin_C_l1)
APP_CFLAGS += -DTker_W_l1=$(Tker_W_l1)
APP_CFLAGS += -DTout_C_l1=$(Tout_C_l1)
#APP_CFLAGS += -DDEBUG
APP_CFLAGS += -mhwloopalign
APP_LDFLAGS += -lm

# STATISTICS
APP_CFLAGS += -DSTATS

# Sources
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c
APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c

#get_golden:
#	python3.6 ./utils/GM.py --step ${STEP}

include $(RULES_DIR)/pmsis_rules.mk
