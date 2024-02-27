# enable app support
EXTRA_CFLAGS += -Isrc  -w
EXTRA_CFLAGS += -I$(PWD)/include

EXTRA_LDFLAGS += -specs=nosys.specs -Wl

CRAZYFLIE_BASE := ../../../crazyflie-firmware

OOT_CONFIG := $(PWD)/app-config

include $(CRAZYFLIE_BASE)/tools/make/oot.mk

