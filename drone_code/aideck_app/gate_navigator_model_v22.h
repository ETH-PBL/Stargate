#ifndef __gate_navigator_model_v22_H__
#define __gate_navigator_model_v22_H__

#define __PREFIX(x) gate_navigator_model_v22 ## x
// Include basic GAP builtins defined in the Autotiler
#include "Gap.h"

#ifdef __EMUL__
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/param.h>
#include <string.h>
#endif

extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE gate_navigator_model_v22_L3_Flash;
#endif
