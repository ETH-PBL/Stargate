#ifndef __main_H__
#define __main_H__

#define __PREFIX1(x) gate_navigator_model_v22 ## x
#define __PREFIX2(x) gate_classifier_model_v51 ## x
// Include basic GAP builtins defined in the Autotiler
#include "Gap.h"

extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE gate_navigator_model_v22_L3_Flash;
extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE gate_classifier_model_v51_L3_Flash;
#endif