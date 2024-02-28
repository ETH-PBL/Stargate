#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"

#include "CNN_Copy_Generators.h"





void gate_navigator_model_v22Model(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 3, "Gap.h", "gate_navigator_model_v22.h", "CNN_BasicKernels_SQ8.h");
    SetGeneratedFilesNames("gate_navigator_model_v22Kernels.c", "gate_navigator_model_v22Kernels.h");
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CVAR_NAME, AT_OPT_VAL("AT_Navigation_Monitor"));
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_CVAR_NAME, AT_OPT_VAL("AT_Navigation_Nodes"));
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS_CVAR_NAME, AT_OPT_VAL("AT_Navigation_Op"));
    //AT_SetGraphCtrl(AT_GRAPH_DUMP_TENSOR, AT_OPT_VAL(7));

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "gate_navigator_model_v22_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "gate_navigator_model_v22_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "gate_navigator_model_v22_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "gate_navigator_model_v22_L3_Flash", "gate_navigator_model_v22_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();


    CNN_GenControl_T gen_ctrl_S6_Conv2d_4x1x5x5;
    CNN_InitGenCtrl(&gen_ctrl_S6_Conv2d_4x1x5x5);
    CNN_SetGenCtrl(&gen_ctrl_S6_Conv2d_4x1x5x5, "PADTYPE", AT_OPT_VAL(3));
    // generator for CONV_2D_0_1
    CNN_ConvolutionPoolAct_SQ8("NAV_S6_Conv2d_4x1x5x5", &gen_ctrl_S6_Conv2d_4x1x5x5,
                               4, 1,
                               1, 4, 168, 168,
                               KOP_CONV, 5, 5, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    // generator for CONV_2D_0_1_activation
    CNN_PoolAct_SQ8("NAV_S7_Act_Relu6", 0,
                    4, 84, 84,
                    KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                    KOP_RELU);
    
    // generator for MAX_POOL_2D_0_2
    CNN_PoolAct_SQ8("NAV_S8_MaxPool_2x2", 0,
                    4, 84, 84,
                    KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                    KOP_NONE);
    
    CNN_GenControl_T gen_ctrl_S11_Conv2d_4x4x3x3;
    CNN_InitGenCtrl(&gen_ctrl_S11_Conv2d_4x4x3x3);
    CNN_SetGenCtrl(&gen_ctrl_S11_Conv2d_4x4x3x3, "PADTYPE", AT_OPT_VAL(1));
    // generator for CONV_2D_0_3
    CNN_ConvolutionPoolAct_SQ8("NAV_S11_Conv2d_4x4x3x3", &gen_ctrl_S11_Conv2d_4x4x3x3,
                               4, 1,
                               4, 4, 42, 42,
                               KOP_CONV, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    // generator for CONV_2D_0_3_activation
    CNN_PoolAct_SQ8("NAV_S12_Act_Relu6", 0,
                    4, 21, 21,
                    KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                    KOP_RELU);
    
    // generator for CONV_2D_0_4
    CNN_ConvolutionPoolAct_SQ8("NAV_S15_Conv2d_4x4x3x3", 0,
                               4, 1,
                               4, 4, 21, 21,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    // generator for CONV_2D_0_4_activation
    CNN_PoolAct_SQ8("NAV_S16_Act_Relu6", 0,
                    4, 21, 21,
                    KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                    KOP_RELU);
    
    // generator for CONV_2D_0_6
    CNN_ConvolutionPoolAct_SQ8("NAV_S19_Conv2d_4x1x3x3", 0,
                               4, 1,
                               1, 4, 21, 21,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    // generator for CONV_2D_0_6_activation
    CNN_PoolAct_SQ8("NAV_S20_Act_Relu6", 0,
                    4, 21, 21,
                    KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                    KOP_RELU);
    
    // generator for CONV_2D_0_8
    CNN_ConvolutionPoolAct_SQ8("NAV_S24_Conv2d_16x8x3x3", 0,
                               4, 1,
                               8, 16, 21, 21,
                               KOP_CONV, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    // generator for CONV_2D_0_8_activation
    CNN_PoolAct_SQ8("NAV_S25_Act_Relu6", 0,
                    16, 11, 11,
                    KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                    KOP_RELU);
    
    // generator for CONV_2D_0_9
    CNN_ConvolutionPoolAct_SQ8("NAV_S28_Conv2d_16x16x3x3", 0,
                               4, 1,
                               16, 16, 11, 11,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    // generator for CONV_2D_0_9_activation
    CNN_PoolAct_SQ8("NAV_S29_Act_Relu6", 0,
                    16, 11, 11,
                    KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                    KOP_RELU);
    
    // generator for CONV_2D_0_10
    CNN_ConvolutionPoolAct_SQ8("NAV_S32_Conv2d_32x16x3x3", 0,
                               4, 1,
                               16, 32, 11, 11,
                               KOP_CONV, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    // generator for CONV_2D_0_10_activation
    CNN_PoolAct_SQ8("NAV_S33_Act_Relu6", 0,
                    32, 6, 6,
                    KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                    KOP_RELU);
    
    // generator for CONV_2D_0_11
    CNN_ConvolutionPoolAct_SQ8("NAV_S36_Conv2d_32x32x3x3", 0,
                               4, 1,
                               32, 32, 6, 6,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_NONE);
    
    // generator for CONV_2D_0_11_activation
    CNN_PoolAct_SQ8("NAV_S37_Act_Relu6", 0,
                    32, 6, 6,
                    KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                    KOP_RELUM);
    
    // generator for FULLY_CONNECTED_0_13
    CNN_LinearAct_SQ8("NAV_S40_Linear_1x1152", 0,
                      4, 1,
                      1152, 1,
                      KOP_LINEAR, KOP_NONE);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("gate_navigator_model_v22CNN",
        /* Arguments either passed or globals */
            CArgs(57,
                TCArgInfo("signed char * __restrict__", "Input_1", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Input_2", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Conv2dlastvaluequantfakequantw", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Conv2dlastvaluequantfakequantw.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Fusedbatchnormv3biasaddreadvar", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Fusedbatchnormv3biasaddreadvar.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S6_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S6_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S6_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S6_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S6_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S6_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.04711 out: 0.04711  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0
                TCArgInfo("signed char * __restrict__", "S7_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S7_Infos.tensor", 1, 1, 8, 0)),
                // no activation ACTSCALE: [1] ACTSCALEN: [0]
                TCArgInfo("signed char * __restrict__", "S8_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S8_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv2dlastvaluequantfakequantw_5c3400a3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Conv2dlastvaluequantfakequantw_5c3400a3.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Fusedbatchnormv3biasaddreadvar_fb958f7e", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Fusedbatchnormv3biasaddreadvar_fb958f7e.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S11_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S11_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S11_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S11_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S11_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S11_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.04634 out: 0.04634  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0
                TCArgInfo("signed char * __restrict__", "S12_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S12_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv2dlastvaluequantfakequantw_a0d91a2a", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Conv2dlastvaluequantfakequantw_a0d91a2a.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Fusedbatchnormv3biasaddreadvar_226e0269", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Fusedbatchnormv3biasaddreadvar_226e0269.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S15_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S15_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S15_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S15_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S15_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S15_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.04041 out: 0.04041  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0
                TCArgInfo("signed char * __restrict__", "S16_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S16_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv2dlastvaluequantfakequantw_44a058cb", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Conv2dlastvaluequantfakequantw_44a058cb.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Fusedbatchnormv3biasaddreadvar_5b219a5a", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Fusedbatchnormv3biasaddreadvar_5b219a5a.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S19_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S19_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S19_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S19_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S19_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S19_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.04041 out: 0.04041  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0
                TCArgInfo("signed char * __restrict__", "S20_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S20_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv2dlastvaluequantfakequantw_da8d4c27", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Conv2dlastvaluequantfakequantw_da8d4c27.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Fusedbatchnormv3biasaddreadvar_ebf0beac", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Fusedbatchnormv3biasaddreadvar_ebf0beac.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S24_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S24_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S24_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S24_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S24_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S24_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.04735 out: 0.04735  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0
                TCArgInfo("signed char * __restrict__", "S25_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S25_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv2dlastvaluequantfakequantw_ff42dfe1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Conv2dlastvaluequantfakequantw_ff42dfe1.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Fusedbatchnormv3biasaddreadvar_b3e609d3", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Fusedbatchnormv3biasaddreadvar_b3e609d3.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S28_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S28_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S28_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S28_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S28_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S28_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.04430 out: 0.04430  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0
                TCArgInfo("signed char * __restrict__", "S29_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S29_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv2dlastvaluequantfakequantw_7679cecd", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Conv2dlastvaluequantfakequantw_7679cecd.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Fusedbatchnormv3biasaddreadvar_cc476866", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Fusedbatchnormv3biasaddreadvar_cc476866.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S32_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S32_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S32_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S32_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S32_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S32_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.04735 out: 0.04735  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0
                TCArgInfo("signed char * __restrict__", "S33_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S33_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Conv2dlastvaluequantfakequantw_699b5be9", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Conv2dlastvaluequantfakequantw_699b5be9.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Fusedbatchnormv3biasaddreadvar_d9ff256d", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Fusedbatchnormv3biasaddreadvar_d9ff256d.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S36_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S36_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S36_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S36_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S36_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S36_Infos.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0
                TCArgInfo("signed char * __restrict__", "S37_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S37_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Matmullastvaluequantfakequantw", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Matmullastvaluequantfakequantw.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Biasaddreadvariableop", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/Biasaddreadvariableop.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S40_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S40_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S40_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S40_Mul_shift.tensor", 1, 1, 8, 0)),
                // no activation BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S40_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/navigation_tensors/S40_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(16,
            TCArgInfo("signed char * __restrict__", "S6_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S7_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S8_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S11_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S12_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S15_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S19_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S21_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S24_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S25_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S28_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S29_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S32_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S33_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S36_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S37_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    AddStackedTensors("S21_Output", 2, "S16_Output", "S20_Output");

    // Node S6_Conv2d_4x1x5x5 inq -8.63<(i8-0.00)*0.06740149<8.56 forced weightsq chan<(i8-0.00)*chan<chan outq -6.03<(i8-0.00)*0.04710625<5.98 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("NAV_S6_Conv2d_4x1x5x5",
        Bindings(7,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "Conv2dlastvaluequantfakequantw", 0),
            GNodeArg(GNA_IN, "Fusedbatchnormv3biasaddreadvar", 0),
            GNodeArg(GNA_OUT, "S6_Output", 0),
            GNodeArg(GNA_IN, "S6_Mul_scale", 0),
            GNodeArg(GNA_IN, "S6_Mul_shift", 0),
            GNodeArg(GNA_IN, "S6_Infos", 0)
        )
    );
    // Node CONV_2D_0_1_activation inq -6.03<(i8-0.00)*0.04710625<5.98 forced outq -6.03<(i8-0.00)*0.04710625<5.98 forced
    AddNode("NAV_S7_Act_Relu6",
        Bindings(3,
            GNodeArg(GNA_IN, "S6_Output", 0),
            GNodeArg(GNA_OUT, "S7_Output", 0),
            GNodeArg(GNA_IN, "S7_Infos", 0)
        )
    );
    // Node MAX_POOL_2D_0_2 inq -6.03<(i8-0.00)*0.04710625<5.98 forced outq -6.03<(i8-0.00)*0.04710625<5.98 forced
    AddNode("NAV_S8_MaxPool_2x2",
        Bindings(3,
            GNodeArg(GNA_IN, "S7_Output", 0),
            GNodeArg(GNA_OUT, "S8_Output", 0),
            GNodeArg(GNA_IN, "S8_Infos", 0)
        )
    );
    // Node S11_Conv2d_4x4x3x3 inq -6.03<(i8-0.00)*0.04710625<5.98 forced weightsq chan<(i8-0.00)*chan<chan outq -5.93<(i8-0.00)*0.04633911<5.89 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("NAV_S11_Conv2d_4x4x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S8_Output", 0),
            GNodeArg(GNA_IN, "Conv2dlastvaluequantfakequantw_5c3400a3", 0),
            GNodeArg(GNA_IN, "Fusedbatchnormv3biasaddreadvar_fb958f7e", 0),
            GNodeArg(GNA_OUT, "S11_Output", 0),
            GNodeArg(GNA_IN, "S11_Mul_scale", 0),
            GNodeArg(GNA_IN, "S11_Mul_shift", 0),
            GNodeArg(GNA_IN, "S11_Infos", 0)
        )
    );
    // Node CONV_2D_0_3_activation inq -5.93<(i8-0.00)*0.04633911<5.89 forced outq -5.93<(i8-0.00)*0.04633911<5.89 forced
    AddNode("NAV_S12_Act_Relu6",
        Bindings(3,
            GNodeArg(GNA_IN, "S11_Output", 0),
            GNodeArg(GNA_OUT, "S12_Output", 0),
            GNodeArg(GNA_IN, "S12_Infos", 0)
        )
    );
    // Node S15_Conv2d_4x4x3x3 inq -5.93<(i8-0.00)*0.04633911<5.89 forced weightsq chan<(i8-0.00)*chan<chan outq -5.17<(i8-0.00)*0.04041220<5.13 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("NAV_S15_Conv2d_4x4x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S12_Output", 0),
            GNodeArg(GNA_IN, "Conv2dlastvaluequantfakequantw_a0d91a2a", 0),
            GNodeArg(GNA_IN, "Fusedbatchnormv3biasaddreadvar_226e0269", 0),
            GNodeArg(GNA_OUT, "S15_Output", 0),
            GNodeArg(GNA_IN, "S15_Mul_scale", 0),
            GNodeArg(GNA_IN, "S15_Mul_shift", 0),
            GNodeArg(GNA_IN, "S15_Infos", 0)
        )
    );
    // Node CONV_2D_0_4_activation inq -5.17<(i8-0.00)*0.04041220<5.13 forced outq -5.17<(i8-0.00)*0.04041220<5.13 forced
    AddNode("NAV_S16_Act_Relu6",
        Bindings(3,
            GNodeArg(GNA_IN, "S15_Output", 0),
            GNodeArg(GNA_OUT, "S16_Output", 0),
            GNodeArg(GNA_IN, "S16_Infos", 0)
        )
    );
    // Node S19_Conv2d_4x1x3x3 inq -4.13<(i8-0.00)*0.03225412<4.10 forced weightsq chan<(i8-0.00)*chan<chan outq -5.17<(i8-0.00)*0.04041220<5.13 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("NAV_S19_Conv2d_4x1x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "Input_2", 0),
            GNodeArg(GNA_IN, "Conv2dlastvaluequantfakequantw_44a058cb", 0),
            GNodeArg(GNA_IN, "Fusedbatchnormv3biasaddreadvar_5b219a5a", 0),
            GNodeArg(GNA_OUT, "S19_Output", 0),
            GNodeArg(GNA_IN, "S19_Mul_scale", 0),
            GNodeArg(GNA_IN, "S19_Mul_shift", 0),
            GNodeArg(GNA_IN, "S19_Infos", 0)
        )
    );
    // Node CONV_2D_0_6_activation inq -5.17<(i8-0.00)*0.04041220<5.13 forced outq -5.17<(i8-0.00)*0.04041220<5.13 forced
    AddNode("NAV_S20_Act_Relu6",
        Bindings(3,
            GNodeArg(GNA_IN, "S19_Output", 0),
            GNodeArg(GNA_OUT, "S20_Output", 0),
            GNodeArg(GNA_IN, "S20_Infos", 0)
        )
    );
    // Node S24_Conv2d_16x8x3x3 inq -5.17<(i8-0.00)*0.04041220<5.13 forced weightsq chan<(i8-0.00)*chan<chan outq -6.06<(i8-0.00)*0.04735255<6.01 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("NAV_S24_Conv2d_16x8x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S21_Output", 0),
            GNodeArg(GNA_IN, "Conv2dlastvaluequantfakequantw_da8d4c27", 0),
            GNodeArg(GNA_IN, "Fusedbatchnormv3biasaddreadvar_ebf0beac", 0),
            GNodeArg(GNA_OUT, "S24_Output", 0),
            GNodeArg(GNA_IN, "S24_Mul_scale", 0),
            GNodeArg(GNA_IN, "S24_Mul_shift", 0),
            GNodeArg(GNA_IN, "S24_Infos", 0)
        )
    );
    // Node CONV_2D_0_8_activation inq -6.06<(i8-0.00)*0.04735255<6.01 forced outq -6.06<(i8-0.00)*0.04735255<6.01 forced
    AddNode("NAV_S25_Act_Relu6",
        Bindings(3,
            GNodeArg(GNA_IN, "S24_Output", 0),
            GNodeArg(GNA_OUT, "S25_Output", 0),
            GNodeArg(GNA_IN, "S25_Infos", 0)
        )
    );
    // Node S28_Conv2d_16x16x3x3 inq -6.06<(i8-0.00)*0.04735255<6.01 forced weightsq chan<(i8-0.00)*chan<chan outq -5.67<(i8-0.00)*0.04430048<5.63 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("NAV_S28_Conv2d_16x16x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S25_Output", 0),
            GNodeArg(GNA_IN, "Conv2dlastvaluequantfakequantw_ff42dfe1", 0),
            GNodeArg(GNA_IN, "Fusedbatchnormv3biasaddreadvar_b3e609d3", 0),
            GNodeArg(GNA_OUT, "S28_Output", 0),
            GNodeArg(GNA_IN, "S28_Mul_scale", 0),
            GNodeArg(GNA_IN, "S28_Mul_shift", 0),
            GNodeArg(GNA_IN, "S28_Infos", 0)
        )
    );
    // Node CONV_2D_0_9_activation inq -5.67<(i8-0.00)*0.04430048<5.63 forced outq -5.67<(i8-0.00)*0.04430048<5.63 forced
    AddNode("NAV_S29_Act_Relu6",
        Bindings(3,
            GNodeArg(GNA_IN, "S28_Output", 0),
            GNodeArg(GNA_OUT, "S29_Output", 0),
            GNodeArg(GNA_IN, "S29_Infos", 0)
        )
    );
    // Node S32_Conv2d_32x16x3x3 inq -5.67<(i8-0.00)*0.04430048<5.63 forced weightsq chan<(i8-0.00)*chan<chan outq -6.06<(i8-0.00)*0.04735255<6.01 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("NAV_S32_Conv2d_32x16x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S29_Output", 0),
            GNodeArg(GNA_IN, "Conv2dlastvaluequantfakequantw_7679cecd", 0),
            GNodeArg(GNA_IN, "Fusedbatchnormv3biasaddreadvar_cc476866", 0),
            GNodeArg(GNA_OUT, "S32_Output", 0),
            GNodeArg(GNA_IN, "S32_Mul_scale", 0),
            GNodeArg(GNA_IN, "S32_Mul_shift", 0),
            GNodeArg(GNA_IN, "S32_Infos", 0)
        )
    );
    // Node CONV_2D_0_10_activation inq -6.06<(i8-0.00)*0.04735255<6.01 forced outq -6.06<(i8-0.00)*0.04735255<6.01 forced
    AddNode("NAV_S33_Act_Relu6",
        Bindings(3,
            GNodeArg(GNA_IN, "S32_Output", 0),
            GNodeArg(GNA_OUT, "S33_Output", 0),
            GNodeArg(GNA_IN, "S33_Infos", 0)
        )
    );
    // Node S36_Conv2d_32x32x3x3 inq -6.06<(i8-0.00)*0.04735255<6.01 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("NAV_S36_Conv2d_32x32x3x3",
        Bindings(7,
            GNodeArg(GNA_IN, "S33_Output", 0),
            GNodeArg(GNA_IN, "Conv2dlastvaluequantfakequantw_699b5be9", 0),
            GNodeArg(GNA_IN, "Fusedbatchnormv3biasaddreadvar_d9ff256d", 0),
            GNodeArg(GNA_OUT, "S36_Output", 0),
            GNodeArg(GNA_IN, "S36_Mul_scale", 0),
            GNodeArg(GNA_IN, "S36_Mul_shift", 0),
            GNodeArg(GNA_IN, "S36_Infos", 0)
        )
    );
    // Node CONV_2D_0_11_activation inq 0.00<(i8--128.00)*0.02352941<6.00 outq 0.00<(i8--128.00)*0.02352941<6.00
    AddNode("NAV_S37_Act_Relu6",
        Bindings(3,
            GNodeArg(GNA_IN, "S36_Output", 0),
            GNodeArg(GNA_OUT, "S37_Output", 0),
            GNodeArg(GNA_IN, "S37_Infos", 0)
        )
    );
    // Node FULLY_CONNECTED_0_13 inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq -0.50<(i8-0.00)*0.00392684<0.50 outq -1.33<(i8-0.00)*0.01038059<1.32
    AddNode("NAV_S40_Linear_1x1152",
        Bindings(7,
            GNodeArg(GNA_IN, "S37_Output", 0),
            GNodeArg(GNA_IN, "Matmullastvaluequantfakequantw", 0),
            GNodeArg(GNA_IN, "Biasaddreadvariableop", 0),
            GNodeArg(GNA_OUT, "Output_1", 0),
            GNodeArg(GNA_IN, "S40_Mul_scale", 0),
            GNodeArg(GNA_IN, "S40_Mul_shift", 0),
            GNodeArg(GNA_IN, "S40_Infos", 0)
        )
    );
    CloseGraph();
#endif
}

int main(int argc, char **argv)

{
    if (TilerParseOptions(argc, argv)) {
            printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
    }
    gate_navigator_model_v22Model(64000, 300000, 8000000, 64*1024*1024);
    GenerateTilingCode();
    return 0;
}
