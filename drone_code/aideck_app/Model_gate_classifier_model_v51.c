#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators_SQ8.h"

#include "CNN_Copy_Generators.h"





void gate_classifier_model_v51Model(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 3, "Gap.h", "gate_classifier_model_v51.h", "CNN_BasicKernels_SQ8.h");
    SetGeneratedFilesNames("gate_classifier_model_v51Kernels.c", "gate_classifier_model_v51Kernels.h");
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CVAR_NAME, AT_OPT_VAL("AT_Classification_Monitor"));
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_CVAR_NAME, AT_OPT_VAL("AT_Classification_Nodes"));
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS_CVAR_NAME, AT_OPT_VAL("AT_Classification_Op"));
    //AT_SetGraphCtrl(AT_GRAPH_DUMP_TENSOR, AT_OPT_VAL(7));

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "gate_classifier_model_v51_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "gate_classifier_model_v51_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "gate_classifier_model_v51_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "gate_classifier_model_v51_L3_Flash", "gate_classifier_model_v51_L3_Flash_Const.dat", 0
    );

    LoadCNN_SQ8_Library();


    // generator for _conv_camera_layer_1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("CLASS_S4_Conv2d_4x1x5x5_MaxPool_2x2_Relu6", 0,
                               4, 1,
                               1, 4, 168, 168,
                               KOP_CONV, 5, 5, 1, 1, 2, 2, 1,
                               KOP_MAXPOOL, 2, 2, 1, 1, 2, 2, 0,
                               KOP_RELU);

    // generator for _camera_block_1_conv1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("CLASS_S7_Conv2d_4x4x3x3_Relu6", 0,
                               4, 1,
                               4, 4, 42, 42,
                               KOP_CONV, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);

    // generator for _camera_block_1_conv2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("CLASS_S10_Conv2d_4x4x3x3_Relu6", 0,
                               4, 1,
                               4, 4, 21, 21,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);

    // generator for _tof_layer_1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("CLASS_S13_Conv2d_4x1x3x3_Relu6", 0,
                               4, 1,
                               1, 4, 21, 21,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);

    // generator for _combined_block_1_conv1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("CLASS_S17_Conv2d_16x8x3x3_Relu6", 0,
                               4, 1,
                               8, 16, 21, 21,
                               KOP_CONV, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);

    // generator for _combined_block_1_conv2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("CLASS_S20_Conv2d_16x16x3x3_Relu6", 0,
                               4, 1,
                               16, 16, 11, 11,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);

    // generator for _combined_block_2_conv1_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("CLASS_S23_Conv2d_32x16x3x3_Relu6", 0,
                               4, 1,
                               16, 32, 11, 11,
                               KOP_CONV, 3, 3, 1, 1, 2, 2, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELU);

    // generator for _combined_block_2_conv2_Conv_fusion
    CNN_ConvolutionPoolAct_SQ8("CLASS_S26_Conv2d_32x32x3x3_Relu6", 0,
                               4, 1,
                               32, 32, 6, 6,
                               KOP_CONV, 3, 3, 1, 1, 1, 1, 1,
                               KOP_NONE, 0, 0, 0, 0, 0, 0, 0,
                               KOP_RELUM);
    
    // generator for _fully_connected_Gemm_fusion
    CNN_LinearAct_SQ8("S29_Op__fully_connected_Gemm_fusion", 0,
                      4, 1,
                      1152, 1,
                      KOP_LINEAR, KOP_SIGMOID);
    

#define GRAPH
#ifdef GRAPH
    CreateGraph("gate_classifier_model_v51CNN",
        /* Arguments either passed or globals */
            CArgs(48,
                TCArgInfo("signed char * __restrict__", "Input_1", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "Input_2", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
                TCArgInfo("signed char * __restrict__", "_conv_camera_layer_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/_conv_camera_layer_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_111", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/Constant_onnx__conv_111.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S4_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S4_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S4_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S4_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S4_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S4_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_camera_block_1_conv1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/_camera_block_1_conv1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_114", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/Constant_onnx__conv_114.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S7_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S7_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S7_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S7_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S7_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S7_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_camera_block_1_conv2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/_camera_block_1_conv2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_117", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/Constant_onnx__conv_117.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S10_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S10_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S10_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S10_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S10_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S10_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_tof_layer_1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/_tof_layer_1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_120", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/Constant_onnx__conv_120.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S13_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S13_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S13_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S13_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S13_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S13_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_combined_block_1_conv1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/_combined_block_1_conv1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_123", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/Constant_onnx__conv_123.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S17_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S17_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S17_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S17_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S17_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S17_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_combined_block_1_conv2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/_combined_block_1_conv2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_126", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/Constant_onnx__conv_126.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S20_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S20_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S20_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S20_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S20_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S20_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_combined_block_2_conv1_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/_combined_block_2_conv1_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_129", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/Constant_onnx__conv_129.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S23_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S23_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S23_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S23_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.04724 out: 0.04724  actscale: [1] actscalen: [0] a0: [0] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S23_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S23_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_combined_block_2_conv2_conv_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/_combined_block_2_conv2_conv_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "Constant_onnx__conv_132", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/Constant_onnx__conv_132.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S26_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S26_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S26_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S26_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.02353 out: 0.02353  actscale: [1] actscalen: [0] a0: [-128] b0: [127] c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S26_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S26_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "_fully_connected_gemm_weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/_fully_connected_gemm_weights.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed int * __restrict__", "_fully_connected_gemm_biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/_fully_connected_gemm_biases.tensor", 1, 1, 32, 0)),
                TCArgInfo("unsigned char * __restrict__", "S29_Mul_scale", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S29_Mul_scale.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "S29_Mul_shift", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S29_Mul_shift.tensor", 1, 1, 8, 0)),
                // in: 0.00024 out: 0.00787  actscale: [127] actscalen: [15] a0: [0] b0: 0 c0: 0 BIASN: 0 PRENORM: 0
                TCArgInfo("signed char * __restrict__", "S29_Infos", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("./tensors/classification_tensors/S29_Infos.tensor", 1, 1, 8, 0)),
                TCArgInfo("signed char * __restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
            ),
        /* Locals, allocated dynamically */
        CArgs(7,
            TCArgInfo("signed char * __restrict__", "S4_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S7_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S14_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S17_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S20_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S23_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char * __restrict__", "S26_Output", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    AddStackedTensors("S14_Output", 2, "S10_Output", "S13_Output");

    // Node CLASS_S4_Conv2d_4x1x5x5_MaxPool_2x2_Relu6 inq -8.64<(i8-0.00)*0.06747100<8.57 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724409<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("CLASS_S4_Conv2d_4x1x5x5_MaxPool_2x2_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "Input_1", 0),
            GNodeArg(GNA_IN, "_conv_camera_layer_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_111", 0),
            GNodeArg(GNA_OUT, "S4_Output", 0),
            GNodeArg(GNA_IN, "S4_Mul_scale", 0),
            GNodeArg(GNA_IN, "S4_Mul_shift", 0),
            GNodeArg(GNA_IN, "S4_Infos", 0)
        )
    );
    // Node CLASS_S7_Conv2d_4x4x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724409<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724409<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("CLASS_S7_Conv2d_4x4x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S4_Output", 0),
            GNodeArg(GNA_IN, "_camera_block_1_conv1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_114", 0),
            GNodeArg(GNA_OUT, "S7_Output", 0),
            GNodeArg(GNA_IN, "S7_Mul_scale", 0),
            GNodeArg(GNA_IN, "S7_Mul_shift", 0),
            GNodeArg(GNA_IN, "S7_Infos", 0)
        )
    );
    // Node CLASS_S10_Conv2d_4x4x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724409<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724409<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("CLASS_S10_Conv2d_4x4x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S7_Output", 0),
            GNodeArg(GNA_IN, "_camera_block_1_conv2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_117", 0),
            GNodeArg(GNA_OUT, "S10_Output", 0),
            GNodeArg(GNA_IN, "S10_Mul_scale", 0),
            GNodeArg(GNA_IN, "S10_Mul_shift", 0),
            GNodeArg(GNA_IN, "S10_Infos", 0)
        )
    );
    // Node CLASS_S13_Conv2d_4x1x3x3_Relu6 inq -4.05<(i8-0.00)*0.03160720<4.01 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724409<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("CLASS_S13_Conv2d_4x1x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "Input_2", 0),
            GNodeArg(GNA_IN, "_tof_layer_1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_120", 0),
            GNodeArg(GNA_OUT, "S13_Output", 0),
            GNodeArg(GNA_IN, "S13_Mul_scale", 0),
            GNodeArg(GNA_IN, "S13_Mul_shift", 0),
            GNodeArg(GNA_IN, "S13_Infos", 0)
        )
    );
    // Node CLASS_S17_Conv2d_16x8x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724409<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724409<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("CLASS_S17_Conv2d_16x8x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S14_Output", 0),
            GNodeArg(GNA_IN, "_combined_block_1_conv1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_123", 0),
            GNodeArg(GNA_OUT, "S17_Output", 0),
            GNodeArg(GNA_IN, "S17_Mul_scale", 0),
            GNodeArg(GNA_IN, "S17_Mul_shift", 0),
            GNodeArg(GNA_IN, "S17_Infos", 0)
        )
    );
    // Node CLASS_S20_Conv2d_16x16x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724409<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724409<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("CLASS_S20_Conv2d_16x16x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S17_Output", 0),
            GNodeArg(GNA_IN, "_combined_block_1_conv2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_126", 0),
            GNodeArg(GNA_OUT, "S20_Output", 0),
            GNodeArg(GNA_IN, "S20_Mul_scale", 0),
            GNodeArg(GNA_IN, "S20_Mul_shift", 0),
            GNodeArg(GNA_IN, "S20_Infos", 0)
        )
    );
    // Node CLASS_S23_Conv2d_32x16x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724409<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq -6.05<(i8-0.00)*0.04724409<6.00 forced biasesq chan<(i32-0.00)*chan<chan
    AddNode("CLASS_S23_Conv2d_32x16x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S20_Output", 0),
            GNodeArg(GNA_IN, "_combined_block_2_conv1_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_129", 0),
            GNodeArg(GNA_OUT, "S23_Output", 0),
            GNodeArg(GNA_IN, "S23_Mul_scale", 0),
            GNodeArg(GNA_IN, "S23_Mul_shift", 0),
            GNodeArg(GNA_IN, "S23_Infos", 0)
        )
    );
    // Node CLASS_S26_Conv2d_32x32x3x3_Relu6 inq -6.05<(i8-0.00)*0.04724409<6.00 forced weightsq chan<(i8-0.00)*chan<chan outq 0.00<(i8--128.00)*0.02352941<6.00 biasesq chan<(i32-0.00)*chan<chan
    AddNode("CLASS_S26_Conv2d_32x32x3x3_Relu6",
        Bindings(7,
            GNodeArg(GNA_IN, "S23_Output", 0),
            GNodeArg(GNA_IN, "_combined_block_2_conv2_conv_weights", 0),
            GNodeArg(GNA_IN, "Constant_onnx__conv_132", 0),
            GNodeArg(GNA_OUT, "S26_Output", 0),
            GNodeArg(GNA_IN, "S26_Mul_scale", 0),
            GNodeArg(GNA_IN, "S26_Mul_shift", 0),
            GNodeArg(GNA_IN, "S26_Infos", 0)
        )
    );
    // Node _fully_connected_Gemm inq 0.00<(i8--128.00)*0.02352941<6.00 weightsq -0.53<(i8-0.00)*0.00420661<0.53 outq -1.01<(i8-0.00)*0.00787402<1.00
    AddNode("S29_Op__fully_connected_Gemm_fusion",
        Bindings(7,
            GNodeArg(GNA_IN, "S26_Output", 0),
            GNodeArg(GNA_IN, "_fully_connected_gemm_weights", 0),
            GNodeArg(GNA_IN, "_fully_connected_gemm_biases", 0),
            GNodeArg(GNA_OUT, "Output_1", 0),
            GNodeArg(GNA_IN, "S29_Mul_scale", 0),
            GNodeArg(GNA_IN, "S29_Mul_shift", 0),
            GNodeArg(GNA_IN, "S29_Infos", 0)
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
    gate_classifier_model_v51Model(64000, 300000, 8000000, 64*1024*1024);
    GenerateTilingCode();
    return 0;
}
