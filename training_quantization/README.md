# Training and Quantization

## Overview of directories and files 
 * [models/](models/),  it contains the Python classes of the CNN models of the project.
   - [gate_classifier_PyTorch_model.py](models/gate_classifier_PyTorch_model.py), gate classifier model described using PyTorch framework.
   - [gate_navigator_Tensorflow_model.py](models/gate_navigator_Tensorflow_model.py), gate navigator model described using Tensorflow framework.
 * [onnx_models/](onnx_models/), all the required files for deploying the gate classifier will be saved here.
 * [tflite_models/](tflite_models/), all the required files for deploying the gate navigator will be saved here.
 * [deep_learning_config.ini](deep_learning_config.ini), contains useful information used throughout the process. The user can modify the parameters of the training through this file.
 * [main_deep_learning.py](main_deep_learning.py), the main file that needs to be executed to perform training and quantization of both models.
 * [nntool_gate_classifier_PyTorch.py](nntool_gate_classifier_PyTorch.py), all the required functions for quantizing the gate classifier model using [NNTool](https://github.com/GreenWaves-Technologies/gap_sdk/tree/master/tools/nntool).
 * [nntool_gate_navigator_Tensorflow.py](nntool_gate_navigator_Tensorflow.py), all the required functions for testing the quantized gate navigator model using [NNTool](https://github.com/GreenWaves-Technologies/gap_sdk/tree/master/tools/nntool).
 * [training_gate_classifier_PyTorch.py](training_gate_classifier_PyTorch.py), all the required functions for training the gate classifier model using the PyTorch framework.
 * [training_gate_navigator_Tensorflow.py](training_gate_navigator_Tensorflow.py), all the required functions for training and QAT the gate navigator model using the Tensorflow framework.
 * [utility.py](utility.py), some useful functions used throughout the process.

## Configuration file ([deep_learning_config.ini](deep_learning_config.ini))
The file contains variables needed as input for the training, as well as parameters that are computed during execution that will be used during the deployment.
These last are overwritten at every training and, therefore are also saved in the [onnx_models/](onnx_models/) and [tflite_models/](tflite_models/) directories as .txt files.

## Command to perform training and quantization
1. Check that the DATA_PATHS in the [deep_learning_config.ini](deep_learning_config.ini) file points to the directories in which you have downloaded the dataset.
2. If you want, you can adjust the training parameters in the same file.
3. Execute the following command: `python3 main_deep_learning.py` 

## Files saved throughout the execution
* [onnx_models/](onnx_models/)
    - gate_classifier_model_{wandb_model_version}.onnx, the .onnx file of the trained gate classifier. 
    - quant_state_gate_classifier_model_{wandb_model_version}.json, quantization stats retrieved by [NNTool](https://github.com/GreenWaves-Technologies/gap_sdk/tree/master/tools/nntool).
    - quant_values_gate_classifier_model_{wandb_model_version}.txt, inputs' and output's quantization values.
* [tflite_models/](tflite_models/)
    - gate_navigator_model_{wandb_model_version}.tflite, the .tflite file of the trained and quantized gate navigator.
    - quant_values_gate_navigator_model_{wandb_model_version}.txt, inputs' and output's quantization values. 
  
