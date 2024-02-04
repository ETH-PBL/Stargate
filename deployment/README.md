# Deployment
##  Overview of directories and files 
* [main_deployment.py](main_deployment.py), the main file that needs to be executed to perform the deployment and testing of the models.
* [test_models_on_target.py](test_models_on_target.py), all the required functions for testing the deployed C code using [GVSoC](https://github.com/GreenWaves-Technologies/gap_sdk/tree/master/gvsoc).
* [utility.py](utility.py), some useful functions used throughout the process.

## Command to perform the deployment
After training and quantization all the required files, stats, and values are saved in the [training_quantization/](../training_quantization/) folder. \
Therefore, during the execution, the [deep_learning_config.ini](../training_quantization/deep_learning_config.ini) is read to retrieve useful information, such as quantization values and versions of the models. \
Please keep in mind that if you want to perform the quantization on older trained versions of the models, you would have to modify the QUANTIZATION_NAVIGATION and QUANTIZATION_CLASSIFICATION parameters in the file accordingly. \
You can find the inputs and output quantization values in the folders on which your models are saved, as a .txt file.

To perform the deployment, execute the following command: `python3 main_deployment.py` 

## Files saved throughout the execution

## Manual steps to fuse the two CNNs produced in this execution
