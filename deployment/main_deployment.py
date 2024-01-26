
import configparser
from test_models_on_target import get_models_val_data_classification, get_models_val_data_navigation, compare_models_unquant_quant_on_target   

def main():

    # TODO: Add automatic writing of the new scales and zero points to the config file.
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("../training_quantization/deep_learning/deep_learning_config.ini")

    num_samples_to_inspect = 1

    model_navigation, val_data_navigation, val_labels_navigation = get_models_val_data_navigation(config)

    print('Running navigation model on board/gvsoc, this might take a while')
    compare_models_unquant_quant_on_target(model_quant=model_navigation, data=val_data_navigation,
                                            labels=val_labels_navigation,
                                            directory="navigation_model_quant",
                                            zero_point=float(config["QUANTIZATION_NAVIGATION"]["ZERO_POINT"]),
                                            scale=float(config["QUANTIZATION_NAVIGATION"]["SCALE"]),
                                            num_iterations=num_samples_to_inspect, is_nav=True, model_prefix="Navigation")
    

    model_classification_unquant, model_classification_quant, val_data_classification, val_labels_classification = get_models_val_data_classification(config)
    
    print('Running classification model on board/gvsoc, this might take a while')
    compare_models_unquant_quant_on_target(model_quant=model_classification_quant, data=val_data_classification, 
                                            labels=val_labels_classification,
                                            directory="classification_model_quant",
                                            zero_point=float(config["QUANTIZATION_CLASSIFICATION"]["ZERO_POINT"]),
                                            scale=float(config["QUANTIZATION_CLASSIFICATION"]["SCALE"]),
                                            num_iterations=num_samples_to_inspect, is_nav=False,
                                            model_prefix="Classification",model_unquant=model_classification_unquant)
    
    print("\n\nIf results are not correctly scaled when dequantized, please adjust scale and zero point values in :\n",
          "training_quantization/deep_learning/deep_learning_config.ini\n\n")

if __name__ == "__main__":
    main()