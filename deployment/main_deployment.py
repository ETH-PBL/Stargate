#
# Copyright (C) 2022-2024 ETH Zurich
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0
# ======================================================================
#
# Authors: 
# Konstantin Kalenberg, ETH Zurich
# Hanna Müller ETH Zurich (hanmuell@iis.ee.ethz.ch)
# Tommaso Polonelli, ETH Zurich
# Alberto Schiaffino, ETH Zurich
# Vlad Niculescu, ETH Zurich
# Cristian Cioflan, ETH Zurich
# Michele Magno, ETH Zurich
# Luca Benini, ETH Zurich
#


import configparser
from test_models_on_target import get_models_val_data_classification, get_models_val_data_navigation, compare_models_unquant_quant_on_target   

def main():

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("../training_quantization/deep_learning_config.ini")

    num_samples_to_inspect = 1

    model_navigation, val_data_navigation, val_labels_navigation = get_models_val_data_navigation(config)

    print('Running navigation model on board/gvsoc, this might take a while')
    compare_models_unquant_quant_on_target(model_quant=model_navigation, data=val_data_navigation,
                                            labels=val_labels_navigation,
                                            directory="navigation_model_quant",
                                            zero_point=float(config["QUANTIZATION_NAVIGATION"]["output_zero_point"]),
                                            scale=float(config["QUANTIZATION_NAVIGATION"]["output_scale"]),
                                            num_iterations=num_samples_to_inspect, is_nav=True, model_prefix="Navigation")
    

    model_classification_unquant, model_classification_quant, val_data_classification, val_labels_classification = get_models_val_data_classification(config)
    
    print('Running classification model on board/gvsoc, this might take a while')
    compare_models_unquant_quant_on_target(model_quant=model_classification_quant, data=val_data_classification, 
                                            labels=val_labels_classification,
                                            directory="classification_model_quant",
                                            zero_point=float(config["QUANTIZATION_CLASSIFICATION"]["output_zero_point"]),
                                            scale=float(config["QUANTIZATION_CLASSIFICATION"]["output_scale"]),
                                            num_iterations=num_samples_to_inspect, is_nav=False,
                                            model_prefix="Classification",model_unquant=model_classification_unquant)
    
    print("\n\nIf results are not correctly scaled when dequantized, please adjust scale and zero point values in :"+
          " training_quantization/deep_learning_config.ini\n"+
          "The quantization values for each version of your models are saved in training_quantization/onnx_models and training_quantization/tflite_models as .txt files\n")

if __name__ == "__main__":
    main()