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
# This script computes the score for the quantized GateNavigator model
# The data used for the scores is the same as the validation data of the networks


import numpy as np
import os
from tqdm import tqdm
import random
import configparser
# utils
from utility import standardize_camera_tof_sample, rmse_loss, nn_tool_get_navigation_model


def compute_navigation_validation_score_quantized_and_unquantized():
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("deep_learning_config.ini")
    data_loading_path_navigation = "../"+config["DATA_PATHS"]["DATA_LOADING_PATH_NAVIGATION"] + 'validation/'
    model_identifier_navigation = config["QUANTIZATION_NAVIGATION"]["MODEL_IDENTIFIER"]
    navigation_loading_path = config["QUANTIZATION_NAVIGATION"]["NAVIGATION_LOADING_MODEL"]
    mean_image = float(config["NORMALIZATION"]["MEAN_IMAGE"])
    std_image = float(config["NORMALIZATION"]["STD_IMAGE"])
    mean_tof = float(config["NORMALIZATION"]["MEAN_TOF"])
    std_tof = float(config["NORMALIZATION"]["STD_TOF"])

    # Compute navigation scores
    print('Collecting navigation model scores. This might take a while')
    model,dict_quant = nn_tool_get_navigation_model(model_loading_path=navigation_loading_path,model_identifier=model_identifier_navigation)
    labels = list()
    preds = list()

    existing_runs_int = [int(run_number) for run_number in os.listdir(data_loading_path_navigation)]
    for current_run in tqdm(existing_runs_int):
        current_run_path = data_loading_path_navigation + str(current_run) + '/'
        existing_data_points_int = [int(os.path.splitext(name)[0]) for name in os.listdir(current_run_path + 'camera_images/')]
        random_sample = random.sample(existing_data_points_int, len(existing_data_points_int) // 20) # change to // 1 for full inference on the whole val dataset
        for current_data_point in random_sample:
            data_point_name = str(current_data_point) + '.npy'
            image = np.load(current_run_path + 'camera_images/' + data_point_name)
            tof = np.load(current_run_path + 'tof_distance_array/' + data_point_name)
            label = np.load(current_run_path + 'label_yaw_rate_desired/' + data_point_name)
            image, tof = standardize_camera_tof_sample(image, tof, mean_image, std_image, mean_tof, std_tof)
            data = [image, tof]
            pred = model.execute(data,dequantize=True,quantize=True)[-1]
            labels.append(label[0])
            preds.append(pred[0][0])

    # Compute RMSE
    rmse_navigation= rmse_loss(np.asarray(labels), np.asarray(preds))

    # Save quantization dictionary
    config.set('QUANTIZATION_NAVIGATION', 'input_1_zero_point', str(dict_quant['input_1_zero']))
    config.set('QUANTIZATION_NAVIGATION', 'input_1_scale', str(dict_quant['input_1_scale']))
    config.set('QUANTIZATION_NAVIGATION', 'input_2_zero_point', str(dict_quant['input_2_zero']))
    config.set('QUANTIZATION_NAVIGATION', 'input_2_scale', str(dict_quant['input_2_scale']))
    config.set('QUANTIZATION_NAVIGATION', 'output_zero_point', str(dict_quant['output_zero']))
    config.set('QUANTIZATION_NAVIGATION', 'output_scale', str(dict_quant['output_scale']))
    with open('deep_learning_config.ini', 'w') as configfile:
        config.write(configfile)

    return rmse_navigation

def validation_score_quantized_nav_model():
    
    score_navigation = compute_navigation_validation_score_quantized_and_unquantized()
    print('\n\nRMSE navigation quantized: ', score_navigation, "\n\n")
