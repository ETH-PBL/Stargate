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
# Konstantin Kalenberg, ETH Zurich (kkalenbe@ethz.ch)
# Hanna Müller ETH Zurich (hanmuell@iis.ee.ethz.ch)
# Tommaso Polonelli, ETH Zurich (tommaso.polonelli@pbl.ee.ethz.ch)
# Alberto Schiaffino, ETH Zurich (aschiaffino@ethz.ch)
# Vlad Niculescu, ETH Zurich (vladn@ethz.ch)
# Cristian Cioflan, ETH Zurich (cioflanc@ethz.ch)
# Michele Magno, ETH Zurich (michele.magno@pbl.ee.ethz.ch)
# Luca Benini, ETH Zurich (lbenini@iis.ee.ethz.ch)
#
# This script runs the quantized models on GVSoC


import numpy as np
import os

from tqdm import tqdm
import random

# utils
from utility import standardize_camera_tof_sample, dequantize, show_image_tof, \
    nn_tool_get_navigation_model, nn_tool_get_class_model


def get_models_val_data_navigation(config):

    data_loading_path_navigation = "../"+config["DATA_PATHS"]["DATA_LOADING_PATH_NAVIGATION"] + 'validation/'
    model_identifier_navigation = config["QUANTIZATION_NAVIGATION"]["MODEL_IDENTIFIER"]
    navigation_loading_path = config["QUANTIZATION_NAVIGATION"]["NAVIGATION_LOADING_MODEL"]
    mean_image = float(config["NORMALIZATION"]["MEAN_IMAGE"])
    std_image = float(config["NORMALIZATION"]["STD_IMAGE"])
    mean_tof = float(config["NORMALIZATION"]["MEAN_TOF"])
    std_tof = float(config["NORMALIZATION"]["STD_TOF"])


    model = nn_tool_get_navigation_model(model_loading_path=navigation_loading_path,
                                         model_identifier=model_identifier_navigation)
    data = list()
    labels = list()

    existing_runs_int = [int(run_number) for run_number in os.listdir(data_loading_path_navigation)]
    for current_run in tqdm(existing_runs_int):
        current_run_path = data_loading_path_navigation + str(current_run) + '/'
        existing_data_points_int = [int(os.path.splitext(name)[0]) for name in
                                    os.listdir(current_run_path + 'camera_images/')]
        random_sample = random.sample(existing_data_points_int, len(existing_data_points_int) // 20)
        for current_data_point in random_sample:
            data_point_name = str(current_data_point) + '.npy'
            image = np.load(current_run_path + 'camera_images/' + data_point_name)
            tof = np.load(current_run_path + 'tof_distance_array/' + data_point_name)
            label = np.load(current_run_path + 'label_yaw_rate_desired/' + data_point_name)

            image, tof = standardize_camera_tof_sample(image, tof, mean_image, std_image, mean_tof, std_tof)
            data.append([image, tof])
            labels.append(label[0])

    return model, data, labels

def get_models_val_data_classification(config):
    data_loading_path_classification = "../"+config["DATA_PATHS"]["DATA_LOADING_PATH_CLASSIFICATION"] + 'validation/'
    classifier_loading_path = config["QUANTIZATION_CLASSIFICATION"]["CLASSIFICATION_LOADING_MODEL"]
    model_identifier_classification = config["QUANTIZATION_CLASSIFICATION"]["MODEL_IDENTIFIER"]

    mean_image = float(config["NORMALIZATION"]["MEAN_IMAGE"])
    std_image = float(config["NORMALIZATION"]["STD_IMAGE"])
    mean_tof = float(config["NORMALIZATION"]["MEAN_TOF"])
    std_tof = float(config["NORMALIZATION"]["STD_TOF"])

    model_unquant = nn_tool_get_class_model(model_loading_path=classifier_loading_path, model_identifier=model_identifier_classification,
                                    quantize=False)
    model_quant = nn_tool_get_class_model(model_loading_path=classifier_loading_path, model_identifier=model_identifier_classification,
                                    quantize=True)

    data = list()
    labels = list()

    print('Processing gate data')
    data_loading_path_classification_gate = data_loading_path_classification + 'gate/'
    existing_runs_int = [int(run_number) for run_number in os.listdir(data_loading_path_classification_gate)]
    for current_run in tqdm(existing_runs_int):
        current_run_path = data_loading_path_classification_gate + str(current_run) + '/'
        existing_data_points_int = [int(os.path.splitext(name)[0]) for name in os.listdir(current_run_path + 'camera_images/')]
        random_sample = random.sample(existing_data_points_int, len(existing_data_points_int) // 20)
        for current_data_point in random_sample:
            data_point_name = str(current_data_point) + '.npy'
            image = np.load(current_run_path + 'camera_images/' + data_point_name)
            tof = np.load(current_run_path + 'tof_distance_array/' + data_point_name)
            label = 1

            image, tof = standardize_camera_tof_sample(image, tof, mean_image, std_image, mean_tof, std_tof)
            data.append([image, tof])
            labels.append(label)

    print('Processing no_gate data')
    data_loading_path_classification_nogate = data_loading_path_classification + 'no_gate/'
    existing_runs_int = [int(run_number) for run_number in os.listdir(data_loading_path_classification_nogate)]
    for current_run in tqdm(existing_runs_int):
        current_run_path = data_loading_path_classification_nogate + str(current_run) + '/'
        existing_data_points_int = [int(os.path.splitext(name)[0]) for name in os.listdir(current_run_path + 'camera_images/')]
        random_sample = random.sample(existing_data_points_int, len(existing_data_points_int) // 20)
        for current_data_point in random_sample:
            data_point_name = str(current_data_point) + '.npy'
            image = np.load(current_run_path + 'camera_images/' + data_point_name)
            tof = np.load(current_run_path + 'tof_distance_array/' + data_point_name)
            label = 0

            image, tof = standardize_camera_tof_sample(image, tof, mean_image, std_image, mean_tof, std_tof)
            data.append([image, tof])
            labels.append(label)

    return model_unquant, model_quant, data, labels


def compare_models_unquant_quant_on_target(model_quant, data, labels, directory, zero_point, scale,
                                           num_iterations, model_prefix,is_nav=False, model_unquant=None):
   
    for _ in range(0, num_iterations):
        print("Iteration ", _)
        data_index = random.randint(0, len(data) - 1)
        quantized_execution = model_quant.execute(data[data_index], quantize=True, dequantize=False)
        if is_nav:
            inputs = [quantized_execution[0][0], quantized_execution[2][0]]
        else:
            inputs = [quantized_execution[0][0], quantized_execution[1][0]]
        res = model_quant.execute_on_target(
            pmsis_os='freertos',
            platform="gvsoc",
            directory=directory,
            input_tensors=inputs,
            output_tensors=True,
            at_log=False,
            dont_run=False,
            do_clean=False,
            source='gapoc_b_v2',
            at_loglevel=1,
            print_output=False,
            pretty=True,
            performance=True,
            settings={
                'graph_produce_node_names': False,
                'graph_produce_operinfos': False,
                'graph_monitor_cycles': True,
                # 'graph_monitor_cvar_name': 'AT_' + model_prefix + '_Monitor',
                # 'graph_produce_operinfos_cvar_name': 'AT_' + model_prefix + '_Op',
                # 'graph_produce_node_cvar_name': 'AT_' + model_prefix + '_Nodes'
            }
        )
        print("///////////////////////////////////////////////////")
        print("///////////////////////////////////////////////////")
        print("///////////////////////////////////////////////////")
        print("///////////////////////////////////////////////////")
        print('Next sample')
        print("#################################################")
        print('Label: ', labels[data_index])
        print("#################################################")
        if model_unquant is not None:
            print('Result FP model: ', model_unquant.execute(data[data_index], quantize=False, dequantize=False)[-1][0])
            print("#################################################")
        print('Result INT8 model, run on local machine, not dequantized: ', quantized_execution[-1][0])
        print("#################################################")
        print('Result INT8 model, run on local machine, dequantized: ',
              model_quant.execute(data[data_index], quantize=True, dequantize=True)[-1][0])
        print("#################################################")
        print('Result INT8 model, run on board/gvsoc, not dequantized: ', res.output_tensors[-1][0])
        print("#################################################")
        print('Result INT8 model, run on board/gvsoc, dequantized: ',
              dequantize(res.output_tensors[-1][0], zero_point=zero_point, scale=scale))
        print("#################################################")


        show_image_tof(data[data_index][0], data[data_index][1])
  
