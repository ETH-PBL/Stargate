"""
This script is used to compute the quantization calibration stats (quantization ranges) for both GateNavigator and GateClassifier networks
The data used for calibration is the same as the training data of the networks

Adapted from: https://github.com/GreenWaves-Technologies/tiny_denoiser/blob/public/model/nntool_scripts/collect_stats.py
author: Konstantin Kalenberg
"""
import configparser
import numpy as np
import os
from tqdm import tqdm
import pickle
import random

# import nntool APIs
from nntool.api import NNGraph
from nntool.stats.activation_ranges_collector import ActivationRangesCollector

# utils
from utility import standardize_camera_tof_sample
from utility import nn_tool_get_class_model, accuracy_loss



def compute_quantization_stats(config):
    data_loading_path_classification = "../../" + config["DATA_PATHS"]["DATA_LOADING_PATH_CLASSIFICATION"] + 'training/'
    model_loading_path = config["QUANTIZATION_CLASSIFICATION"]["CLASSIFICATION_LOADING_MODEL"]
    model_identifier_classification = config["QUANTIZATION_CLASSIFICATION"]["MODEL_IDENTIFIER"]

    mean_image = float(config["NORMALIZATION"]["MEAN_IMAGE"])
    std_image = float(config["NORMALIZATION"]["STD_IMAGE"])
    mean_tof = float(config["NORMALIZATION"]["MEAN_TOF"])
    std_tof = float(config["NORMALIZATION"]["STD_TOF"])
    quantization_stats_path_classification = model_loading_path + 'quant_stats_gate_classifier_model_' + model_identifier_classification + '.json'

    print("Calibration data taken from: ")
    print(data_loading_path_classification)

    print("Saving quantization stats files to: ")
    print(quantization_stats_path_classification)

    # Collect Classification stats
    print('Collecting classification quantization stats. This might take a while')
    graph_classification = NNGraph.load_graph(model_loading_path + 'gate_classifier_model_' + model_identifier_classification + '.onnx',
                                              load_quantization=False)
    graph_classification.adjust_order()
    graph_classification.fusions('scaled_match_group')
    graph_classification.quantization = None
    stats_collector_classification = ActivationRangesCollector(use_ema=False)

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

            image, tof = standardize_camera_tof_sample(image, tof, mean_image, std_image, mean_tof, std_tof)
            data = [image, tof]
            stats_collector_classification.collect_stats(graph_classification, data)

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

            image, tof = standardize_camera_tof_sample(image, tof, mean_image, std_image, mean_tof, std_tof)
            data = [image, tof]
            stats_collector_classification.collect_stats(graph_classification, data)

    # Save quantization stats to file
    astats = stats_collector_classification.stats
    with open(quantization_stats_path_classification, 'wb') as fp:
        pickle.dump(astats, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved classification quantization stats')

def compute_classification_validation_score_quantized_and_unquantized(config):
    data_loading_path_classification = "../../" + config["DATA_PATHS"]["DATA_LOADING_PATH_CLASSIFICATION"] + 'training/'
    model_loading_path = config["QUANTIZATION_CLASSIFICATION"]["CLASSIFICATION_LOADING_MODEL"]
    model_identifier_classification = config["QUANTIZATION_CLASSIFICATION"]["MODEL_IDENTIFIER"]

    mean_image = float(config["NORMALIZATION"]["MEAN_IMAGE"])
    std_image = float(config["NORMALIZATION"]["STD_IMAGE"])
    mean_tof = float(config["NORMALIZATION"]["MEAN_TOF"])
    std_tof = float(config["NORMALIZATION"]["STD_TOF"])
    print('Collecting classification model scores. This might take a while')
    model_unquant = nn_tool_get_class_model(model_loading_path=model_loading_path, model_identifier=model_identifier_classification,
                                       quantize=False)
    model_quant = nn_tool_get_class_model(model_loading_path=model_loading_path, model_identifier=model_identifier_classification,
                                    quantize=True)

    labels = list()
    preds_unquant = list()
    preds_quant = list()

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
            data = [image, tof]

            pred_unquant = model_unquant.execute(data)[-1]
            pred_quant = model_quant.execute(data, quantize=True, dequantize=True)[-1]

            labels.append(label)
            preds_unquant.append(1 if pred_unquant[0][0] > 0.5 else 0)
            preds_quant.append(1 if pred_quant[0][0] > 0.5 else 0)

    print('Processing no_gate data')
    data_loading_path_classification_nogate = data_loading_path_classification + 'no_gate/'
    existing_runs_int = [int(run_number) for run_number in os.listdir(data_loading_path_classification_nogate)]
    for current_run in tqdm(existing_runs_int):
        current_run_path = data_loading_path_classification_nogate + str(current_run) + '/'
        existing_data_points_int = [int(os.path.splitext(name)[0]) for name in os.listdir(current_run_path + 'camera_images/')]
        random_sample = random.sample(existing_data_points_int, len(existing_data_points_int) // 20) # // 1 if you want the whole dataset
        for current_data_point in random_sample:
            data_point_name = str(current_data_point) + '.npy'
            image = np.load(current_run_path + 'camera_images/' + data_point_name)
            tof = np.load(current_run_path + 'tof_distance_array/' + data_point_name)
            label = 0

            image, tof = standardize_camera_tof_sample(image, tof, mean_image, std_image, mean_tof, std_tof)
            data = [image, tof]

            pred_unquant = model_unquant.execute(data)[-1]
            pred_quant = model_quant.execute(data, quantize=True, dequantize=True)[-1]

            labels.append(label)
            preds_unquant.append(1 if pred_unquant[0][0] > 0.5 else 0)
            preds_quant.append(1 if pred_quant[0][0] > 0.5 else 0)

    # Compute accuracy
    accuracy_classification_unquant = accuracy_loss(np.asarray(labels), np.asarray(preds_unquant))
    accuracy_classification_quant = accuracy_loss(np.asarray(labels), np.asarray(preds_quant))

    print('#################################################')
    print('Accuracy classification un-quantized / quantized: ', accuracy_classification_unquant, ' / ', accuracy_classification_quant)
    print('#################################################')


def quantize_classifier():
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("deep_learning_config.ini")
   
    compute_quantization_stats(config)
    compute_classification_validation_score_quantized_and_unquantized(config)
