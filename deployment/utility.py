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


from nntool.api import NNGraph

import pickle
import cv2
import numpy as np

def nn_tool_get_class_model(model_loading_path, model_identifier, quantize):

    loading_path_model = "../training_quantization/" + model_loading_path + 'gate_classifier_model_' + model_identifier + '.onnx'
    loading_path_quant_stats_file = "../training_quantization/" + model_loading_path + 'quant_stats_gate_classifier_model_' + model_identifier + '.json'

    model = NNGraph.load_graph(loading_path_model, load_quantization=False)
    model.adjust_order()

    if quantize:
        fp = open(loading_path_quant_stats_file, 'rb')
        astats = pickle.load(fp)
        fp.close()
        model.fusions('scaled_match_group')

        model.quantize(statistics=astats, schemes=['scaled'])
        print(model.qshow())
        # model.draw(quant_labels=True)
    else:
        model.fusions()
        model.quantization = None

    #print(model.show())
    return model

def nn_tool_get_navigation_model(model_identifier,  model_loading_path):
    loading_path_model = "../training_quantization/"+model_loading_path + 'gate_navigator_model_' + model_identifier + '.tflite'
    model = NNGraph.load_graph(loading_path_model, load_quantization=True)
    model.adjust_order()
    model.fusions()

    print(model.qshow())

    return model

################################################################################
#This function transforms an np.array image from the range [0,255] to the range [0, 1] and then standardizes both image and tof
################################################################################
def standardize_camera_tof_sample(image, tof, mean_image, std_image, mean_tof, std_tof):

    image = ((image / 255.0) - mean_image) / std_image
    tof = (tof - mean_tof) / std_tof
    return image, tof

def show_image_tof(image, tof):
    image = (image * 255).astype(np.uint8)
    # Scale tof from 0.0-3.0 into 0-255
    tof = (tof * 255 / 3).astype(np.uint8)

    # For visibility of output only
    tof = cv2.resize(tof, dsize=[168, 168], interpolation=cv2.INTER_NEAREST)

    result_image = cv2.hconcat([image, tof])

    # Show processed images
    cv2.imshow('Image and ToF', result_image)
    cv2.waitKey(5000)

def dequantize(val, zero_point, scale):
    return (np.float32(val) - zero_point) * scale