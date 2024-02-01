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
# This script provides utility functions for the deep learning process


import shutil

import numpy

import torch
import tensorflow as tf
from tensorflow import keras as kr
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot

import configparser
import os
from tqdm import tqdm

from datetime import datetime
import wandb
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utility import ImavChallengeNavigationDataset
from models.gate_navigator_Tensorflow_model import gate_navigator_model


# Unique id to save best model of this run
unique_run_id = str(datetime.now())


def training(train_loader, validation_loader, config):

    # Load parameters
    verbose = config.getboolean("TRAINING_NAVIGATION", "VERBOSE")
    use_wandb = config.getboolean("WANDB", "USE_WANDB")
    batch_size = int(config["TRAINING_NAVIGATION"]["BATCH_SIZE"])
    num_channels_start = int(config["TRAINING_NAVIGATION"]["NUM_CHANNELS_START"])
    lr = float(config["TRAINING_NAVIGATION"]["LEARNING_RATE"])
    lr_decay = float(config["TRAINING_NAVIGATION"]["LEARNING_RATE_DECAY"])
    epochs_training = int(config["TRAINING_NAVIGATION"]["EPOCHS"])
    dropout_p = float(config["TRAINING_NAVIGATION"]["DROPOUT_PROB"])
    data_loading_path_navigation = "../" + config["DATA_PATHS"]["DATA_LOADING_PATH_NAVIGATION"]

    # Set up weights and biases
    if use_wandb:
        wandb.init(project="StarGate", group="")

        # Log important parameters to weights and biases
        wandb.config.data_path = data_loading_path_navigation
        wandb.config.batch_size = batch_size
        wandb.config.num_channels_start = num_channels_start
        wandb.config.lr = lr
        wandb.config.epochs = epochs_training
        wandb.config.dropout_p = dropout_p
    
    # Create gate navigator model
    model = gate_navigator_model(num_channels_start)

    # Optimizer
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, epsilon=1e-08, weight_decay=lr_decay)

    # error
    train_loss = kr.metrics.RootMeanSquaredError(name='train_loss')
    validation_loss = kr.metrics.RootMeanSquaredError(name='validation_loss')
    loss_function = kr.losses.MeanSquaredError()

    if verbose:
        # print summary of navigation model
        model.summary()

    # ############################################################################
    # # Train/Val Loop
    # ############################################################################
    best_val_loss = float('inf')  # Used for saving best model of this run
    for epoch in range(epochs_training):
        print('\nEpoch: ', epoch + 1, ' / ', epochs_training)

        # Average train losses + reset_errors
        train_loss.reset_states()
        validation_loss.reset_states()

        num_train_iterations = len(train_loader)
        num_val_iterations = len(validation_loader)

        # Training Loop
        train_rmse_yaw = 0.0

        with tqdm(total=num_train_iterations, desc='Train', disable=not True) as t:

            for batch_idx, data in enumerate(train_loader):

                # Load data from torch and modify to tensorflow
                image, tof, label_yaw = data[0], data[1], data[2]
                image = image.numpy()
                image = tf.convert_to_tensor(image)
                tof = tof.numpy()
                tof = tf.convert_to_tensor(tof)
                label_yaw = label_yaw.numpy()
                label_yaw = tf.convert_to_tensor(label_yaw)

                # permute  the input from NCWH to NWHC
                image = kr.layers.Permute((2, 3, 1))(image)
                tof = kr.layers.Permute((2, 3, 1))(tof)

                with tf.GradientTape() as tape:
                    # Forward pass
                    pred_yaw = model((image, tof), training=True)
                    # Loss computation
                    loss_yaw = tf.sqrt(loss_function(label_yaw, pred_yaw))
                    # Save avg train mse losses
                    train_rmse_yaw += loss_yaw.numpy()
                # Backward pass
                gradients = tape.gradient(loss_yaw, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
         
                t.update(1)
        
        # Validation

        # Average validation losses
        val_rmse_yaw = 0.0

        for batch_idx, data in enumerate(validation_loader):
            # Load data
            image, tof, label_yaw = data[0], data[1], data[2]
            image = image.numpy()
            image = tf.convert_to_tensor(image)
            tof = tof.numpy()
            tof = tf.convert_to_tensor(tof)
            label_yaw = label_yaw.numpy()
            label_yaw = tf.convert_to_tensor(label_yaw)

            # permute from NCWH to NWHC the input
            image = kr.layers.Permute((2, 3, 1))(image)
            tof = kr.layers.Permute((2, 3, 1))(tof)

            # Validation
            pred_yaw = model((image, tof))
            loss_yaw = tf.sqrt(loss_function(label_yaw, pred_yaw))

            val_rmse_yaw += loss_yaw.numpy()
        # Compute average losses at end of epoch
        train_rmse_yaw /= num_train_iterations
        val_rmse_yaw /= num_val_iterations

        # Save model if better val loss than previous best val loss --> used later to upload to wandb
        if val_rmse_yaw < best_val_loss:
            # Remove previous best if it exists
            if os.path.exists('throwaway_models/best_model_gate_navigator' + unique_run_id):
                print('Deleted model dict')
                shutil.rmtree('throwaway_models/best_model_gate_navigator' + unique_run_id)

            # Save
            kr.models.save_model(model=model, filepath='throwaway_models/best_model_gate_navigator' +
                                                                      unique_run_id, save_format="tf")
            print('Saved new best model')

            # Update new best loss
            best_val_loss = val_rmse_yaw

        print('\n')
        print('Avg Loss Yaw Train / Val: ', train_rmse_yaw, ' / ', val_rmse_yaw)
        print('-------------------------------------------------------------')

        if use_wandb:
            wandb.log({'epoch': epoch, 'train_rmse_yaw': train_rmse_yaw, 'val_rmse_yaw': val_rmse_yaw})

    # Load model with lowest validation error of the training
    model = kr.models.load_model(filepath='throwaway_models/best_model_gate_navigator' + unique_run_id)

    # Computes RMSE loss over the whole validation dataset.
    print("\nDoing inference using RMSE over the whole validation dataset before QAT:\n")
    val_rmse_yaw = 0.0
    iterations = 0.0
    for batch_idx, data in enumerate(validation_loader):
        # Load data
        image, tof, label_yaw = data[0], data[1], data[2]
        image = image.numpy()
        image = tf.convert_to_tensor(image)
        tof = tof.numpy()
        tof = tf.convert_to_tensor(tof)
        label_yaw = label_yaw.numpy()
        label_yaw = tf.convert_to_tensor(label_yaw)

        # permute from NCWH to NWHC the input
        image = kr.layers.Permute((2, 3, 1))(image)
        tof = kr.layers.Permute((2, 3, 1))(tof)

        # Validation
        pred_yaw = model((image, tof))
        loss_yaw = (pred_yaw-label_yaw)**2

        val_rmse_yaw += tf.reduce_sum(loss_yaw)
        iterations += tf.cast(tf.size(label_yaw), tf.float32)
    val_rmse_yaw = tf.sqrt(val_rmse_yaw/iterations)
    print("\nRMSE over the whole validation dataset:",val_rmse_yaw.numpy(),"\n")

    return model


def fine_tuning(model, train_loader, validation_loader, config):
   # Load parameters
    use_wandb = config.getboolean("WANDB", "USE_WANDB")
    lr = float(config["TRAINING_NAVIGATION"]["LEARNING_RATE"])
    lr_decay = float(config["TRAINING_NAVIGATION"]["LEARNING_RATE_DECAY"])
    epochs_training = int(config["TRAINING_NAVIGATION"]["EPOCHS"])
    epochs_tuning = int(config["FINE_TUNING_NAVIGATION"]["EPOCHS"])
    
    model = tfmot.quantization.keras.quantize_model(model)
    
    # Optimizer
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, epsilon=1e-08, weight_decay=lr_decay)

    # error
    train_loss = kr.metrics.RootMeanSquaredError(name='train_loss')
    validation_loss = kr.metrics.RootMeanSquaredError(name='validation_loss')
    loss_function = kr.losses.MeanSquaredError()

    # ############################################################################
    # # Train/Val Loop
    # ############################################################################
    best_val_loss = float('inf')  # Used for saving best model of this run
    for epoch in range(epochs_tuning):
        print('\nEpoch: ', epoch + 1, ' / ', epochs_tuning)

        # Average train losses + reset_errors
        train_loss.reset_states()
        validation_loss.reset_states()

        num_train_iterations = len(train_loader)

        # Training Loop
        train_rmse_yaw = 0.0
        with tqdm(total=num_train_iterations, desc='Train', disable=not True) as t:
            for batch_idx, data in enumerate(train_loader):
                # Load data from torch and modify to tensorflow, passing by numpy
                image, tof, label_yaw = data[0], data[1], data[2]
                image = image.numpy()
                image = tf.convert_to_tensor(image)
                tof = tof.numpy()
                tof = tf.convert_to_tensor(tof)
                label_yaw = label_yaw.numpy()
                label_yaw = tf.convert_to_tensor(label_yaw)

                # permute from NCWH to NWHC the input, as required by Tensorflow
                image = kr.layers.Permute((2, 3, 1))(image)
                tof = kr.layers.Permute((2, 3, 1))(tof)

                with tf.GradientTape() as tape:
                    # Forward pass
                    pred_yaw = model((image, tof), training=True)
                    # Loss computation
                    loss_yaw = tf.sqrt(loss_function(label_yaw, pred_yaw))
                    # Save avg train rmse losses
                    train_rmse_yaw += loss_yaw.numpy()

                # Backward pass
                gradients = tape.gradient(loss_yaw, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                t.update(1)

        # Validation
        # Average validation losses
        tot_loss_yaw = 0.0
        iterations = 0
        for batch_idx, data in enumerate(validation_loader):
            # Load data
            image, tof, label_yaw = data[0], data[1], data[2]
            image = image.numpy()
            image = tf.convert_to_tensor(image)
            tof = tof.numpy()
            tof = tf.convert_to_tensor(tof)
            label_yaw = label_yaw.numpy()
            label_yaw = tf.convert_to_tensor(label_yaw)

            # permute from NCWH to NWHC the input
            image = kr.layers.Permute((2, 3, 1))(image)
            tof = kr.layers.Permute((2, 3, 1))(tof)

            # Validation done using RMSE loss over the whole dataset
            pred_yaw = model((image, tof))
            tot_loss_yaw += tf.reduce_sum((pred_yaw - label_yaw) ** 2).numpy()
            iterations += tf.size(pred_yaw)

        # Compute average losses at end of epoch
        train_rmse_yaw /= num_train_iterations
        val_rmse_yaw = numpy.sqrt(tot_loss_yaw / iterations)

        # Save model if better val loss than previous best val loss --> used later to upload to wandb
        if val_rmse_yaw < best_val_loss:
            # Remove previous best if it exists
            if os.path.exists('throwaway_models/best_model_gate_navigator' + unique_run_id):
                print('Deleted model dict')
                shutil.rmtree('throwaway_models/best_model_gate_navigator' + unique_run_id)

            # Save
            kr.models.save_model(model=model, filepath='throwaway_models/best_model_gate_navigator' +
                                                                      unique_run_id, save_format="tf")
            print('Saved new best model')

            # Update new best loss
            best_val_loss = val_rmse_yaw

        print('\n')
        print('Avg Loss Yaw Train / Val: ', train_rmse_yaw, ' / ', val_rmse_yaw)
        print('-------------------------------------------------------------')

        if use_wandb:
            wandb.log({'epoch': epoch + epochs_training, 'train_rmse_yaw': train_rmse_yaw, 'val_rmse_yaw': val_rmse_yaw})
       

    if use_wandb:
        # Log best model to wandb artifact
        artifact = wandb.Artifact('best_model_gate_navigator', type='full_state_dict')
        artifact.add_dir('throwaway_models/best_model_gate_navigator' + unique_run_id)
        wandb.log_artifact(artifact).wait()

        # Get the version of the saved artifact
        artifact_version = artifact.version
        wandb.finish()

    #convert the dataset
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Sub-set of the training dataset used for tuning the model
    def representative_dataset():
        for batch_idx_rd, data_rd in enumerate(train_loader):
            image_rd, tof_rd = data_rd[0], data_rd[1]
            image_rd = image_rd.numpy()
            tof_rd = tof_rd.numpy()
            image_rd = numpy.transpose(image_rd, (0, 2, 3, 1))
            image_rd = numpy.resize(image_rd, (1, 168, 168, 1))
            tof_rd = numpy.transpose(tof_rd, (0, 2, 3, 1))
            tof_rd = numpy.resize(tof_rd, (1, 21, 21, 1))
            yield [image_rd, tof_rd]

    converter.representative_dataset = representative_dataset
    model = converter.convert()

    file_path_saved_model = 'tflite_models/gate_navigator_model_' + artifact_version + '.tflite'
    print("\nModel .tflite saved in: ", file_path_saved_model, "\n")
    with open(file_path_saved_model, 'wb') as f:
        f.write(model)

    # Test quantized tflite model over the validation dataset
    tf.lite.experimental.Analyzer.analyze(model_content=model)
    interpreter = tf.lite.Interpreter(model_content=model)
    image_details = interpreter.get_input_details()[0]
    tof_details = interpreter.get_input_details()[1]

    output_details = interpreter.get_output_details()[0]
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    test_rmse_loss = 0.0
    iterations = 0

    for batch_idx, data in enumerate(validation_loader):
        # Load data
        image, tof, label_yaw = data[0], data[1], data[2]
        image = image.numpy()
        tof = tof.numpy()
        label_yaw = label_yaw.numpy()
        for i in range(label_yaw.size):
            # Iterate over each one of the inputs/ouputs of a batch
            label_yaw_it = label_yaw[i, 0]
            image_it = numpy.transpose(image, (0, 2, 3, 1))
            image_it = image_it[i, :, :, :]
            tof_it = numpy.transpose(tof, (0, 2, 3, 1))
            tof_it = tof_it[i, :, :, :]

            input_scale, input_zero_point = image_details["quantization"]
            image_it = image_it / input_scale + input_zero_point
            input_scale, input_zero_point = tof_details["quantization"]
            tof_it = tof_it / input_scale + input_zero_point
            image_it = numpy.expand_dims(image_it, axis=0).astype(image_details["dtype"])
            tof_it = numpy.expand_dims(tof_it, axis=0).astype(tof_details["dtype"])
            interpreter.set_tensor(image_details["index"], image_it)
            interpreter.set_tensor(tof_details["index"], tof_it)
            interpreter.invoke()
            pred_yaw = interpreter.get_tensor(output_details["index"])[0]
            ouput_scale, output_zero_point = output_details["quantization"]
            float_pred_yaw = (pred_yaw.astype(numpy.float32) - output_zero_point) * ouput_scale
            loss_yaw = (float_pred_yaw - label_yaw_it) ** 2
            test_rmse_loss += loss_yaw
            iterations += 1

    # RMSE loss
    test_rmse_loss = numpy.sqrt(test_rmse_loss / iterations)
    print("\nInference on the quantized tflite model on 100% of the validation dataset resulted in the following loss:",
          test_rmse_loss[0], "\n")

    # Save the new version index of the model in the config file

    config.set("QUANTIZATION_NAVIGATION", "model_identifier", artifact_version)
    with open('deep_learning_config.ini', 'w') as configfile:
        config.write(configfile)



def process_loaders(config):
    
    # Load parameters
    data_loading_path = os.getcwd() + '/../' + config["DATA_PATHS"]["DATA_LOADING_PATH_NAVIGATION"]
    num_workers = int(config["TRAINING_NAVIGATION"]["NUM_WORKERS"])
    batch_size = int(config["TRAINING_NAVIGATION"]["BATCH_SIZE"])
    mean_image = float(config["NORMALIZATION"]["MEAN_IMAGE"])
    std_image = float(config["NORMALIZATION"]["STD_IMAGE"])
    mean_tof = float(config["NORMALIZATION"]["MEAN_TOF"])
    std_tof = float(config["NORMALIZATION"]["STD_TOF"])
    
    # preprare images and ToF fow training and validation
    standartizer_image = transforms.Normalize(mean=[mean_image],
                                              std=[std_image])  # Determined from: from utility import batch_mean_and_sd
    standartizer_tof = transforms.Normalize(mean=[mean_tof],
                                            std=[std_tof])  # Determined from: from utility import batch_mean_and_sd
    color_jitter_image = transforms.ColorJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0))
    gaussian_blur_image = transforms.GaussianBlur(kernel_size=(3, 5))
    random_invert_image = transforms.RandomInvert()
    random_adjust_sharpness_image = transforms.RandomAdjustSharpness(sharpness_factor=10)
    transforms_image_train = [transforms.ToTensor(), color_jitter_image, gaussian_blur_image, random_invert_image,
                              random_adjust_sharpness_image, standartizer_image]
    transforms_image_val = [transforms.ToTensor(), standartizer_image]
    transforms_tof = [transforms.ToTensor(), standartizer_tof]

    # Create dataloader for training and validation dataset
    train_dataset = ImavChallengeNavigationDataset(root=data_loading_path + 'training/',
                                                   transforms_image=transforms_image_train,
                                                   transforms_tof=transforms_tof)

    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)

    validation_dataset = ImavChallengeNavigationDataset(root=data_loading_path + 'validation/',
                                                        transforms_image=transforms_image_val,
                                                        transforms_tof=transforms_tof)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=num_workers)
    return train_loader, validation_loader


def training_gate_navigator():
    # Load config file
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("deep_learning_config.ini")

    print("Tensorflow found the following GPUs :\n", tf.config.list_physical_devices('GPU'))

    train_loader,validation_loader = process_loaders(config) 
    
    print("\n\n#Training using classic method\n\n")

    model = training(train_loader=train_loader, validation_loader=validation_loader, config=config)


    print("\n\n#Fine tuning\n\n")
    fine_tuning(model=model, train_loader=train_loader,validation_loader=validation_loader, config=config)
