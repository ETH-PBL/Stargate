"""
This script provides utility functions for the deep learning process

author: Alberto Schiaffino, adapted from Konstantin Kalenberg
"""

# essentials
import os
import numpy as np


# torch
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from nntool.api import NNGraph
import pickle


################################################################################
# Navigation Data Loader Class
################################################################################
class ImavChallengeNavigationDataset(Dataset):
    """ImavChallengeNavigationDataset dataset."""
    def __init__(self, root, transforms_image=None, transforms_tof=None):
        """
        Args:
            root (string): Top level directory containing all runs of respective training/validation/testing folder, expects trailing slash
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform_image = transforms.Compose(transforms_image)
        self.transform_tof = transforms.Compose(transforms_tof)
        self.filenames_image = []
        self.filenames_tof = []
        self.filenames_labels = []

        existing_runs_int = [int(run_number) for run_number in os.listdir(root)]
        for current_run in existing_runs_int:
            current_run_path = root + str(current_run) + '/'
            existing_data_points_int = [int(os.path.splitext(name)[0]) for name in os.listdir(current_run_path + 'camera_images/')]
            existing_data_points_int.sort()
            for current_data_point in existing_data_points_int:
                data_point_name = str(current_data_point) + '.npy'
                self.filenames_image.append(current_run_path + 'camera_images/' + data_point_name)
                self.filenames_tof.append(current_run_path + 'tof_distance_array/' + data_point_name)
                self.filenames_labels.append(current_run_path + 'label_yaw_rate_desired/' + data_point_name)

    def __len__(self):
        if len(self.filenames_image) == len(self.filenames_tof) == len(self.filenames_labels):
            return len(self.filenames_image)
        else:
            print("Dataset size error: ", len(self.filenames_image), "/", len(self.filenames_tof), "/", len(self.filenames_labels))


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = np.load(self.filenames_image[idx])
        tof = np.load(self.filenames_tof[idx])
        label_yaw = np.load(self.filenames_labels[idx])

        # # Uncomment for vizualizing of transforms
        # import cv2
        # prev_image = image

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_tof:
            tof = self.transform_tof(tof)
        label_yaw = torch.tensor(label_yaw, dtype=torch.float32)

        # # Uncomment for vizualizing of transforms
        # transformed_image = np.squeeze(image.numpy(), axis=0)
        # transformed_image = (transformed_image * 255).astype(np.uint8)
        # show_image = cv2.hconcat([prev_image, transformed_image])
        # cv2.imshow('original/transformed', show_image)
        # cv2.waitKey(0)

        sample = [image, tof, label_yaw]
        return sample


################################################################################
# Classification Data Loader Class
################################################################################
class ImavChallengeClassificationDataset(Dataset):
    """ImavChallengeClassificationDataset dataset."""
    def __init__(self, root, transforms_image=None, transforms_tof=None):
        """
        Args:
            root (string): Top level directory containing all runs of respective training/validation/testing folder, expects trailing slash
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform_image = transforms.Compose(transforms_image)
        self.transform_tof = transforms.Compose(transforms_tof)
        self.filenames_image = []
        self.filenames_tof = []
        self.labels = []

        existing_runs_int_gate = [int(run_number) for run_number in os.listdir(root + 'gate/')]
        for current_run in existing_runs_int_gate:
            current_run_path = root + 'gate/' + str(current_run) + '/'
            existing_data_points_int = [int(os.path.splitext(name)[0]) for name in os.listdir(current_run_path + 'camera_images/')]
            existing_data_points_int.sort()
            for current_data_point in existing_data_points_int:
                data_point_name = str(current_data_point) + '.npy'
                self.filenames_image.append(current_run_path + 'camera_images/' + data_point_name)
                self.filenames_tof.append(current_run_path + 'tof_distance_array/' + data_point_name)
                self.labels.append(1)

        existing_runs_int_no_gate = [int(run_number) for run_number in os.listdir(root + 'no_gate/')]
        for current_run in existing_runs_int_no_gate:
            current_run_path = root + 'no_gate/' + str(current_run) + '/'
            existing_data_points_int = [int(os.path.splitext(name)[0]) for name in os.listdir(current_run_path + 'camera_images/')]
            existing_data_points_int.sort()
            for current_data_point in existing_data_points_int:
                data_point_name = str(current_data_point) + '.npy'
                self.filenames_image.append(current_run_path + 'camera_images/' + data_point_name)
                self.filenames_tof.append(current_run_path + 'tof_distance_array/' + data_point_name)
                self.labels.append(0)

    def __len__(self):
        if len(self.filenames_image) == len(self.filenames_tof) == len(self.labels):
            return len(self.filenames_image)
        else:
            print("Dataset size error: ", len(self.filenames_image), "/", len(self.filenames_tof), "/", len(self.labels))


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = np.load(self.filenames_image[idx])
        tof = np.load(self.filenames_tof[idx])
        label = self.labels[idx]

        # # Uncomment for vizualizing of transforms
        # import cv2
        # prev_image = image

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_tof:
            tof = self.transform_tof(tof)
        label = torch.tensor(label, dtype=torch.float32)

        # # Uncomment for vizualizing of transforms
        # transformed_image = np.squeeze(image.numpy(), axis=0)
        # transformed_image = (transformed_image * 255).astype(np.uint8)
        # show_image = cv2.hconcat([prev_image, transformed_image])
        # cv2.imshow('original/transformed', show_image)
        # cv2.waitKey(0)

        sample = [image, tof, label]
        return sample


################################################################################
# Multi Epochs Data Loader Class (Adapted from Liam Boyle)
################################################################################
class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """MultiEpochsDataLoader dataloader."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


################################################################################
# Custom Losses
################################################################################
def custom_rmse_loss_PyTorch(label_yaw, pred_yaw, mse_loss):
    output_rmse_yaw_rate = torch.sqrt(mse_loss(pred_yaw, label_yaw))

    return output_rmse_yaw_rate

def custom_rmse_loss_TensorFlow(label_yaw, pred_yaw, rmse_loss):

    output_rmse_yaw_rate = rmse_loss(pred_yaw, label_yaw)
    return output_rmse_yaw_rate
def custom_mse_loss(label_yaw, pred_yaw, mse_loss):
    output_mse_yaw_rate = mse_loss(pred_yaw, label_yaw)

    return output_mse_yaw_rate

def rmse_loss(labels, preds):
    return np.sqrt(np.mean((preds - labels) ** 2))

def custom_bce_loss(label, pred, bce_loss):
    label = label[:, None]  # Fix since predictions already have correct torch dimension [batch_size, 1] but labels are [batch_size,]
    output_bce_loss = bce_loss(pred, label)

    return output_bce_loss

def custom_accuracy_loss(label, pred, accuracy_loss):
    label = label[:, None]  # Fix since predictions already have correct torch dimension [batch_size, 1] but labels are [batch_size,]
    output_accuracy_loss = accuracy_loss(pred, label)

    return output_accuracy_loss

def custom_f1_loss(label, pred, binary_f1_loss):
    label = label[:, None]  # Fix since predictions already have correct torch dimension [batch_size, 1] but labels are [batch_size,]
    output_f1_loss = binary_f1_loss(pred, label)

    return output_f1_loss

def custom_auroc(label, pred, binary_auroc):
    label = label[:, None]  # Fix since predictions already have correct torch dimension [batch_size, 1] but labels are [batch_size,]
    output_auroc = binary_auroc(pred, label)

    return output_auroc

def accuracy_loss(labels, preds):
    return (labels == preds).sum() / labels.shape[0]

################################################################################
# Compute mean and std of images and ToF in dataset
################################################################################
def batch_mean_and_sd(data_loader):
    cnt_im = 0
    fst_moment_im = torch.empty(1)
    snd_moment_im = torch.empty(1)

    cnt_tof = 0
    fst_moment_tof = torch.empty(1)
    snd_moment_tof = torch.empty(1)

    for data in data_loader:
        images = data[0]
        b, c, h, w = images.shape
        nb_pixels_im = b * h * w
        sum = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment_im = (cnt_im * fst_moment_im + sum) / (cnt_im + nb_pixels_im)
        snd_moment_im = (cnt_im * snd_moment_im + sum_of_square) / (cnt_im + nb_pixels_im)
        cnt_im += nb_pixels_im

        tofs = data[1]
        b, c, h, w = tofs.shape
        nb_pixels_tof = b * h * w
        sum = torch.sum(tofs, dim=[0, 2, 3])
        sum_of_square = torch.sum(tofs ** 2, dim=[0, 2, 3])
        fst_moment_tof = (cnt_tof * fst_moment_tof + sum) / (cnt_tof + nb_pixels_tof)
        snd_moment_tof = (cnt_tof * snd_moment_tof + sum_of_square) / (cnt_tof + nb_pixels_tof)
        cnt_tof += nb_pixels_tof

    mean_im, std_im = fst_moment_im, torch.sqrt(snd_moment_im - fst_moment_im ** 2)
    mean_tof, std_tof = fst_moment_tof, torch.sqrt(snd_moment_tof - fst_moment_tof ** 2)

    print('Mean/std images: ', mean_im, ' / ', std_im)
    print('Mean/std ToF: ', mean_tof, ' / ', std_tof)

################################################################################
#This function transforms an np.array image from the range [0,255] to the range [0, 1] and then standardizes both image and tof
################################################################################
def standardize_camera_tof_sample(image, tof, mean_image, std_image, mean_tof, std_tof):

    image = ((image / 255.0) - mean_image) / std_image
    tof = (tof - mean_tof) / std_tof
    return image, tof


def nn_tool_get_class_model(model_loading_path, model_identifier, quantize):

    loading_path_model = model_loading_path + 'gate_classifier_model_' + model_identifier + '.onnx'
    loading_path_quant_stats_file = model_loading_path + 'quant_stats_gate_classifier_model_' + model_identifier + '.json'

    model = NNGraph.load_graph(loading_path_model, load_quantization=False)
    model.adjust_order()
    dict_quantization = None
    if quantize:
        fp = open(loading_path_quant_stats_file, 'rb')
        astats = pickle.load(fp)
        fp.close()
        model.fusions('scaled_match_group')

        model.quantize(statistics=astats, schemes=['scaled'])
        print(model.qshow())

        quantization = model.quantization
        dict_quantization = {
        "input_1_zero" : quantization['input_1'].out_qs[0]._zero_point[0],
        "input_1_scale" : quantization['input_1'].out_qs[0]._scale[0],
        "input_2_zero" : quantization['input_2'].out_qs[0]._zero_point[0],
        "input_2_scale" : quantization['input_2'].out_qs[0]._scale[0],
        "output_zero" : quantization['output_1'].out_qs[0]._zero_point[0],
        "output_scale" : quantization['output_1'].out_qs[0]._scale[0]
    }
    else:
        model.fusions()
        model.quantization = None

    #print(model.show())
    return model, dict_quantization

def nn_tool_get_navigation_model(model_identifier,  model_loading_path):
    loading_path_model = model_loading_path + 'gate_navigator_model_' + model_identifier + '.tflite'
    model = NNGraph.load_graph(loading_path_model, load_quantization=True)
    model.adjust_order()
    model.fusions()
    print(model.qshow())

    quantization = model.quantization
    dict_quantization = {
        "input_1_zero" : quantization['input_1'].out_qs[0]._zero_point[0],
        "input_1_scale" : quantization['input_1'].out_qs[0]._scale[0],
        "input_2_zero" : quantization['input_2'].out_qs[0]._zero_point[0],
        "input_2_scale" : quantization['input_2'].out_qs[0]._scale[0],
        "output_zero" : quantization['output_1'].out_qs[0]._zero_point[0],
        "output_scale" : quantization['output_1'].out_qs[0]._scale[0]
    }

    return model, dict_quantization
