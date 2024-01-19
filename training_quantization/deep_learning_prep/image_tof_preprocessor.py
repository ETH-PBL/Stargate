"""
Script to crop camera images, augment them to look more real and upsample ToF image
CAREFULL, THIS SCRIPT REPLACES THE IMAGES AND TOF IMAGES IN THE ORIGINAL FOLDER BY MODIFIED IMAGES AND TOF IMAGES

author: Konstantin Kalenberg, (get_vignette_mask() by Hanna Müller)
"""

import configparser
import os
import numpy as np
import cv2
import math


# Generates vignette mask of size width * height with gaussian kernel of sigma
def get_vignette_mask(rows, cols, sigma=150):
    kernel_x = cv2.getGaussianKernel(cols, sigma)
    kernel_y = cv2.getGaussianKernel(rows, sigma)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    return mask

def crop_and_augment_camera_images_upsample_tof(data_loading_path):
    # Params
    image_crop_width = 168
    image_crop_height = 168
    tof_width = 21
    tof_height = 21
    show_images = False

    existing_runs_int = [int(run_number) for run_number in os.listdir(data_loading_path)]
    existing_runs_int.sort()

    current_run_index = existing_runs_int.index(min(existing_runs_int))

    while current_run_index < len(existing_runs_int):
        current_run = existing_runs_int[current_run_index]
        current_run_path = data_loading_path + str(current_run) + '/'

        existing_data_points_int = [int(os.path.splitext(run_number)[0]) for run_number in os.listdir(current_run_path + '/camera_images/')]
        existing_data_points_int.sort()
        for run_number in existing_data_points_int:
            data_point = str(run_number) + '.npy'
            camera_image_original = np.load(current_run_path + 'camera_images/' + data_point)
            tof_distance_array = np.load(current_run_path + 'tof_distance_array/' + data_point)
            tof_validity_array = np.load(current_run_path + 'tof_validity_array/' + data_point)

            # Augment camera image to mimic real himax camera
            camera_image = camera_image_original
            rows, cols = camera_image_original.shape
            #camera_image = cv2.GaussianBlur(camera_image, (3, 3), 0)  # Not needed as real HIMAX is not that blurry
            vignetteMask = get_vignette_mask(rows, cols)
            camera_image = camera_image * vignetteMask[0:rows, 0:cols]
            camera_image = cv2.convertScaleAbs(camera_image)

            # Crop camera image to resemble original 324x244 by removing bottom rows following
            # page 10 of https://cdn.sparkfun.com/assets/7/f/c/8/3/HM01B0-MNA-Datasheet.pdf
            camera_image = camera_image[0:244, :]
            camera_image_original = camera_image_original[0:244, :]

            # Crop camera image size, crop image rows by preserving bottom rows (to overlap FOV with ToF), crop image cols from center
            rows, cols = camera_image.shape
            start_row = rows - image_crop_height
            start_col = cols // 2 - (image_crop_height // 2)
            camera_image = camera_image[start_row:start_row + image_crop_height, start_col:start_col + image_crop_width]
            camera_image_original = camera_image_original[start_row:start_row + image_crop_height, start_col:start_col + image_crop_width]

            # Upsample ToF and ToF Validity
            tof_distance_array = cv2.resize(tof_distance_array, dsize=(tof_width, tof_height), interpolation=cv2.INTER_NEAREST)

            tof_validity_array = tof_validity_array.astype(int)
            tof_validity_array = cv2.resize(tof_validity_array, dsize=(tof_width, tof_height), interpolation=cv2.INTER_NEAREST)
            tof_validity_array = tof_validity_array.astype(bool)

            # Overwrite augmented images
            np.save(current_run_path + 'camera_images/' + str(run_number), camera_image)
            np.save(current_run_path + 'tof_distance_array/' + str(run_number), tof_distance_array)
            np.save(current_run_path + 'tof_validity_array/' + str(run_number), tof_validity_array)

            if show_images:
                # Create visualizable images
                camera_image_show = None
                try:
                    camera_image_show = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2RGB)
                    camera_image_original = cv2.cvtColor(camera_image_original, cv2.COLOR_GRAY2RGB)
                except:
                    camera_image_show = camera_image
                tof_distance_array_show = (tof_distance_array * 255 / 3).astype(np.uint8)
                tof_distance_array_show = cv2.cvtColor(tof_distance_array_show, cv2.COLOR_GRAY2RGB)
                for row in range(0, tof_distance_array_show.shape[0]):
                    for col in range(0, tof_distance_array_show.shape[1]):
                        if tof_validity_array[row, col] == False:
                            tof_distance_array_show[row, col, 0] = 0
                            tof_distance_array_show[row, col, 1] = 0
                            tof_distance_array_show[row, col, 2] = 255

                tof_distance_array_show = cv2.resize(tof_distance_array_show, dsize=(camera_image.shape[1], camera_image.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST)

                show_img = cv2.hconcat([camera_image_original, camera_image_show, tof_distance_array_show])

                cv2.imshow('Camera image and ToF image of run: ' + str(current_run) + ' / data_point: ' + data_point + ' of total: ' + str(
                    len(existing_runs_int)), show_img)

                print('Data for run: ' + str(current_run) + ' data_point: ' + data_point + ' of total: ' + str(len(existing_runs_int)))

                key_1 = cv2.waitKey(0)
                # Jump back to run selection menu
                if key_1 == 27:  # ESC
                    cv2.destroyAllWindows()
                    print('##################################################################')
                    break
                print('##################################################################')
                cv2.destroyAllWindows()

        # Go to next run
        print('Finished run: ', current_run_index)
        current_run_index += 1

def main():
    # Dataset consists of different "runs", which contain sequences of "moments"
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("../config.ini")
    data_loading_path = config["CF_CONTROLLER_PY"]["DATA_LOADING_PATH"]

    crop_and_augment_camera_images_upsample_tof(data_loading_path)

if __name__ == "__main__":
    main()