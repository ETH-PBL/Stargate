"""
Script to vizualize dataset distribution and single runs and moments

author: Konstantin Kalenberg
"""

import configparser
import os
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
# import scienceplots
# plt.style.use('science')

# INSTRUCTIONS
# Cycle through runs by pressing random keys
#   --> Abort with 'ESC' key
#   --> Go into detail view of run py pressing 's' key
#       --> Jump back out to runs by pressing 'ESC' key
#       --> Delete folder of current run by pressing 'x' key
#       --> At end of run we jump back out to runs

def data_visualizer():
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("../config.ini")
    data_loading_path = config["CF_CONTROLLER_PY"]["DATA_LOADING_PATH"]

    existing_runs_int = [int(run_number) for run_number in os.listdir(data_loading_path)]
    existing_runs_int.sort()

    # Go through all runs and select a run for further inspection
    desired_starting_index = 0

    try:
        current_run_index = existing_runs_int.index(desired_starting_index)
    except:
        current_run_index = existing_runs_int.index(min(existing_runs_int))

    while current_run_index < len(existing_runs_int):
        current_run = existing_runs_int[current_run_index]
        current_run_path = data_loading_path + str(current_run) + '/'
        setup_img = cv2.imread(current_run_path + 'setup_img.png')

        inspect_further = False

        # Display run overview
        cv2.imshow('Setup image of run in folder: ' + str(current_run), setup_img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            quit()
        elif key == ord('s'):
            inspect_further = True
            cv2.destroyAllWindows()

        # Go through selected run in detail
        if inspect_further:
            existing_data_points_int = [int(os.path.splitext(run_number)[0]) for run_number in os.listdir(current_run_path + '/camera_images/')]
            existing_data_points_int.sort()
            for run_number in existing_data_points_int:
                if not (run_number % 8 == 0):
                     continue

                # # Uncomment for removing last few runs for each folder
                # remove_last_n = 8
                # if (len(existing_data_points_int) - run_number <= remove_last_n):
                #     data_point = str(run_number) + '.npy'
                #     os.remove(current_run_path + 'camera_images/' + data_point)
                #     os.remove(current_run_path + 'tof_distance_array/' + data_point)
                #     os.remove(current_run_path + 'tof_validity_array/' + data_point)
                #     os.remove(current_run_path + 'roll_pitch_yaw/' + data_point)
                #     os.remove(current_run_path + 'linear_velocities/' + data_point)
                #     os.remove(current_run_path + 'angular_velocities/' + data_point)
                #     os.remove(current_run_path + 'linear_accelerations/' + data_point)
                #     os.remove(current_run_path + 'label_forward_velocity_desired/' + data_point)
                #     os.remove(current_run_path + 'label_yaw_rate_desired/' + data_point)
                # continue

                data_point = str(run_number) + '.npy'
                camera_image = np.load(current_run_path + 'camera_images/' + data_point)
                tof_distance_array = np.load(current_run_path + 'tof_distance_array/' + data_point)
                tof_validity_array = np.load(current_run_path + 'tof_validity_array/' + data_point)
                roll_pitch_yaw = np.load(current_run_path + 'roll_pitch_yaw/' + data_point)
                linear_velocities = np.load(current_run_path + 'linear_velocities/' + data_point)
                angular_velocities = np.load(current_run_path + 'angular_velocities/' + data_point)
                linear_accelerations = np.load(current_run_path + 'linear_accelerations/' + data_point)
                label_forward_velocity_desired = np.load(current_run_path + 'label_forward_velocity_desired/' + data_point)
                label_yaw_rate_desired = np.load(current_run_path + 'label_yaw_rate_desired/' + data_point)

                # Only crop images if they have not been cropped before
                if camera_image.shape[0] > 168:
                    # Crop camera image size, crop image rows by preserving bottom rows (to overlap FOV with ToF), crop image cols from center
                    camera_image = camera_image[0:244, :]
                    rows, cols = camera_image.shape
                    start_row = rows - 168
                    start_col = cols // 2 - (168 // 2)
                    camera_image = camera_image[start_row:start_row + 168, start_col:start_col + 168]

                # Create visualizable images
                camera_image_show = cv2.cvtColor(camera_image, cv2.COLOR_GRAY2RGB)

                tof_distance_array_show = (tof_distance_array * 255 / 3).astype(np.uint8)
                tof_distance_array_show = cv2.cvtColor(tof_distance_array_show, cv2.COLOR_GRAY2RGB)
                for row in range(0, tof_distance_array_show.shape[0]):
                    for col in range(0, tof_distance_array_show.shape[1]):
                        if tof_validity_array[row, col] == False:
                            tof_distance_array_show[row, col, 0] = 0
                            tof_distance_array_show[row, col, 1] = 0
                            tof_distance_array_show[row, col, 2] = 255

                tof_distance_array_show = cv2.resize(tof_distance_array_show, dsize=(camera_image.shape[0], camera_image.shape[0]), interpolation=cv2.INTER_NEAREST)

                show_img = cv2.hconcat([camera_image_show, tof_distance_array_show])

                cv2.imshow('Camera image and ToF image of run: ' + str(current_run) + ' / data_point: ' + data_point + ' of total: ' + str(len(existing_runs_int)), show_img)

                print('Data for run: ' + str(current_run) + ' data_point: ' + data_point + ' of total: ' + str(len(existing_runs_int)))
                print('roll_pitch_yaw: ', roll_pitch_yaw)
                print('linear_velocities: ', linear_velocities)
                print('angular_velocities: ', angular_velocities)
                print('linear_accelerations: ', linear_accelerations)
                print('label_forward_velocity_desired: ', label_forward_velocity_desired)
                print('label_yaw_rate_desired: ', label_yaw_rate_desired)

                key_1 = cv2.waitKey(0)
                # Jump back to run selection menu
                if key_1 == 27:  # ESC
                    cv2.destroyAllWindows()
                    print('##################################################################')
                    break
                # Delete run folder
                elif key_1 == ord('x'):
                    cv2.destroyAllWindows()
                    print('Deleting data collection folder of run: ' + str(current_run))
                    print('##################################################################')
                    shutil.rmtree(current_run_path)
                    break
                print('##################################################################')
                cv2.destroyAllWindows()

        # Go to next run
        current_run_index += 1
        cv2.destroyAllWindows()

def data_distribution_visualizer():
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("../config.ini")
    data_loading_path = config["CF_CONTROLLER_PY"]["DATA_LOADING_PATH"]

    existing_runs_int = [int(run_number) for run_number in os.listdir(data_loading_path)]
    existing_runs_int.sort()

    forward_velocities = list()
    yaw_rates = list()

    bins_vel = np.linspace(0.0, 2.0, 21)
    bins_yaw = np.linspace(-1.0, 1.0, 21)

    # Go through all runs and select a run for further inspection
    current_run_index = existing_runs_int.index(min(existing_runs_int))
    while current_run_index < len(existing_runs_int):
        current_run = existing_runs_int[current_run_index]
        current_run_path = data_loading_path + str(current_run) + '/'

        existing_data_points_int = [int(os.path.splitext(run_number)[0]) for run_number in os.listdir(current_run_path + '/camera_images/')]
        existing_data_points_int.sort()
        for current_moment in existing_data_points_int:
            data_point = str(current_moment) + '.npy'
            forward_velocities.append(np.load(current_run_path + 'label_forward_velocity_desired/' + data_point)[0])
            yaw_rates.append(np.load(current_run_path + 'label_yaw_rate_desired/' + data_point)[0])
        current_run_index += 1

    print('Total num datapoints vel: ', len(forward_velocities))
    print('Total num datapoints yaw: ', len(yaw_rates))

    # Prepare plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Visualize forward velocity probabilities
    forward_velocities_histogram, bins_vel = np.histogram(forward_velocities, bins_vel)
    forward_velocities_histogram = forward_velocities_histogram / len(forward_velocities)
    bins_vel_bar = [x + 0.05 for x in bins_vel]
    ax1.bar(bins_vel_bar[:-1], forward_velocities_histogram, width=0.09)
    ax1.set_xlim([bins_vel[0]-0.05, bins_vel[-1]+0.05])
    ax1.set_ylim([0, 0.16])
    ax1.set_title('Velocity probability distribution', fontsize=30)
    ax1.set_xlabel('Velocities [m/s]', fontsize=30)
    ax1.set_ylabel('Probability', fontsize=30)
    ax1.set_xticks(bins_vel[::2])
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)

    print('Forward velocities histogram: ', forward_velocities_histogram)
    print('Bins vel: ', bins_vel)

    # Visualize yaw rate probabilities
    yaw_rates_histogram, bins_yaw = np.histogram(yaw_rates, bins_yaw)
    yaw_rates_histogram = yaw_rates_histogram / len(yaw_rates)
    bins_yaw_bar = [x + 0.05 for x in bins_yaw]
    ax2.bar(bins_yaw_bar[:-1], yaw_rates_histogram, width=0.09)
    ax2.set_xlim([bins_yaw[0]-0.05, bins_yaw[-1]+0.05])
    ax2.set_ylim([0, 0.16])
    ax2.set_title('Yaw rate probability distribution', fontsize=30)
    ax2.set_xlabel('Yaw rates [rad/s]', fontsize=30)
    ax2.set_ylabel('Probability', fontsize=30)
    ax2.set_xticks(bins_yaw[::2])
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)

    print('Forward velocities histogram: ', yaw_rates_histogram)
    print('Bins vel: ', bins_yaw)

    # Visualize forward velocity and yaw rate probabilities combined
    hist_2d, bins_vel, bins_yaw = np.histogram2d(forward_velocities, yaw_rates, bins=[bins_vel, bins_yaw])
    hist_2d = hist_2d / len(forward_velocities)
    hist_2d = hist_2d.T
    ax3.imshow(hist_2d, interpolation='nearest', origin='lower', extent=[bins_vel[0], bins_vel[-1], bins_yaw[0], bins_yaw[-1]])
    ax3.set_title('Combined probability distribution', fontsize=30)
    ax3.set_xlabel('Velocities [m/s]', fontsize=30)
    ax3.set_ylabel('Yaw rates [rad/s]', fontsize=30)
    ax3.set_xticks(bins_vel[::2])
    ax3.set_yticks(bins_yaw[::2])
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)


    fig.set_size_inches(20, 12, forward=True)
    fig.savefig('dataset_distribution.png')
    #fig.show()

def approach_angle_distribution_visualizer():
    approach_angle_bins = np.linspace(0, 60, 7)
    approach_angle_bins = [x + 5 for x in approach_angle_bins]

    # Approach angle distribution sim unadapted (no NN split) v11
    # hist_success = [64, 52, 59, 56, 40, 43]  #, 36, 34, 8]
    # hist_crash_gate = [0, 1, 2, 5, 16, 9]    #, 11, 7, 14]
    # hist_crash_object = [1, 5, 1, 8, 4, 10]  #, 25, 31, 36]

    # Approach angle distribution sim adapted (no NN split) v11
    # hist_success = [4, 5, 4, 3, 2, 4]
    # hist_crash_gate = [2, 1, 4, 2, 1, 3]
    # hist_crash_object = [10, 4, 9, 15, 13, 17]

    # Approach angle distribution real (no NN split) v11
    # hist_success = [5, 5, 4, 2, 3, 0]
    # hist_crash_gate = [4, 3, 4, 5, 3, 3]
    # hist_crash_object = [2, 3, 3, 5, 5, 6]

    ##### Training image augmentations experiments #####
    # Approach angle distribution sim v2 0.01 and 0.16 exposure, no lights  --> trained without augmentations
    # hist_success = [19, 11, 10, 8, 4, 6]
    # hist_crash_gate = [0, 1, 0, 2, 0, 3]
    # hist_crash_object = [21, 15, 11, 29, 13, 27]

    # Approach angle distribution sim v4 0.01 and 0.16 exposure, no lights  --> trained with augmentations
    # hist_success = [16, 24, 27, 15, 22, 15]
    # hist_crash_gate = [1, 0, 0, 0, 2, 0]
    # hist_crash_object = [5, 3, 9, 14, 13, 18]

    # Approach angle distribution sim v5 (same as v4, just sanity check) 0.01 and 0.16 exposure, no lights  --> trained with augmentations
    # hist_success = [14, 16, 13, 13, 12, 11]
    # hist_crash_gate = [0, 0, 0, 1, 2, 1]
    # hist_crash_object = [2, 7, 6, 10, 10, 9]

    # Approach angle distribution sim v5 0.01 and 0.16 exposure, no lights. Live computed image standardizer values  --> trained with augmentations
    # hist_success = [33, 31, 44, 49, 34, 29]
    # hist_crash_gate = [1, 0, 1, 0, 3, 3]
    # hist_crash_object = [1, 5, 4, 5, 10, 17]

    # Approach angle distribution sim v6 (normalized, not standardized) 0.01 and 0.16 exposure, no lights  --> trained with augmentations
    # hist_success = [47, 57, 51, 47, 57, 47]
    # hist_crash_gate = [1, 0, 1, 2, 1, 6]
    # hist_crash_object = [19, 30, 23, 24, 25, 39]

    # Make probabilities
    total_datapoints = [sum(x) for x in zip(hist_success, hist_crash_gate, hist_crash_object)]
    hist_success = [hist_success[i] / total_datapoints[i] for i in range(0, len(hist_success))]
    hist_crash_gate = [hist_crash_gate[i] / total_datapoints[i] + hist_success[i] for i in range(0, len(hist_crash_gate))]
    hist_crash_object = [hist_crash_object[i] / total_datapoints[i] + hist_crash_gate[i] for i in range(0, len(hist_crash_object))]

    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, (ax1) = plt.subplots(1, 1)

    ax1.bar(approach_angle_bins[:-1], hist_crash_object, width=8, label='gate miss')
    ax1.bar(approach_angle_bins[:-1], hist_crash_gate, width=8, label='gate crash')
    ax1.bar(approach_angle_bins[:-1], hist_success, width=8, label='gate pass')

    ax1.set_xlim([approach_angle_bins[0]-6.5, approach_angle_bins[-2]+6.5])
    ax1.set_ylim([0, 1.05])
    ax1.set_title('Success', fontsize=30)
    ax1.set_ylabel('Probability', fontsize=30)
    ax1.set_xlabel('Unsigned Relative Gate Approach Angle [deg]', fontsize=30)
    ax1.set_xticks(approach_angle_bins[:-1], fontsize=30)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.legend(prop={'size': 30})
    fig.set_size_inches(20, 12, forward=True)
    fig.savefig('approach_angle_distribution.png')

def main():
    #data_distribution_visualizer()
    #approach_angle_distribution_visualizer()
    data_visualizer()

if __name__ == "__main__":
    main()