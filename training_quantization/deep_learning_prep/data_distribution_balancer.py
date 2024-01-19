"""
Data distribution balancing script to ensure a more even distribution of training data

author: Konstantin Kalenberg
"""

import configparser
import numpy as np
import os
import random

def distribution_balancer():
    # Parameters
    higher_probability_of_retaining_beginning_datapoints = True
    desired_dataset_size = 70000

    scaling_probability_priority_moments = 2.0
    num_priority_moments_beginning = 80  # How many moments are getting a higher selection probability at beginning of run

    # Dataset consists of different "runs", which contain sequences of "moments"
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("../config.ini")
    data_loading_path = config["CF_CONTROLLER_PY"]["DATA_LOADING_PATH"]

    existing_runs_int = [int(run_number) for run_number in os.listdir(data_loading_path)]
    existing_runs_int.sort()

    deleted_datapoints = 0
    total_number_datapoints = 0
    runs_for_all_moments = list()

    # Go through all runs and collect the total number of datapoints
    for run_index in existing_runs_int:
        for moment_number in os.listdir(data_loading_path + str(run_index) + '/camera_images/'):
            total_number_datapoints += 1
            runs_for_all_moments.append(run_index)

    # Delete random datapoints until only total_number_of_datapoints datapoints are left
    while total_number_datapoints - deleted_datapoints > desired_dataset_size:
        random_run_index = random.randint(0, len(runs_for_all_moments)-1)
        random_run = runs_for_all_moments[random_run_index]

        moments_in_random_run = [int(os.path.splitext(moment_number)[0]) \
                                 for moment_number in os.listdir(data_loading_path + str(random_run) + '/camera_images/')]
        moments_in_random_run.sort()

        if len(moments_in_random_run) <= 1:  # To avoid zero divisions later
            continue
        selection_probability_per_moment = 1.0 / len(moments_in_random_run)
        if higher_probability_of_retaining_beginning_datapoints and \
                len(moments_in_random_run) > scaling_probability_priority_moments * num_priority_moments_beginning:
            selection_probability_priority_moments = scaling_probability_priority_moments * selection_probability_per_moment
            selection_probability_insubstantial_moments = (1 - selection_probability_priority_moments * num_priority_moments_beginning) / \
                                                          (len(moments_in_random_run) - num_priority_moments_beginning)
        else:
            selection_probability_priority_moments = selection_probability_per_moment
            selection_probability_insubstantial_moments = selection_probability_per_moment

        # Calculate the probability of deletion for each moment in the run and store in list
        prob_of_deletion = []
        for i in range(0, len(moments_in_random_run)):
            if moments_in_random_run[i] < num_priority_moments_beginning:
                prob_of_deletion.append(1 - selection_probability_priority_moments)
            else:
                prob_of_deletion.append(1 - selection_probability_insubstantial_moments)
        prob_of_deletion = [element / sum(prob_of_deletion) for element in prob_of_deletion]

        # Select index to delete
        index_moment_to_delete_in_random_run = np.random.choice(moments_in_random_run, 1, prob_of_deletion)
        index_moment_to_delete_in_random_run = str(index_moment_to_delete_in_random_run[0])

        # Delete everything related to this moment
        deleting_path = data_loading_path + str(random_run) + '/'
        os.remove(deleting_path + 'camera_images/' + index_moment_to_delete_in_random_run + '.npy')
        os.remove(deleting_path + 'tof_distance_array/' + index_moment_to_delete_in_random_run + '.npy')
        os.remove(deleting_path + 'tof_validity_array/' + index_moment_to_delete_in_random_run + '.npy')
        os.remove(deleting_path + 'roll_pitch_yaw/' + index_moment_to_delete_in_random_run + '.npy')
        os.remove(deleting_path + 'linear_velocities/' + index_moment_to_delete_in_random_run + '.npy')
        os.remove(deleting_path + 'angular_velocities/' + index_moment_to_delete_in_random_run + '.npy')
        os.remove(deleting_path + 'linear_accelerations/' + index_moment_to_delete_in_random_run + '.npy')
        os.remove(deleting_path + 'label_forward_velocity_desired/' + index_moment_to_delete_in_random_run + '.npy')
        os.remove(deleting_path + 'label_yaw_rate_desired/' + index_moment_to_delete_in_random_run + '.npy')

        deleted_datapoints += 1
        runs_for_all_moments.pop(random_run_index)

        if (deleted_datapoints % 1000 == 0):
            print("Datapoints remaining to delete: ", total_number_datapoints - deleted_datapoints - desired_dataset_size)


def main():
    distribution_balancer()

if __name__ == "__main__":
    main()