import argparse
import os.path
import pickle
import time
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import ErrorRatioData
from utils.SPCDetectors import CUSUM, EWMA, L1TW, L2TW


def get_batch_alarm_threshold(search_space: dict, file_path: str):
    # Read and batch process the error ratios
    para_module = Parallel(n_jobs=-1, verbose=100)
    with open(file_path, "rb") as f:
        meta_data_pairs = pd.read_pickle(f)
        prep_iter = iter(delayed(preprocess_dataframe)(meta_info, df)
                         for meta_info, df in meta_data_pairs)
        processed_data = para_module(prep_iter)

    # Calculate the threshold per flight, include min threshold to avoid FP and max limit to get TP
    # Ideally the threshold should satisfy
    def task_iterator(class_param_dict: dict, data_pairs):
        for detector_class, class_space in class_param_dict.items():
            for args in product(*list(class_space.values())):
                detector_kwargs = dict(zip(class_space.keys(), args))
                for meta_info, err_ratio, atk_flag in data_pairs:
                    yield meta_info, err_ratio, atk_flag, detector_class, detector_kwargs

    results = para_module(delayed(process_single_flight)(*func_args)
                          for func_args in task_iterator(search_space, processed_data))

    # Sort the result by detector type
    data_by_detector = defaultdict(list)
    for ratio_frame in results:
        if len(ratio_frame.index) == 0:
            continue
        det_type = ratio_frame.loc[0, 'detector_type']
        data_by_detector[det_type].append(ratio_frame)

    result_dict = {}
    for k, vl in data_by_detector.items():
        result_dict[k] = pd.concat(vl, axis=0, ignore_index=True)

    return result_dict


def preprocess_dataframe(meta_info, single_flight_ratios):
    result_ratios = defaultdict(list)
    result_attack_flag = defaultdict(list)
    attack_start = meta_info.get('attack_timestamp')
    for k, vl in single_flight_ratios.items():
        if not vl:
            continue
        for df in vl:
            if attack_start is not None:
                stop = attack_start + CLIP_STOP_AFTER_ATK_S * SECOND_TO_MILLISECONDS
                full_timestamp = df['timestamp'].values
                df = df.loc[full_timestamp <= stop, :]

            arr = df.loc[:, [col for col in df.columns if col.endswith('error_ratio')]].values
            is_attacked = None
            if 'is_attacked' in df:
                is_attacked = df.loc[:, 'is_attacked'].values

            result_ratios[k].append(arr)
            result_attack_flag[k].append(is_attacked)

    return meta_info, ErrorRatioData(**result_ratios), ErrorRatioData(**result_attack_flag)


def process_single_flight(meta_info: dict,
			              error_ratio_data: ErrorRatioData, attack_flag_data: ErrorRatioData,
                          detector_class, detector_kwargs):
    # Extract limits (per flight, per sensor type) from 1 flight record
    results = []
    for state_name, vl in error_ratio_data.items():
        if vl:
            for error_ratio_array, is_attacked in zip(vl, attack_flag_data[state_name]):
                # Refresh detector status by re-initialize new one
                detector = detector_class(**detector_kwargs)
                min_lim, max_lim = get_single_alarm_threshold(detector, error_ratio_array, is_attacked)
                if np.isfinite(min_lim) or np.isfinite(max_lim):
                    results.append([state_name, min_lim, max_lim])
    results = pd.DataFrame(results, columns=['sensor_type', 'tn_min_limit', 'tp_max_limit'])
    results = results.assign(detector_type=detector_class.__name__, log_path=meta_info['log_path'], **detector_kwargs)
    new_col_order = ['sensor_type', 'detector_type'] + list(detector_kwargs.keys()) + ['tn_min_limit', 'tp_max_limit', 'log_path']
    results = results.loc[:, new_col_order]

    return results


def get_single_alarm_threshold(detector, input_array: np.ndarray, is_attacked: np.ndarray = None):
    """Calculate the alarm threshold for single flight.
    Ideally the detector threshold should satisfy tn_min_limit < threshold <= tp_max_limit.

    :param detector: initialized detector object
    :param input_array: error ratio of a sensor instance
    :param is_attacked: flags to mark the attack status per timestamp, by default its None (no attack).
    :return: The minimum threshold to avoid false positive (tn_min_limit)
            and the maximum to avoid false negative (tp_max_limit)
    """
    tn_min_limit = np.nan
    tp_max_limit = np.nan
    for idx in range(len(input_array)):
        detector.validate(input_array[idx])
        test_values = detector.test_values
        limit = np.nanmax(test_values)
        if is_attacked is None or ~is_attacked[idx]:
            # Find the largest test value as the lower bound to achieve no false alarm
            tn_min_limit = limit if np.isnan(tn_min_limit) else max(limit, tn_min_limit)
        else:
            # Find the largest test value as the upper bound to avoid false negative (attack missed)
            tp_max_limit = limit if np.isnan(tp_max_limit) else max(limit, tp_max_limit)
    return tn_min_limit, tp_max_limit


def main(input_path, export_path):
    if not os.path.isfile(input_path):
        return

    detector_params = get_batch_alarm_threshold(SEARCH_SPACE, input_path)
    if len(detector_params) > 0:
        result_frames = []
        os.makedirs(export_path, exist_ok=True)
        for detector_type, output in detector_params.items():
            result_frames.append(output)
            output.to_csv(f"{export_path}/{detector_type}_threshold_per_flight.csv", index=False)


DETECTOR_PARAMS = [
    (
        'savior',
        {
            CUSUM: {
                'control_limit': [np.inf],
                'mean_shift': [0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
            },
        }
    ),
    (
        'software_sensor_sup',
        {
            L1TW: {
                'control_limit': [np.inf],
                'time_window': list(range(10, 101, 10)) + [120, 160, 200]
            },
        }
    ),
    (
        'virtual_imu_cusum',
        {
            CUSUM: {
                'control_limit': [np.inf],
                'mean_shift': [0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
            },
        }
    ),
    (
        'virtual_imu',
        {
            CUSUM: {
                'control_limit': [np.inf],
                # 'mean_shift': [1.0]
                'mean_shift': [0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
            },
            EWMA: {
                'control_limit': [0.01],
                'alpha': [0.01, 0.02, 0.05],
                'cap': [0.52, 0.85, 1.1, 1.3, np.inf]

            }
        }
    ),
    (
        'control_invariant',
        {
            L2TW: {
                'control_limit': [np.inf],
                'time_window': list(range(10, 101, 10)) + [120, 160, 200]
            }
        }
    ),
    (
        'virtual_imu_cusum_angvel',
        {
            CUSUM: {
                'control_limit': [np.inf],
                'mean_shift': [0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
            },
        }
    ),
    (
        'virtual_imu',
        {
            CUSUM: {
                'control_limit': [np.inf],
                # 'mean_shift': [1.0]
                'mean_shift': [0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
            },
            EWMA: {
                'control_limit': [0.01],
                'alpha': [0.01, 0.02, 0.05],
                'cap': [0.52, 0.85, 1.1, 1.3, np.inf]
            }
        }
    )
]

if __name__ == '__main__':
    CLIP_STOP_AFTER_ATK_S = 1
    # CLIP_STOP_AFTER_ATK_S = 10
    SECOND_TO_MILLISECONDS = 1e6

    SELECTED_BASELINES = ["virtual_imu"]
    SELECTED_DETECTOR_PARAMS = [(name, params) for name, params in DETECTOR_PARAMS if name in SELECTED_BASELINES]

    # Set related test case
    IDENTIFICATION_TEST_CASE = [
        # 'no_attack_or_detection',
    ]
    VALIDATION_TEST_CASE = [
        # 'no_attack_or_detection',
    ]
    HOVERING_TEST_CASE = [
        # 'gps_joint_attack',
        'gps_overt_attack',
        # 'gps_stealthy_attack_default',
        # 'gyro_overt_attack_default',
        # 'gyro_overt_attack_icm20602',
        # 'gyro_overt_attack_icm20689',
        # 'gyro_stealthy_attack_default',
        # 'recovery_test',
        # 'tmr_test_default',
        # 'tmr_test_icm20602',
        # 'tmr_test_icm20689',
        # 'ttd_test',
    ]

    MOVING_TEST_CASE = [
        # 'gps_joint_attack',
        'gps_overt_attack',
        # 'gps_stealthy_attack_default',
        # 'gyro_overt_attack_default',
        # 'gyro_overt_attack_icm20602',
        # 'gyro_overt_attack_icm20689',
        # 'gyro_stealthy_attack_default',
        # 'recovery_test',
        # 'tmr_test_default',
        # 'tmr_test_icm20602',
        # 'tmr_test_icm20689',
        # 'ttd_test',
    ]

    COMP_MAN_TEST_CASE = [
        'no_attack_or_detection',
        # 'gps_joint_attack',
        # 'gps_overt_attack',
        # 'gps_stealthy_attack_default',
        # 'gyro_overt_attack_default',
        # 'gyro_overt_attack_icm20602',
        # 'gyro_overt_attack_icm20689',
        # 'gyro_stealthy_attack_default',
        # 'recovery_test',
        # 'tmr_test_default',
        # 'tmr_test_icm20602',
        # 'tmr_test_icm20689',
        # 'ttd_test',
    ]

    PKL_FILE_NAME = "error_ratios.pkl"
    for BASELINE_NAME, SEARCH_SPACE in SELECTED_DETECTOR_PARAMS:
        for test_case in IDENTIFICATION_TEST_CASE:
            pkl_folder = f'data/evaluation/{BASELINE_NAME}/identification/{test_case}/processed_data/'
            export_folder = os.path.join(pkl_folder, 'param_selection')
            main(os.path.join(pkl_folder, PKL_FILE_NAME), export_folder)

        for test_case in VALIDATION_TEST_CASE:
            pkl_folder = f'data/evaluation/{BASELINE_NAME}/validation/{test_case}/processed_data/'
            export_folder = os.path.join(pkl_folder, 'param_selection')
            main(os.path.join(pkl_folder, PKL_FILE_NAME), export_folder)

        for test_case in HOVERING_TEST_CASE:
            pkl_folder = f'data/evaluation/{BASELINE_NAME}/hover_test/{test_case}/processed_data/'
            export_folder = os.path.join(pkl_folder, 'param_selection')
            main(os.path.join(pkl_folder, PKL_FILE_NAME), export_folder)

        for test_case in MOVING_TEST_CASE:
            pkl_folder = f'data/evaluation/{BASELINE_NAME}/moving_test/{test_case}/processed_data/'
            export_folder = os.path.join(pkl_folder, 'param_selection')
            main(os.path.join(pkl_folder, PKL_FILE_NAME), export_folder)

        for test_case in COMP_MAN_TEST_CASE:
            pkl_folder = f'data/evaluation/{BASELINE_NAME}/complex_maneuver/{test_case}/processed_data/'
            export_folder = os.path.join(pkl_folder, 'param_selection')
            main(os.path.join(pkl_folder, PKL_FILE_NAME), export_folder)
