import os.path
from os import walk
import pickle
from pathlib import Path

import argparse

import numpy as np
import pandas as pd
import pyulog
from joblib import delayed, Parallel

from utils.file_utils import SensorData
from utils.ulog_extractors import extract_error_ratios, find_attack_timestamp
from utils.evaluation_helpers import ATK_PARAM_COLUMNS

parser = argparse.ArgumentParser(description="Calculate error ratios by test cases")
parser.add_argument('log_dir', metavar='LOG_DIR', action='store',
                    help='path to flight log folder, will recursive collect all logs under this folder, ignore symlink')
parser.add_argument('--export_dir', action='store', default=None,
                    help='path to export the analysis reports')


def clip_landing_data(sensor_error_ratios: SensorData):
    # Inplace remove landing abnormal data in error_ratios
    for k, df_list in sensor_error_ratios.items():
        if not df_list or not isinstance(df_list, (list, tuple)):
            continue
        for idx, v in enumerate(df_list):
            v: pd.DataFrame
            # Discard last LANDING_CLIP_S second data
            clipped_min = v.loc[:, 'timestamp'].min() + TAKEOFF_CLIP_S * SECOND_TO_MILLISECONDS
            clipped_max = v.loc[:, 'timestamp'].max() - LANDING_CLIP_S * SECOND_TO_MILLISECONDS
            df_list[idx] = v.loc[(v['timestamp'] < clipped_max) & (v['timestamp'] >= clipped_min)]


def concat_error_ratios(dataframes, columns):
    ratios = pd.concat([x.loc[:, columns].reset_index(drop=True) for x in dataframes], axis=1, ignore_index=True)
    ratios = ratios.fillna(value=0.0).values
    return ratios


def main(data_folder, export_folder=None):
    if export_folder is None:
        export_folder = os.path.join(data_folder, 'processed_data')

    log_paths = []
    for (abs_path, _, file_names) in walk(data_folder):
        log_paths.extend([os.path.join(abs_path, f) for f in file_names if f.endswith('.ulg')])

    print(f"Found {len(log_paths)} flight logs under {data_folder}")
    print("Convert flight logs to error ratios...")

    def para_func(log_path):
        # Read ULog typically take 3~4 second for a 120M ULog,
        # which take most of the times for processing.
        # Extract is fast: ~0.2s for one ULog
        ulog_obj = pyulog.ULog(log_path)

        attack_triples = find_attack_timestamp(ulog_obj)
        attack_start = np.inf
        atk_apply_type = 0
        if len(attack_triples) > 0:
            attack_start, atk_apply_type, _ = attack_triples[0]

        sub_ratios = extract_error_ratios(ulog_obj)
        meta_info = dict(log_path=log_path,
                         attack_timestamp=attack_start,
                         attack_parameters=dict())

        for k in ATK_PARAM_COLUMNS:
            meta_info['attack_parameters'][k] = ulog_obj.initial_parameters.get(k, 0)
        if atk_apply_type == 0:
            # Clip data before landing if no attack initiated
            clip_landing_data(sub_ratios)
        elif np.isfinite(attack_start):
            # Clip the error ratio data between ATK-CLIP_START_BEFORE_ATK_S to ATK+CLIP_STOP_AFTER_ATK_S
            meta_info['attack_parameters']['ATK_APPLY_TYPE'] = atk_apply_type
            start = attack_start - CLIP_START_BEFORE_ATK_S * SECOND_TO_MILLISECONDS
            stop = attack_start + CLIP_STOP_AFTER_ATK_S * SECOND_TO_MILLISECONDS
            for k, vl in sub_ratios.items():
                if vl and len(vl) > 0:
                    for idx, df in enumerate(vl):
                        # This would lead to empty dataframe
                        full_timestamp = df.loc[:, 'timestamp']
                        mask = (full_timestamp >= start) & (full_timestamp <= stop)
                        vl[idx] = df.loc[mask, :]

        return meta_info, sub_ratios

    # Parallel the disk I/O greatly accelerate the process speed, YaY!
    para_module = Parallel(n_jobs=N_JOBS, verbose=100)
    results = para_module(delayed(para_func)(log_file) for log_file in log_paths)

    if len(results) > 0:
        os.makedirs(export_folder, exist_ok=True)
        with open(os.path.join(export_folder, PKL_FILE_NAME), 'wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    N_JOBS = -1
    TAKEOFF_CLIP_S = 5
    LANDING_CLIP_S = 7
    CLIP_START_BEFORE_ATK_S = 10
    CLIP_STOP_AFTER_ATK_S = 15
    # CLIP_STOP_AFTER_ATK_S = 25
    SECOND_TO_MILLISECONDS = 1e6
    PKL_FILE_NAME = "error_ratios.pkl"

    # Set related solution
    BASELINE_NAMES = [
        # 'control_invariant',
        # 'savior',
        # 'savior_with_buffer',
        # 'software_sensor',
        # 'software_sensor_sup',
        'virtual_imu',
        # 'virtual_imu_no_buffer',
        # 'virtual_imu_cusum',
        # 'virtual_imu_cusum_angvel',
    ]

    # Set related test case
    IDENTIFICATION_TEST_CASE = [
        # 'no_attack_or_detection',
    ]
    VALIDATION_TEST_CASE = [
        # 'no_attack_or_detection'
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

    for BASELINE_NAME in BASELINE_NAMES:
        for test_case in IDENTIFICATION_TEST_CASE:
            log_folder = f'data/evaluation/{BASELINE_NAME}/identification/{test_case}/'
            # Use default export folder
            main(log_folder)

        for test_case in VALIDATION_TEST_CASE:
            log_folder = f'data/evaluation/{BASELINE_NAME}/validation/{test_case}/'
            # Use default export folder
            main(log_folder)

        for test_case in HOVERING_TEST_CASE:
            log_folder = f'data/evaluation/{BASELINE_NAME}/hover_test/{test_case}/'
            # Use default export folder
            main(log_folder)

        for test_case in MOVING_TEST_CASE:
            log_folder = f'data/evaluation/{BASELINE_NAME}/moving_test/{test_case}/'
            # Use default export folder
            main(log_folder)

        for test_case in COMP_MAN_TEST_CASE:
            log_folder = f'data/evaluation/{BASELINE_NAME}/complex_maneuver/{test_case}/'
            # Use default export folder
            main(log_folder)
