import datetime
import os

from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd


def calculate_roc_curve(normal_thresholds: np.ndarray, attack_thresholds: np.ndarray = None):
    detection_statistic = normal_thresholds
    detection_label = np.zeros_like(detection_statistic, dtype=int)

    if attack_thresholds is not None:
        attack_label = np.ones_like(attack_thresholds, dtype=int)
        detection_statistic = np.concatenate([detection_statistic, attack_thresholds])
        detection_label = np.concatenate([detection_label, attack_label])

    fpr, tpr, thresholds = roc_curve(detection_label, detection_statistic, pos_label=1, drop_intermediate=False)
    result_frame = pd.DataFrame(
        np.stack([fpr, tpr, thresholds], axis=1),
        columns=['FPR', 'TPR', 'Threshold']
    )

    return result_frame


def wrap_detector_params(roc_frame: pd.DataFrame, param_dict: dict):
    output = roc_frame.assign(**param_dict)
    output = output.loc[:, list(param_dict.keys()) + ['FPR', 'TPR', 'Threshold']]
    return output


def get_roc_curve(normal_threshold_path, attack_threshold_path=None):
    if not os.path.isfile(normal_threshold_path):
        return

    if attack_threshold_path is not None and not os.path.isfile(attack_threshold_path):
        return

    normal_threshold = pd.read_csv(normal_threshold_path)
    if 'log_path' in normal_threshold.columns:
        normal_threshold.drop(columns=['log_path'], inplace=True)  # fixme monkey patch
    group_cols = list(normal_threshold.drop(columns=['tn_min_limit', 'tp_max_limit']).columns)
    norm_groups = normal_threshold.drop(columns=['tp_max_limit']).groupby(group_cols)
    if attack_threshold_path is None:
        # calculate FPR Only
        task_iter = iter((k, norm_groups.get_group(k), None) for k in norm_groups.groups.keys())
    else:
        # Calculate the full ROC curve
        attack_threshold = pd.read_csv(attack_threshold_path)
        if 'log_path' in attack_threshold.columns:
            attack_threshold.drop(columns=['log_path'], inplace=True)  # fixme monkey patch
        # Change the reference col to attack_group
        group_cols = list(attack_threshold.drop(columns=['tn_min_limit', 'tp_max_limit']).columns)
        attk_groups = attack_threshold.drop(columns=['tn_min_limit']).groupby(group_cols)
        # Take the intersection of keys
        task_iter = iter((k, norm_groups.get_group(k), attk_groups.get_group(k)) for k in attk_groups.groups.keys()
                         if k in norm_groups.groups and k in attk_groups.groups)

    roc_results = []
    for k, normal_frame, attack_frame in task_iter:
        normal_val = normal_frame.loc[:, 'tn_min_limit'].dropna().values
        attack_val = None
        if isinstance(attack_frame, pd.DataFrame):
            attack_val = attack_frame.loc[:, 'tp_max_limit'].dropna().values
            if len(attack_val) == 0:
                # No valid attack, skip
                continue
        sub_frame = calculate_roc_curve(normal_val, attack_val)
        sub_frame = wrap_detector_params(sub_frame, dict(zip(group_cols, k)))
        roc_results.append(sub_frame)

    output_frame = None
    if len(roc_results) > 0:
        output_frame = pd.concat(roc_results, axis=0, ignore_index=True)

    return output_frame


if __name__ == '__main__':
    DETECTOR_TYPES = [
        'CUSUM',
        'EWMA',
        'L1TW',
        'L2TW'
    ]
    EXPORT_PATH = f'data/evaluation_reports/roc_curves/{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}'

    # Set related solution & normal threshold path
    LINEAR_MODEL_BASELINE = [
        # 'control_invariant',
        # 'software_sensor_sup'
    ]

    NON_LINEAR_MODEL_BASELINE = [
        # 'savior',
        'virtual_imu',
        # 'virtual_imu_no_buffer',
        # 'virtual_imu_cusum',
    ]

    # Set related test case
    VALIDATION_TEST_CASE = [
        'no_attack_or_detection',
        # 'no_attack_or_detection_wind_north_mean_1.0',
        # 'no_attack_or_detection_wind_north_mean_2.0',
        # 'no_attack_or_detection_wind_north_mean_5.0'
    ]
    HOVERING_TEST_CASE = [
        # 'gps_joint_attack',
        # 'gps_overt_attack',
        # 'gps_stealthy_attack_default',
        # 'gyro_overt_attack_default',
        # 'gyro_overt_attack_icm20602',
        # 'gyro_overt_attack_icm20689',
        # 'gyro_stealthy_attack_default',
        # 'gyro_stealthy_attack_wind_mean_0.0',
        # 'gyro_stealthy_attack_wind_mean_1.0',
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
        # 'gyro_stealthy_attack_wind_mean_0.0',
        # 'gyro_stealthy_attack_wind_mean_1.0',
        # 'recovery_test',
        # 'tmr_test_default',
        # 'tmr_test_icm20602',
        # 'tmr_test_icm20689',
        # 'ttd_test',
    ]

    COMP_MAN_TEST_CASE = [
        # 'no_attack_or_detection',
        # 'gps_joint_attack',
        # 'gps_overt_attack',
        # 'gps_stealthy_attack_default',
        # 'gyro_overt_attack_default',
        # 'gyro_overt_attack_icm20602',
        # 'gyro_overt_attack_icm20689',
        # 'gyro_stealthy_attack_default',
        # 'recovery_test',
        # 'recovery_test_wind',
        # 'tmr_test_default',
        # 'tmr_test_icm20602',
        # 'tmr_test_icm20689',
        # 'ttd_test',
        # 't_buffer_test'
    ]

    for detector_type in DETECTOR_TYPES:
        data_file_name = f'{detector_type}_threshold_per_flight.csv'
        sub_frames = []
        for baseline_name in LINEAR_MODEL_BASELINE + NON_LINEAR_MODEL_BASELINE:
            baseline_frames = []
            test_cases = VALIDATION_TEST_CASE + HOVERING_TEST_CASE + MOVING_TEST_CASE + COMP_MAN_TEST_CASE
            scene_names = ['NoAttack'] * len(VALIDATION_TEST_CASE) +\
                          ['Hovering'] * len(HOVERING_TEST_CASE) +\
                          ['Moving'] * len(MOVING_TEST_CASE) +\
                          ['Maneuver'] * len(COMP_MAN_TEST_CASE)

            NORMAL_DATA_FOLDER = f"data/evaluation/{baseline_name}/validation/no_attack_or_detection/processed_data/param_selection"
            normal_file_path = os.path.join(NORMAL_DATA_FOLDER, data_file_name)
            attack_file_path = None

            for test_case, scene_name in zip(test_cases, scene_names):
                wind_disuturbance_case = test_case in ['gyro_stealthy_attack_wind_mean_0.0', 'gyro_stealthy_attack_wind_mean_1.0']
                if wind_disuturbance_case and baseline_name != 'virtual_imu':
                    continue
                elif test_case == 'gyro_stealthy_attack_wind_mean_1.0' and baseline_name == 'virtual_imu':
                    NORMAL_DATA_FOLDER = f"data/evaluation/virtual_imu/validation/no_attack_or_detection_wind_north_mean_1.0" \
                    f"/processed_data/param_selection"
                    normal_file_path = os.path.join(NORMAL_DATA_FOLDER, data_file_name)

                if scene_name == 'Hovering':
                    attack_file_path = f"data/evaluation/{baseline_name}/hover_test/"\
                                       f"{test_case}/processed_data/param_selection/{data_file_name}"
                    scene_name = 'Hovering'
                elif scene_name == 'Moving':
                    attack_file_path = f"data/evaluation/{baseline_name}/moving_test/"\
                                       f"{test_case}/processed_data/param_selection/{data_file_name}"
                    scene_name = 'Moving'
                elif scene_name == 'NoAttack':
                    NORMAL_DATA_FOLDER = f"data/evaluation/{baseline_name}/validation/{test_case}/processed_data/param_selection"
                    normal_file_path = os.path.join(NORMAL_DATA_FOLDER, data_file_name)
                elif scene_name == 'Maneuver':
                    NORMAL_DATA_FOLDER = f"data/evaluation/{baseline_name}/complex_maneuver/no_attack_or_detection/processed_data/param_selection"
                    normal_file_path = os.path.join(NORMAL_DATA_FOLDER, data_file_name)
                    if test_case != "no_attack_or_detection":
                        attack_file_path = f"data/evaluation/{baseline_name}/complex_maneuver/"\
                                           f"{test_case}/processed_data/param_selection/{data_file_name}"

                # if attack_file_path is not None and baseline_name == 'virtual_imu_cusum_angvel':
                #     # Use Attack File from virtual_imu
                #     scene_folder = 'hover_test' if scene_name == 'Hovering' else 'moving_test'
                #     attack_file_path = f"data/evaluation/virtual_imu/{scene_folder}/"\
                #                        f"{test_case}/processed_data/param_selection/{data_file_name}"

                result = get_roc_curve(normal_file_path, attack_file_path)
                if result is not None:
                    result.insert(0, "test_case", test_case)
                    result.insert(1, "scene_name", scene_name)
                    baseline_frames.append(result)

            if len(baseline_frames) > 0:
                baseline_result = pd.concat(baseline_frames, ignore_index=True)
                baseline_result.insert(0, 'baseline_name', baseline_name)
                sub_frames.append(baseline_result)

        if len(sub_frames) > 0:
            os.makedirs(EXPORT_PATH, exist_ok=True)
            detector_result = pd.concat(sub_frames, ignore_index=True)
            export_path = os.path.join(EXPORT_PATH, f'{detector_type}_roc_curve_threshold.csv')
            detector_result.to_csv(export_path, index=False)
