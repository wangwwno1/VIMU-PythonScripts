import os.path

import pandas as pd
import numpy as np
import pyulog

from .file_utils import SensorData
from .mav_utils import AttackType
from .helper_functions import extract_ulog_dict
from .ulog_extractors import extract_recovery_status, extract_test_ratios, find_attack_timestamp

DETECTION_RAW_METRICS = ['timestamp_alarm_us', 'sample_to_detection', 'false_sample', 'total_sample']
DETECTION_METRICS = ['timestamp_alarm_us', 'sample_to_detection', 'time_to_detection_ms', 'false_sample',
                     'total_sample']
GENERAL_INFO_COLUMNS = ['log_name', 'attack_not_found', 'false_alarm_before_attack', 'detect_timeout_ms',
                        'timestamp_attack_us', 'recovery_start_us', 'recovery_end_us', 'recovery_duration_s']
ATK_PARAM_COLUMNS = ['ATK_APPLY_TYPE', 'ATK_STEALTH_TYPE', 'ATK_COUNTDOWN_MS', 'ATK_GPS_P_IV',
                     'ATK_MULTI_IMU', 'ATK_GYR_AMP', 'ATK_GYR_FREQ', 'ATK_GYR_PHASE', 'ATK_ACC_BIAS', 'IV_DELAY_MASK', 'IV_TTD_DELAY_MS']


def evaluate_performance(ulog_path, deviation_threshold: float = 3.0, detect_timeout_ms: int = None):
    ulog_obj = pyulog.ULog(ulog_path)
    log_name = os.path.split(ulog_path)[-1]
    log_info = pd.DataFrame([[log_name, False, False, detect_timeout_ms,
                              0, 0, 0, 0.] + [0] * len(ATK_PARAM_COLUMNS)],
                            columns=GENERAL_INFO_COLUMNS + ATK_PARAM_COLUMNS)
    attack_timestamps = find_attack_timestamp(ulog_obj)
    param_atk_multi_imu = ulog_obj.initial_parameters['ATK_MULTI_IMU']
    for k in ATK_PARAM_COLUMNS:
        log_info[k] = ulog_obj.initial_parameters[k] if k in ulog_obj.initial_parameters else 0
    downcast_params = ['ATK_STEALTH_TYPE', 'ATK_MULTI_IMU', 'IV_DELAY_MASK', 'IV_TTD_DELAY_MS']
    log_info.loc[:, downcast_params] = log_info.loc[:, downcast_params].astype(int, errors='ignore')

    if len(attack_timestamps) == 0:
        log_info['attack_not_found'] = True
        return [log_info, None]

    ulog_dict = extract_ulog_dict(ulog_obj)

    # Only examine the first attack
    attack_timestamp, attack_flags, end_timestamp = attack_timestamps[0]
    log_info['timestamp_attack_us'] = attack_timestamp
    log_info['ATK_APPLY_TYPE'] = int(attack_flags)

    test_ratios = extract_test_ratios(ulog_dict)
    false_alarm = check_test_ratios(test_ratios, attack_timestamp)
    log_info['false_alarm_before_attack'] = false_alarm
    # if false_alarm:
    #     return [log_info, None]

    first_alarm_timestamp = np.inf
    detection_results = []
    has_blocked_attack = attack_flags & (AttackType.Barometer | AttackType.Magnetometer)
    for sens_type, df_list in test_ratios.items():
        sensor_results = []
        sensor_test_ratio = df_list[0]
        sensor_count = 0
        for k in sensor_test_ratio.columns:
            if 'test_ratio' in k:
                expect_attack = 0
                if sens_type == 'gps':
                    expect_attack = AttackType.GpsPosition | AttackType.GpsVelocity | AttackType.GpsJointPVAttack
                elif sens_type == 'barometer':
                    expect_attack = AttackType.Barometer
                elif sens_type == 'magnetometer':
                    expect_attack = AttackType.Magnetometer
                elif sens_type == 'accelerometer' and (param_atk_multi_imu & (1 << sensor_count)):
                    expect_attack = AttackType.Accelerometer
                elif sens_type == 'gyroscope' and (param_atk_multi_imu & (1 << sensor_count)):
                    expect_attack = AttackType.Gyroscope
                is_attacked = bool(expect_attack & attack_flags)

                # Evaluate
                sens_k_tr = sensor_test_ratio[k]
                mask = (sens_k_tr.index >= attack_timestamp)  # Greater than attack timestamp
                if end_timestamp is not None:
                    mask &= sens_k_tr.index <= end_timestamp
                detect_timeout = attack_timestamp + detect_timeout_ms * 1e3 if detect_timeout_ms else None
                if has_blocked_attack and (sens_type == 'barometer' or sens_type == 'magnetometer'):
                    # block attack will cut the test ratio record.
                    alarm_timestamp = attack_timestamp
                    result = pd.Series([alarm_timestamp, 0, 0, np.nan, 1], index=DETECTION_METRICS)
                else:
                    assert len(sens_k_tr[mask] > 0), RuntimeError(f"Unexpected zero test_ratio sequences in {sens_type}: {ulog_path}")
                    result = evaluate_detection_performance(sens_k_tr[mask], is_attacked, detect_timeout)
                    result['time_to_detection_ms'] = np.nan
                    if result.loc['timestamp_alarm_us'] > 0:
                        result['time_to_detection_ms'] = (result.loc['timestamp_alarm_us'] - attack_timestamp) / 1.e3
                        if result.loc['timestamp_alarm_us'] < first_alarm_timestamp:
                            first_alarm_timestamp = result.loc['timestamp_alarm_us']

                result['is_attacked'] = is_attacked
                result['sensor'] = sens_type
                result = result.loc[['sensor', 'is_attacked'] + DETECTION_METRICS]

                sensor_results.append(result)
                sensor_count += 1
        if len(sensor_results) > 0:
            detection_results.append(pd.concat(sensor_results, axis=1, ignore_index=True))

    if has_blocked_attack and not np.isfinite(first_alarm_timestamp):
        first_alarm_timestamp = attack_timestamp

    detection_results = pd.concat(detection_results, axis=1, ignore_index=True).T
    old_columns = detection_results.columns.to_list()
    detection_results.loc[:, 'log_name'] = log_name
    detection_results.loc[:, ATK_PARAM_COLUMNS] = log_info.loc[:, ATK_PARAM_COLUMNS]
    detection_results.loc[:, ATK_PARAM_COLUMNS] = detection_results.loc[:, ATK_PARAM_COLUMNS].fillna(method='ffill',
                                                                                                     axis=0)
    detection_results = detection_results.loc[:, ['log_name'] + ATK_PARAM_COLUMNS + old_columns]

    downcast_params = ['timestamp_alarm_us', 'sample_to_detection', 'time_to_detection_ms',
                        'false_sample', 'total_sample']
    detection_results.loc[:, downcast_params] = detection_results.loc[:, downcast_params].astype(int, errors='ignore')

    if np.isfinite(first_alarm_timestamp):
        (est_state, truth_state) = extract_recovery_status(ulog_dict)
        est_state, truth_state = est_state.position, truth_state.position
        mask = (est_state.index >= attack_timestamp) & ((end_timestamp is None) | (est_state.index < end_timestamp))
        sub_index = est_state.index[mask]
        deviated_timestamp = get_first_deviated_timestamp(est_state.loc[sub_index],
                                                          truth_state.loc[sub_index],
                                                          deviation_threshold)

        log_info['recovery_start_us'] = first_alarm_timestamp
        log_info['recovery_end_us'] = deviated_timestamp
        if deviated_timestamp >= first_alarm_timestamp:
            log_info['recovery_duration_s'] = (deviated_timestamp - first_alarm_timestamp) / 1.e6
        else:
            log_info['recovery_duration_s'] = 0

    return [log_info, detection_results]


def merge_full_test_ratios(single_log_output: SensorData):
    full_test_ratios = None
    for k, df in single_log_output.items():
        sub_frame = df[0].copy()
        sub_frame.rename(columns={sk: f"{k}_{sk}" for sk in sub_frame.columns if sk.startswith('test_ratio')},
                         inplace=True)
        if full_test_ratios is None:
            full_test_ratios = sub_frame
        else:
            full_test_ratios = pd.merge(full_test_ratios, sub_frame, how='outer', left_index=True, right_index=True)
    return full_test_ratios


def check_test_ratios(single_log_test_ratios: SensorData, attack_timestamp: int):
    false_alarm_before_attack = False
    for df_list in single_log_test_ratios.values():
        sub_frame = df_list[0]
        test_ratio_array = None
        for k in sub_frame.keys():
            if 'test_ratio' in k:
                if test_ratio_array is None:
                    test_ratio_array = sub_frame.loc[:, k].values
                else:
                    test_ratio_array = np.maximum(test_ratio_array, sub_frame.loc[:, k].values)

        if np.any(test_ratio_array[sub_frame.index < attack_timestamp] >= 1.0):
            false_alarm_before_attack = True
            break

    return false_alarm_before_attack


def evaluate_detection_performance(test_ratio: pd.DataFrame, is_attacked: bool, detect_timeout: int = None):
    has_alarm = test_ratio >= 1.0
    first_alarm_idx = find_first(has_alarm.values)

    if detect_timeout is not None:
        has_timeout = test_ratio.index >= detect_timeout
        timeout_index = find_first(has_timeout)
        has_alarm = has_alarm.loc[has_alarm.index <= has_alarm.index[timeout_index]]

    sample_counts = len(has_alarm)
    false_counts = np.sum(has_alarm != is_attacked)

    if first_alarm_idx >= 0:
        # We catch an alarm
        alarm_timestamp = test_ratio.index[first_alarm_idx]
        sample_to_detection = first_alarm_idx + 1
    else:
        alarm_timestamp = 0
        sample_to_detection = 0
    return pd.Series([alarm_timestamp, sample_to_detection, false_counts, sample_counts],
                     index=DETECTION_RAW_METRICS)


def get_first_deviated_timestamp(estimate_state: pd.DataFrame, groundtruth_state: pd.DataFrame, deviation_threshold):
    # Return first timestamp that deviation exceed threshold, use L2 Norm
    # Assume all values has been aligned
    has_deviated = pd.Series(
        np.linalg.norm(estimate_state.values - groundtruth_state.values, axis=1) >= deviation_threshold,
        index=estimate_state.index)
    first_index = find_first(has_deviated.values)
    return has_deviated.index[first_index]


def find_first(x):
    # Solution from this dude tstanisl: https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast
    # Can only find True, but quite enough for me
    idx = x.view(bool).argmax()
    return idx if x[idx] else -1
