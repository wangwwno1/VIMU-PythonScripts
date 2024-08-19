import datetime
import os
from os.path import join

import numpy as np
import pandas as pd

from utils.evaluation_helpers import ATK_PARAM_COLUMNS


def mark_fp_tp(res):
    ms_to_second = 1e-3
    # Only return the first row, assume all rows are belong to the same flight log.
    # Use [0] to return as pd.Dataframe (with row & col) instead of pd.Series
    output_frame = res.iloc[[0], :]

    # Mark the false negative flight, its time to detection and first true positive sensor
    is_attacked = res.loc[:, 'is_attacked']
    is_alarmed = res.loc[:, 'time_to_detection_ms'].notna()
    legacy_fix = False
    if output_frame.iloc[0].at['ATK_APPLY_TYPE'] == 16:
        # Fix an old bug in early logs where the log does not record barometer attack
        legacy_fix = True

    if legacy_fix:
        # Monkey Patch: Assume the barometer attack is immediately identified
        output_frame['false_negative'] = False
        output_frame['tp_ttd_s'] = 0.0
        output_frame['first_tp_sensor'] = res.loc[is_attacked, 'sensor'].iloc[0]
    elif np.all(~is_alarmed[is_attacked]):
        # None of the attacked sensor is being detected, declare false negative
        output_frame['false_negative'] = True
        output_frame['tp_ttd_s'] = np.nan
        output_frame['first_tp_sensor'] = np.nan
    else:
        # Correctly identify one attacked instance, record the first time to detection
        output_frame['false_negative'] = False
        output_frame['tp_ttd_s'] = ms_to_second * np.nanmin(res.loc[is_attacked, 'time_to_detection_ms'])
        valid_tp_mask = is_attacked & is_alarmed
        sensor_index = np.nanargmin(res.loc[valid_tp_mask, 'time_to_detection_ms'])
        output_frame['first_tp_sensor'] = res.loc[valid_tp_mask, 'sensor'].iloc[sensor_index]

    # Mark the false positive flight, and do the same things.
    if np.any(is_alarmed[~is_attacked]):
        # At least one of the normal sensors has false alarm, record the first time to detection
        output_frame['false_positive'] = True
        output_frame['fp_ttd_s'] = ms_to_second * np.nanmin(res.loc[~is_attacked, 'time_to_detection_ms'])
        valid_fp_mask = ~is_attacked & is_alarmed
        sensor_index = np.nanargmin(res.loc[valid_fp_mask, 'time_to_detection_ms'])
        output_frame['first_fp_sensor'] = res.loc[valid_fp_mask, 'sensor'].iloc[sensor_index]
    else:
        # No false alarm, good!
        output_frame['false_positive'] = False
        output_frame['fp_ttd_s'] = np.nan
        output_frame['first_fp_sensor'] = np.nan

    return output_frame


def craft_new_dataframe(detection_report, meta_data):

    detection_report = detection_report.copy()
    detection_report = detection_report.drop(['sample_to_detection', 'false_sample', 'total_sample'],
                                             axis=1, errors='ignore')
    detection_report = calculate_time_to_detection(detection_report, meta_data)
    # Rearrange the column order, put the param columns at last
    prefix_cols = [col for col in detection_report.columns if col not in ATK_PARAM_COLUMNS]
    suffix_cols = [col for col in ATK_PARAM_COLUMNS if col in detection_report.columns]
    detection_report = detection_report.loc[:, prefix_cols + suffix_cols]

    # Handle Mixed boolean string
    if detection_report.loc[:, 'is_attacked'].dtype == 'str':
        col_data = detection_report.loc[:, 'is_attacked']
        detection_report.loc[col_data == 'False', 'is_attacked'] = 0
        detection_report.loc[col_data == 'True', 'is_attacked'] = 1
    if detection_report.loc[:, 'is_attacked'].dtype != bool:
        detection_report.loc[:, 'is_attacked'] = detection_report.loc[:, 'is_attacked'].astype(bool)

    recovery_report = detection_report.copy()
    recovery_report = recovery_report.groupby(['log_name'], group_keys=True).apply(mark_fp_tp).reset_index(drop=True)
    recovery_report.drop(['timestamp_alarm_us', 'time_to_detection_ms', 'sample_to_detection', 'sensor', 'is_attacked'],
                         axis=1, inplace=True, errors='ignore')

    # Extract the recovery duration from meta data
    recovery_report = recovery_report.set_index('log_name', drop=True)
    recovery_report['recovery_duration_s'] = np.nan
    recovery_duration_s = meta_data.loc[:, ['log_name', 'recovery_duration_s']].set_index('log_name', drop=True)

    # Only record the recovery duration that have correctly report the anomaly sensors
    tp_log_names = recovery_report.index[np.isfinite(recovery_report['tp_ttd_s'])]
    overlap_logs = recovery_duration_s.index.intersection(tp_log_names)
    recovery_report.loc[overlap_logs, 'recovery_duration_s'] = recovery_duration_s.loc[overlap_logs,
                                                                                       'recovery_duration_s']
    recovery_report.reset_index(inplace=True)

    # Rearrange the column order, put the param columns at last
    prefix_cols = [col for col in recovery_report.columns if col not in ATK_PARAM_COLUMNS]
    suffix_cols = [col for col in ATK_PARAM_COLUMNS if col in recovery_report.columns]
    recovery_report = recovery_report.loc[:, prefix_cols + suffix_cols]

    return detection_report, recovery_report


def calculate_time_to_detection(detection_report, meta_data):
    attack_ts = meta_data.loc[:, ['log_name', 'timestamp_attack_us']]
    detection_report = detection_report.set_index('log_name').join(attack_ts.set_index('log_name'), on='log_name')
    detection_report.reset_index(inplace=True)

    # Reorder the column
    old_cols = detection_report.columns.tolist()
    col_idx = detection_report.columns.get_loc('timestamp_alarm_us')
    new_columns = old_cols[:col_idx] + ['timestamp_attack_us'] + old_cols[col_idx:-1]
    detection_report = detection_report[new_columns]

    has_alarm = detection_report.loc[:, 'timestamp_alarm_us'] > 0
    detection_report.loc[~has_alarm, 'timestamp_alarm_us'] = np.nan
    alarm_timestamp = detection_report['timestamp_alarm_us']
    attack_timestamp = detection_report['timestamp_attack_us']
    detection_report['time_to_detection_ms'] = 1e-3 * (alarm_timestamp - attack_timestamp)

    return detection_report


if __name__ == '__main__':
    SEC_TO_MS = 1e3
    TTD_THRESHOLD_MS = 1.0 * SEC_TO_MS
    # TTD_THRESHOLD_MS = 20 * SEC_TO_MS
    BASELINE_NAMES = [
        # 'control_invariant',
        'savior',
        # 'savior_with_buffer',
        # 'software_sensor',
        # 'software_sensor_sup',
        'virtual_imu',
        # 'virtual_imu_no_buffer',
        # 'virtual_imu_cusum',
        # 'virtual_imu_t_buf_250',
        # 'virtual_imu_t_buf_500',
        # 'virtual_imu_t_buf_750',
    ]

    HOVER_TEST_CASES = [
        # 'gps_joint_attack',
        # 'gps_overt_attack',
        # 'gps_stealthy_attack_default',
        # 'gyro_overt_attack_default',
        # 'gyro_overt_attack_icm20602',
        # 'gyro_overt_attack_icm20689',
        # 'gyro_stealthy_attack_default',
        # 'gyro_stealthy_attack_wind_mean_0.0',
        # 'gyro_stealthy_attack_wind_mean_1.0',
        # 'gyro_stealthy_attack_wind_mean_2.0',
        # 'recovery_test',
        # 'recovery_test_wind_mean_1.0',
        # 'recovery_test_wind_mean_2.0',
        # 'recovery_test_wind_mean_5.0',
        # 'tmr_test_default',
        # 'tmr_test_icm20602',
        # 'tmr_test_icm20689',
        # 'ttd_test',
        # 't_buffer_test',
    ]

    MOVING_TEST_CASES = [
        # 'gps_joint_attack',
        # 'gps_overt_attack',
        # 'gps_stealthy_attack_default',
        # 'gyro_overt_attack_default',
        # 'gyro_overt_attack_icm20602',
        # 'gyro_overt_attack_icm20689',
        # 'gyro_stealthy_attack_default',
        # 'gyro_stealthy_attack_wind_mean_0.0',
        # 'gyro_stealthy_attack_wind_mean_1.0',
        # 'gyro_stealthy_attack_wind_mean_2.0',
        # 'recovery_test',
        # 'recovery_test_wind_mean_1.0',
        # 'recovery_test_wind_mean_2.0',
        # 'recovery_test_wind_mean_5.0',
        # 'tmr_test_default',
        # 'tmr_test_icm20602',
        # 'tmr_test_icm20689',
        # 'ttd_test',
        # 't_buffer_test',
    ]

    COMP_MAN_TEST_CASE = [
        # 'no_attack_or_detection',
        # 'gps_joint_attack',
        # 'gps_overt_attack',
        # 'gps_stealthy_attack_default',
        'gyro_overt_attack_default',
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

    detection_reports = []
    recovery_reports = []
    for sol_name in BASELINE_NAMES:
        sol_results = []
        test_cases = HOVER_TEST_CASES + MOVING_TEST_CASES + COMP_MAN_TEST_CASE
        hover_test_flag = ["hover_test"] * len(HOVER_TEST_CASES) + ["moving_test"] * len(MOVING_TEST_CASES) + ["complex_maneuver"] * len(COMP_MAN_TEST_CASE)
        for test_case, test_folder in zip(test_cases, hover_test_flag):
            if test_folder == "hover_test":
                scene_flag = "Hovering"
            elif test_folder == "moving_test":
                scene_flag = "Moving"
            elif test_folder == "complex_maneuver":
                scene_flag = "Maneuver"
            else:
                scene_flag = "Unknown"
            log_meta_path = f'data/evaluation/{sol_name}/{test_folder}/{test_case}/log_metas.csv'
            report_path = f'data/evaluation/{sol_name}/{test_folder}/{test_case}/detection_reports.csv'
            sub_detection_report, sub_recovery_report = craft_new_dataframe(
                pd.read_csv(report_path, index_col=False),
                pd.read_csv(log_meta_path, index_col=False)
            )

            for df in (sub_detection_report, sub_recovery_report):
                df.sort_values('log_name', inplace=True, ignore_index=True)
                df.insert(1, "test_case", test_case)
                df.insert(2, "scene_name", scene_flag)

            sol_results.append([sub_detection_report, sub_recovery_report])

        for sub_reports, export_list in zip(list(zip(*sol_results)), [detection_reports, recovery_reports]):
            sub_reports = pd.concat(sub_reports, ignore_index=True)
            sub_reports.insert(1, "Solution", sol_name)
            export_list.append(sub_reports)

    detection_reports = pd.concat(detection_reports, ignore_index=True)
    recovery_reports = pd.concat(recovery_reports, ignore_index=True)
    entry_dir = join('data/evaluation_reports/time_metrics', datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(entry_dir, exist_ok=True)
    detection_reports.to_csv(join(entry_dir, "full_detection_reports.csv"), index=False)
    recovery_reports.to_csv(join(entry_dir, "recovery_reports.csv"), index=False)
