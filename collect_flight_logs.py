import datetime
import os
import shlex
import shutil
import sys
from os.path import join, expanduser
from typing import Optional

from utils.file_utils import TestCaseConfig

import pexpect


def generate_config(
        baseline_name: str,
        test_case_name: str,
        attack_case_name: Optional[str] = None
):

    sol_param_folder = shlex.quote(f'data/parameter_yaml/solution/{baseline_name}')
    if test_case_name in ['identification', 'validation']:
        if attack_case_name is not None and attack_case_name != 'no_attack_or_detection':
            RuntimeWarning("Attack case setting in 'identification' and 'validation' "
                           "will override by \"no_attack_or_detection\".")
        attack_case_name = 'no_attack_or_detection'
        detector_param_path = attack_param_dir = attack_waypoint = max_deviation = None
        mission_file = 'data/missions/Survey for Identification and Evaluation.plan'
        if test_case_name == 'identification':
            topic_file_path = 'data/logger_topics/logger_topics - System Identification.txt'
        else:
            topic_file_path = 'data/logger_topics/logger_topics - Detector Identification.txt'
    elif test_case_name in ['hover_test', 'moving_test', 'complex_maneuver']:
        assert attack_case_name is not None
        max_deviation = 5.0
        if test_case_name == 'hover_test':
            mission_file = 'data/missions/Savior Test Mission for GPS and GYRO Attack.plan'
            attack_waypoint = 3
        elif test_case_name == 'moving_test':
            mission_file = 'data/missions/VIMU Test Mission for Moving Scenario.plan'
            attack_waypoint = 1
        else:
            mission_file = 'data/missions/Complex Maneuvers.plan'
            attack_waypoint = 3

        if attack_case_name == 'no_attack_or_detection':
            topic_file_path = 'data/logger_topics/logger_topics - Detector Identification.txt'
            detector_param_path = attack_param_dir = attack_waypoint = max_deviation = None
        else:
            topic_file_path = 'data/logger_topics/logger_topics - Evaluation.txt'
            detector_param_path = join(sol_param_folder, 'detector_params.yaml')
            attack_param_dir = shlex.quote(f'data/parameter_yaml/attack/{attack_case_name}')

    elif test_case_name in ["fpr_validation"]:
        if attack_case_name is not None and attack_case_name != 'false_positive_test':
            RuntimeWarning("Attack case setting in 'identification' and 'validation' "
                           "will override by \"false_positive_test\".")
        attack_case_name = 'false_positive_test'
        topic_file_path = 'data/logger_topics/logger_topics - Evaluation.txt'
        detector_param_path = join(sol_param_folder, 'detector_params.yaml')
        attack_param_dir = attack_waypoint = max_deviation = None
        mission_file = 'data/missions/Survey for Identification and Evaluation.plan'
    else:
        raise NotImplementedError

    config = TestCaseConfig(
        firmware_path=BASELINE_FIRMWARE_PATH[baseline_name],
        mission_file_path=shlex.quote(mission_file),
        topic_file_path=shlex.quote(topic_file_path),
        log_export_path=shlex.quote(f'data/evaluation/{baseline_name}/{test_case_name}/{attack_case_name}/'),
        init_param_path=join(sol_param_folder, 'default_params.yaml'),
        detector_param_path=detector_param_path,
        attack_param_dir=attack_param_dir,
        attack_waypoint=attack_waypoint,
        max_deviation=max_deviation
    )

    return config


def run(config: TestCaseConfig, num_trials: int = 5, attack_timeout: int = 90):
    mavros_cmd = ["python3 mavros_main.py",
                  f"{config.firmware_path}",  # PX4_PATH
                  f"{config.mission_file_path}",  # Mission files
                  f"--logger_topics {config.topic_file_path}",
                  f"--init_param {config.init_param_path}",
                  f"-s 10 -n {num_trials}",
                  f"--attack_timeout {attack_timeout}"]

    if config.attack_param_dir:
        attack_param_files = get_param_files(config.attack_param_dir)
    else:
        attack_param_files = [None]  # Place holder

    for attack_param in attack_param_files:
        entry_dir = join(config.log_export_path, datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
        entry_log_dir = join(entry_dir, 'logs')
        os.makedirs(entry_log_dir, exist_ok=True)

        entry_config = [f"--output_dir {entry_log_dir}"]
        shutil.copy(config.init_param_path, join(entry_dir, 'init_params.yaml'))
        if config.detector_param_path is not None:
            entry_config.append(f"--detector_param {config.detector_param_path}", )
            shutil.copy(config.detector_param_path, join(entry_dir, 'detector_params.yaml'))
        if attack_param is not None and config.attack_waypoint is not None:
            entry_config.extend([
                f"--attack_param {attack_param}",
                f"--attack_wp {config.attack_waypoint}"
            ])
            shutil.copy(attack_param, join(entry_dir, 'attack_params.yaml'))

        if config.max_deviation is not None:
            entry_config.append(f"--max_deviation {config.max_deviation}")

        final_cmd = " ".join(mavros_cmd + entry_config)
        child = pexpect.spawn(final_cmd, encoding='utf-8', timeout=900 * NUM_TRIAL)
        child.logfile = sys.stdout
        child.expect([pexpect.EOF, pexpect.TIMEOUT])
        child.close()


def get_param_files(param_folder):
    param_files = []
    for (abs_path, _, file_names) in os.walk(param_folder):
        param_files.extend([os.path.join(abs_path, f) for f in file_names if f.endswith('.yaml')])
    return param_files


if __name__ == "__main__":
    # Modify the path to firmware
    BASELINE_FIRMWARE_PATH = {
        # 'control_invariant': join(expanduser('~'), 'Path/To/Project-VIMU/baseline/Control-Invariant'),
        # 'savior': join(expanduser('~'), 'Path/To/Project-VIMU/baseline/SAVIOR'),
        # 'savior_with_buffer': join(expanduser('~'), 'Path/To/Project-VIMU/baseline/SAVIOR/'),
        # 'software_sensor_sup': join(expanduser('~'), 'Path/To/Project-VIMU/baseline/SoftwareSensor-SupCompensation/'),
        'virtual_imu': join(expanduser('~'), 'Path/To/Project-VIMU/baseline/Virtual-IMU/'),
        # 'virtual_imu_cusum': join(expanduser('~'), 'Path/To/Project-VIMU/baseline/Virtual-IMU+CUSUM-Detector'),
    }

    NUM_NORMAL_TRIAL = 10
    NUM_ATTACK_TRIAL = 10

    TEST_CASES = [
        # 'identification',
        'validation',
        # 'fpr_validation',
        # 'hover_test',
        # 'moving_test'
        # 'complex_maneuver'
    ]

    HOVERING_TEST_CASE = [
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
        # 'ttd_test'
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
        # 'recovery_test_wind',
        # 'tmr_test_default',
        # 'tmr_test_icm20602',
        # 'tmr_test_icm20689',
        # 'ttd_test',
        # 't_buffer_test'
    ]

    param_file_counts = int('identification' in TEST_CASES) + int('validation' in TEST_CASES) + int('fpr_validation' in TEST_CASES)
    if 'hover_test' in TEST_CASES:
        for test_case in HOVERING_TEST_CASE:
            result = get_param_files(f'data/parameter_yaml/attack/{test_case}')
            param_file_counts += len(result)
    if 'moving_test' in TEST_CASES:
        for test_case in MOVING_TEST_CASE:
            result = get_param_files(f'data/parameter_yaml/attack/{test_case}')
            param_file_counts += len(result)
    if 'complex_maneuver' in TEST_CASES:
        for test_case in COMP_MAN_TEST_CASE:
            result = get_param_files(f'data/parameter_yaml/attack/{test_case}')
            param_file_counts += len(result)
    print(f"receive {param_file_counts * len(BASELINE_FIRMWARE_PATH)} files")

    for baseline_name in BASELINE_FIRMWARE_PATH.keys():
        for test_case in TEST_CASES:
            if test_case == 'hover_test':
                attack_cases = HOVERING_TEST_CASE
            elif test_case == 'moving_test':
                attack_cases = MOVING_TEST_CASE
            elif test_case == 'complex_maneuver':
                attack_cases = COMP_MAN_TEST_CASE
            else:
                attack_cases = [None]

            for attack_case in attack_cases:
                n_trial = NUM_ATTACK_TRIAL
                if attack_case is None or attack_case == "no_attack_or_detection":
                    n_trial = NUM_NORMAL_TRIAL

                attack_timeout = 90
                cfg = generate_config(baseline_name, test_case, attack_case)
                run(cfg, n_trial, attack_timeout)
