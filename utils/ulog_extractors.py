import numpy as np
import pandas as pd
import pyulog
import quaternion

from .mav_utils import AttackType
from .file_utils import StateData, SensorData, ErrorRatioData
from .helper_functions import align_and_interpolate, extract_ulog_dict
from .sensors import get_mag_test_ratio, extract_baro_dataframes, get_gps_error_ratio, get_mag_error_ratio, \
    extract_imu_data, calculate_gps_error_ratio


REF_ATT_MAP = {f'states[{idx}]': f'q[{idx}]' for idx in range(4)}
REF_VEL_MAP = {f'states[{idx + 4}]': f'v{ax}' for idx, ax in enumerate(['x', 'y', 'z'])}
REF_POS_MAP = {f'states[{idx + 7}]': ax for idx, ax in enumerate(['x', 'y', 'z'])}
RATE_MAP = {f'xyz[{idx}]': f'r{ax}' for idx, ax in enumerate(['x', 'y', 'z'])}
REF_STATE_RENAME_MAP = dict(**REF_ATT_MAP, **REF_VEL_MAP, **REF_POS_MAP)


# Can also apply to groundtruth
def extract_vehicle_state(vehicle_local_position: pd.DataFrame,
                          vehicle_attitude: pd.DataFrame,
                          vehicle_angular_velocity: pd.DataFrame):
    local_pos = vehicle_local_position.set_index('timestamp')
    att = vehicle_attitude.set_index('timestamp')
    angular_rate = vehicle_angular_velocity.set_index('timestamp')

    return StateData(
        position=local_pos.loc[:, ['x', 'y', 'z']],
        velocity=local_pos.loc[:, ['vx', 'vy', 'vz']],
        attitude=att.loc[:, [f'q[{i}]' for i in range(4)]],
        angular_rate=angular_rate.loc[:, RATE_MAP.keys()].rename(columns=RATE_MAP)
    )


def extract_reference_state(vehicle_reference_states: pd.DataFrame, reference_angular_velocity: pd.DataFrame):
    ref_state = vehicle_reference_states.set_index('timestamp_sample')
    ref_rate = reference_angular_velocity.set_index('timestamp_sample')

    ref_state.rename(columns=REF_STATE_RENAME_MAP, inplace=True)
    ref_rate.rename(columns=RATE_MAP, inplace=True)

    return StateData(
        position=ref_state.loc[:, REF_POS_MAP.values()],
        velocity=ref_state.loc[:, REF_VEL_MAP.values()],
        attitude=ref_state.loc[:, REF_ATT_MAP.values()],
        angular_rate=ref_rate.loc[:, RATE_MAP.values()]
    )


def extract_recovery_status(ulog: dict):
    vehicle_estimation = extract_vehicle_state(ulog['vehicle_local_position_0'],
                                               ulog['vehicle_attitude_0'],
                                               ulog['vehicle_angular_velocity_0'])
    groundtruth = extract_vehicle_state(ulog['vehicle_local_position_groundtruth_0'],
                                        ulog['vehicle_attitude_groundtruth_0'],
                                        ulog['vehicle_angular_velocity_groundtruth_0'])
    groundtruth.position.loc[:, 'z'] *= -1.  # Convert from Up to Down
    estimate_groundtruth = align_and_interpolate(vehicle_estimation, groundtruth)
    return estimate_groundtruth


def extract_reference_error(ulog: dict):
    groundtruth = extract_vehicle_state(ulog['vehicle_local_position_groundtruth_0'],
                                        ulog['vehicle_attitude_groundtruth_0'],
                                        ulog['vehicle_angular_velocity_groundtruth_0'])
    groundtruth.position.loc[:, 'z'] *= -1.  # Convert from Up to Down

    reference_state = extract_reference_state(ulog['vehicle_reference_states_0'], ulog['reference_angular_velocity_0'])
    reference_groundtruth = align_and_interpolate(reference_state, groundtruth)
    return reference_groundtruth


def extract_test_ratios(ulog: dict):
    gps_test_ratio = ulog['sensors_status_gps_0'].copy()
    gps_test_ratio.rename(columns={'test_ratio': 'test_ratio_0'}, inplace=True)
    gps_test_ratio.set_index('timestamp', drop=True, inplace=True)
    gps_test_ratio = gps_test_ratio.loc[:, ['test_ratio_0']]

    mag_test_ratios = get_mag_test_ratio(ulog['estimator_innovation_test_ratios_0'].copy())
    mag_test_ratios.iloc[:, 1] = mag_test_ratios.iloc[:, 1:].values.max(axis=1)
    mag_test_ratios = mag_test_ratios.iloc[:, [0, 1]]
    mag_test_ratios.columns = ['timestamp', 'test_ratio_0']
    mag_test_ratios.set_index('timestamp', drop=True, inplace=True)

    baro_dataframes = extract_baro_dataframes(ulog['sensor_baro_error_0'].copy())
    baro_test_ratios = _merge_sensor_test_ratios(baro_dataframes)

    accel_test_ratios = []
    gyro_test_ratios = []
    for idx in range(4):
        accel_key = f'sensor_accel_errors_{idx}'
        gyro_key = f'sensor_gyro_errors_{idx}'
        if accel_key in ulog:
            accel_test_ratios.append(ulog[accel_key].loc[:, ['timestamp', 'test_ratio']])
        if gyro_key in ulog:
            gyro_test_ratios.append(ulog[gyro_key].loc[:, ['timestamp', 'test_ratio']])
    accel_test_ratios = _merge_sensor_test_ratios(accel_test_ratios)
    gyro_test_ratios = _merge_sensor_test_ratios(gyro_test_ratios)

    return SensorData(
        gps=[gps_test_ratio],
        barometer=[baro_test_ratios],
        magnetometer=[mag_test_ratios],
        accelerometer=[accel_test_ratios],
        gyroscope=[gyro_test_ratios]
    )


def _merge_sensor_test_ratios(test_ratios):
    merged_dataframe = None
    for idx, df in enumerate(test_ratios):
        sub_frame: pd.DataFrame = df.loc[:, ['timestamp', 'test_ratio']].copy()
        sub_frame.rename(columns={'test_ratio': f'test_ratio_{idx}'}, inplace=True)
        sub_frame.set_index('timestamp', inplace=True)
        if merged_dataframe is None:
            merged_dataframe = sub_frame
        else:
            merged_dataframe = pd.merge(merged_dataframe, sub_frame, how='outer', left_index=True, right_index=True)
    if merged_dataframe is not None:
        merged_dataframe.fillna(method='ffill', axis=0, inplace=True)
    return merged_dataframe


def extract_error_ratios(ulog_obj: pyulog.ULog):
    ulog_dict = extract_ulog_dict(ulog_obj)

    gps_position_error_ratio = None
    gps_velocity_error_ratio = None
    has_gps_data = 'sensor_gps_error_ratios_0' in ulog_dict or 'sensor_gps_error_0' in ulog_dict
    if has_gps_data:
        if 'sensor_gps_error_ratios_0' in ulog_dict:
            sensor_gps_error_ratios = ulog_dict['sensor_gps_error_ratios_0']
        else:
            sensor_gps_error = ulog_dict['sensor_gps_error_0']
            if 'sensor_gps_error_variances_0' in ulog_dict:
                sensor_gps_error_variances = ulog_dict['sensor_gps_error_variances_0']
                sensor_gps_error_ratios = calculate_gps_error_ratio(sensor_gps_error, sensor_gps_error_variances)
            else:
                position_variances = ulog_obj.initial_parameters.get('EKF2_GPS_P_NOISE', 0.01)
                velocity_variances = ulog_obj.initial_parameters.get('EKF2_GPS_V_NOISE', 0.01)
                sensor_gps_error_ratios = calculate_gps_error_ratio(sensor_gps_error,
                                                                    position_variances=position_variances,
                                                                    velocity_variances=velocity_variances)

        gps_position_error_ratio = [get_gps_error_ratio(sensor_gps_error_ratios, pos_only=True)]
        gps_velocity_error_ratio = [get_gps_error_ratio(sensor_gps_error_ratios, vel_only=True)]

    baro_dataframes = None
    if 'sensor_baro_error_0' in ulog_dict:
        # print(f"Processing baro error ratio!")
        baro_error = ulog_dict['sensor_baro_error_0']
        baro_dataframes = extract_baro_dataframes(baro_error)
        if len(baro_dataframes) == 0:
            print(f"Cannot find any valid barometer data in sensor_baro_error topic, how could that possible?!")
            baro_dataframes = None

    mag_error_ratio = None
    if 'estimator_innovations_0' in ulog_dict and 'estimator_innovation_variances_0' in ulog_dict:
        # print(f"Processing mag error ratio!")
        estimator_innovations = ulog_dict['estimator_innovations_0']
        estimator_innovation_variances = ulog_dict['estimator_innovation_variances_0']
        mag_error_ratio = [get_mag_error_ratio(estimator_innovations, estimator_innovation_variances)]

    accel_dataframes = []
    for idx in range(4):
        if f'sensor_accel_errors_{idx}' in ulog_dict:
            # print(f'\tsensor_accel_errors_{idx} detected!')
            accel_dataframes.append(extract_imu_data(ulog_dict[f'sensor_accel_errors_{idx}'], 0.35))
    if len(accel_dataframes) == 0:
        # No accel data acquired
        accel_dataframes = None

    gyro_dataframes = []
    for idx in range(4):
        if f'sensor_gyro_errors_{idx}' in ulog_dict:
            # print(f'\tsensor_gyro_errors_{idx} detected!')
            gyro_dataframes.append(extract_imu_data(ulog_dict[f'sensor_gyro_errors_{idx}'], 0.1))
    if len(gyro_dataframes) == 0:
        # No gyro data acquired
        gyro_dataframes = None

    output = ErrorRatioData(
        gps_position=gps_position_error_ratio,
        gps_velocity=gps_velocity_error_ratio,
        barometer=baro_dataframes,
        magnetometer=mag_error_ratio,
        accelerometer=accel_dataframes,
        gyroscope=gyro_dataframes
    )

    # GPS attack will affect both state
    attack_flag = dict(
        gps_position=AttackType.GpsPosition | AttackType.GpsVelocity | AttackType.GpsJointPVAttack,
        gps_velocity=AttackType.GpsVelocity | AttackType.GpsPosition | AttackType.GpsJointPVAttack,
        barometer=AttackType.Barometer,
        magnetometer=AttackType.Magnetometer,
        accelerometer=AttackType.Accelerometer,
        gyroscope=AttackType.Gyroscope
    )

    # Get the attacked instance for multi-instance sensor
    attacked_instance = dict(
        ATK_MULTI_IMU=0,
        ATK_MULTI_MAG=0,
        ATK_MULTI_BARO=0
    )
    for k in attacked_instance.keys():
        attacked_instance[k] = ulog_obj.initial_parameters.get(k, 0)
    for _, param_name, param_value in ulog_obj.changed_parameters:
        if param_name in attacked_instance:
            attacked_instance[param_name] = param_value

    attack_triples = find_attack_timestamp(ulog_obj)
    attack_types = 0
    attack_start = np.inf
    if len(attack_triples) > 0:
        attack_start, attack_types, _ = attack_triples[0]

    # Label the error ratios
    for k in attack_flag.keys():
        ratio_list = output.get(k)
        if ratio_list is not None:
            has_attack = attack_types & attack_flag[k]
            if k == 'barometer':
                affected_instance = attacked_instance['ATK_MULTI_BARO']
            elif k == 'magnetometer':
                affected_instance = attacked_instance['ATK_MULTI_MAG']
            elif k == 'accelerometer' or k == 'gyroscope':
                affected_instance = attacked_instance['ATK_MULTI_IMU']
            else:
                affected_instance = int(bool(has_attack))

            for inst_id, df in enumerate(ratio_list):
                if has_attack and (1 << inst_id) & affected_instance:
                    df.loc[:, 'is_attacked'] = (df.loc[:, 'timestamp'].values >= attack_start)
                else:
                    df.loc[:, 'is_attacked'] = False

    return output


def find_attack_timestamp(ulog_obj: pyulog.ULog):
    # Find first pair of attack timestamps
    attack_timestamp = None
    attack_value = None
    results = []
    for timestamp, param_name, param_value in ulog_obj.changed_parameters:
        if param_name == 'ATK_APPLY_TYPE':
            if attack_timestamp is None and param_value > 0:
                attack_timestamp = timestamp + ulog_obj.initial_parameters['ATK_COUNTDOWN_MS'] * 1000
                attack_value = param_value
            if attack_timestamp is not None and param_value == 0:
                if timestamp > attack_timestamp:
                    end_timestamp = timestamp
                    results.append((attack_timestamp, attack_value, end_timestamp))
                attack_timestamp = None
    if attack_timestamp is not None:
        results.append((attack_timestamp, attack_value, None))

    return results


def calculate_body_velocity(vehicle_attitudes: pd.DataFrame, local_velocity_ned: pd.DataFrame):
    # Calculate Velocity in FRD Body Frame
    aligned_quaternions, _ = vehicle_attitudes.align(local_velocity_ned, axis=0)
    aligned_quaternions.bfill(axis=0, inplace=True)
    aligned_quaternions = aligned_quaternions.loc[local_velocity_ned.index, [f"q[{i}]" for i in range(4)]]
    aligned_quaternions.dropna(inplace=True)

    aligned_index = aligned_quaternions.index
    aligned_local_velocity = local_velocity_ned.loc[aligned_index, ['vx', 'vy', 'vz']]
    quat_arr = quaternion.as_quat_array(aligned_quaternions.values)
    q_vl = quaternion.as_quat_array(
        np.concatenate([
            np.zeros((len(aligned_local_velocity), 1)), aligned_local_velocity.values
        ], axis=-1)
    )
    q_vb = quat_arr.conjugate() * q_vl * quat_arr
    local_vel_body = quaternion.as_float_array(q_vb)[:, 1:4]

    return pd.DataFrame(local_vel_body, index=aligned_index, columns=['vx', 'vy', 'vz'])


def filter_takeoff_land_data(data, dist_bottom, clip_height_m: float = 1.0):
    aligned_dist_bottom = pd.merge_asof(pd.DataFrame(index=data.index), dist_bottom,
                                        left_index=True, right_index=True, direction='nearest')
    takeoff_mask = (aligned_dist_bottom['dist_bottom'] > clip_height_m).values
    takeoff_mask[np.nancumsum(takeoff_mask, dtype=bool) & (np.nancumsum(takeoff_mask[::-1], dtype=bool)[::-1])] = True
    return data[takeoff_mask]


def get_motor_relative_velocity(
        body_velocity: pd.DataFrame,
        angular_velocity: pd.DataFrame,
        position_to_cog: np.ndarray
):
    num_motors = position_to_cog.shape[0]
    rel_vel = []
    for idx in range(num_motors):
        rel_pos = position_to_cog[idx]
        motor_vel_body = body_velocity + np.cross(angular_velocity.values, rel_pos)
        motor_vel_body = motor_vel_body.values
        rel_vel.append(motor_vel_body)

    rel_vel = np.stack(rel_vel, axis=1)
    rel_vel = rel_vel.transpose((2, 0, 1))

    return rel_vel


