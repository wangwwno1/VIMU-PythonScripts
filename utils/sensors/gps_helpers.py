import pandas as pd

__all__ = ['reconstruct_gps_state', 'get_gps_error_ratio', 'calculate_gps_error_ratio']


def reconstruct_gps_state(vehicle_reference_states: pd.DataFrame, sensor_gps_error: pd.DataFrame):
    """
    :param vehicle_reference_states: reference state from 'vehicle_reference_states' topic
    :param sensor_gps_error: detector records from 'sensor_gps_error' topic
    :return:
    """

    # state vel - 4-6, pos - 7-9
    ref_cols = ['timestamp', 'timestamp_sample'] + [f'states[{i}]' for i in range(4, 10)]
    ref_states = vehicle_reference_states.loc[:, ref_cols]
    ref_states.columns = ['timestamp', 'timestamp_sample'] + ['vx_ref', 'vy_ref', 'vz_ref', 'x_ref', 'y_ref', 'z_ref']

    # Reset index & align dataframe with timestamp
    ref_states.set_index('timestamp_sample', inplace=True)
    sensor_gps_error = sensor_gps_error.set_index("timestamp_reference")

    ref_gps_data, gps_data = ref_states.align(sensor_gps_error, 'inner', axis=0)
    state_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    ref_gps_cols = ['x_ref', 'y_ref', 'z_ref', 'vx_ref', 'vy_ref', 'vz_ref']

    gps_data.columns = ['timestamp'] + state_cols
    ref_gps_data = ref_gps_data.loc[:, ['timestamp'] + ref_gps_cols]
    gps_data.loc[:, state_cols] = ref_gps_data.loc[:, ref_gps_cols].values - gps_data.loc[:, state_cols].values
    gps_data.reset_index(drop=True, inplace=True)
    gps_data = pd.merge(gps_data, ref_gps_data, suffixes=('', '_right'))

    return gps_data


def calculate_gps_error_ratio(sensor_gps_error: pd.DataFrame,
                              sensor_gps_error_variances: pd.DataFrame = None,
                              position_variances=None, velocity_variances=None):
    pos_cols = [f'position_error[{idx}]' for idx in range(3)]
    vel_cols = [f'velocity_error[{idx}]' for idx in range(3)]
    if sensor_gps_error_variances:
        position_variances = sensor_gps_error_variances.loc[:, pos_cols]
        position_variances = position_variances.values
        velocity_variances = sensor_gps_error_variances.loc[:, vel_cols]
        velocity_variances = velocity_variances.values
    error_ratios = sensor_gps_error.copy()
    error_ratios.loc[:, pos_cols] /= position_variances
    error_ratios.loc[:, vel_cols] /= velocity_variances
    return error_ratios


def get_gps_error_ratio(sensor_gps_error_ratios: pd.DataFrame, pos_only: bool = False, vel_only: bool = False):
    sensor_gps_error_ratios = sensor_gps_error_ratios.copy()
    if pos_only:
        # Discard velocity error
        sensor_gps_error_ratios.drop(columns=[f'velocity_error[{idx}]' for idx in range(3)], inplace=True)
    elif vel_only:
        # Discard position error
        sensor_gps_error_ratios.drop(columns=[f'position_error[{idx}]' for idx in range(3)], inplace=True)

    col_mapper = {f'position_error[{idx}]': f'{ax}_error_ratio' for idx, ax in zip(range(3), ('x', 'y', 'z'))}
    col_mapper.update({f'velocity_error[{idx}]': f'v{ax}_error_ratio' for idx, ax in zip(range(3), ('x', 'y', 'z'))})
    return sensor_gps_error_ratios.rename(columns=col_mapper)
