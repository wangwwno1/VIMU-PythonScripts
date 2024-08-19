import numpy as np
import pandas as pd
import pyulog
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.spatial.transform import Slerp, Rotation as R
from sklearn.svm import LinearSVR

from .file_utils import StateData


def read_ulog(ulog_filename, messages=None):
    """
    Convert ulog to pandas dataframe.
    """
    log = pyulog.ULog(ulog_filename, messages)
    return extract_ulog_dict(log)


def extract_ulog_dict(ulog_obj):
    data_dict = {}
    for msg in ulog_obj.data_list:
        msg_data = pd.DataFrame.from_dict(msg.data)
        timestamp_ns = pd.to_timedelta(msg_data['timestamp'].rename('timestamp_ns'), unit='us')
        msg_data.set_index(timestamp_ns, drop=True, inplace=True)
        data_dict['{:s}_{:d}'.format(msg.name, msg.multi_id)] = msg_data
    return data_dict


def slerp_interpolate(input_quaternions: np.ndarray, key_times: np.ndarray, interpolate_times: np.ndarray):
    """
    Interpolate quaternions using the timestamp as time.
    :param input_quaternions: (N, 4) Estimated quaternions in order (w, i, j, k)
    :param key_times: (N, ) shape timestamps that aligned with input quaternions
    :param interpolate_times: (M, ) shape timestamps that need to interpolate, should within the range of key_times
    :return: (M, 4) interpolated quaternions with order (w, i, j, k)
    """
    # Scipy quaternion use (i, j, k, w) convention, so we swap the order first before do the interpolation
    xyzw_quats = input_quaternions.take([1, 2, 3, 0], axis=-1)
    slerp = Slerp(key_times, R.from_quat(xyzw_quats))
    interpolated_quaternions = slerp(interpolate_times).as_quat().take([3, 0, 1, 2], axis=-1)
    return interpolated_quaternions


def align_and_interpolate(estimate_state: StateData, groundtruth: StateData):
    aligned_estimate = {}
    aligned_groundtruth = {}
    for k, est_frame in estimate_state.items():
        if est_frame is None:
            continue
        # Clip the groundtruth timestamp to the range of estimation, also remove duplicated index
        est_frame = est_frame.loc[~est_frame.index.duplicated(), :]
        gt_frame = groundtruth[k]
        gt_frame = gt_frame.loc[~gt_frame.index.duplicated(), :]
        gt_frame = gt_frame.loc[(est_frame.index.min() <= gt_frame.index) & (gt_frame.index <= est_frame.index.max())]
        aligned_frame, _ = est_frame.align(gt_frame, 'outer', axis=0)
        if k == 'attitude':
            # Do Spherical linear interpolation for attitude quaternions
            diff_index = gt_frame.index.difference(est_frame.index)
            if len(diff_index) > 0:
                interpolated = slerp_interpolate(est_frame.values, est_frame.index, diff_index)
                aligned_frame.loc[diff_index] = interpolated.astype(np.float32)
        else:
            # Use 1st order spline interpolate for position, velocity and angular rate
            aligned_frame.interpolate(method='slinear', limit_direction='both', limit_area='inside', inplace=True)

        # Drop nan value and align the estimation timestamp with the groundtruth
        aligned_frame.dropna(inplace=True)
        aligned_estimate[k] = aligned_frame.loc[gt_frame.index]
        aligned_groundtruth[k] = gt_frame

    return StateData(**aligned_estimate), StateData(**aligned_groundtruth)


def fast_ema(data: pd.DataFrame, alpha: pd.Series):
    split_idxes = np.arange(1, len(data), 1000)
    split_idxes[-1] = len(data)
    start_end = np.stack((split_idxes[:-1], split_idxes[1:]), axis=-1)

    # Note: joblib loky backend does not release worker automatically
    # Use code snippet below to terminate them:
    # from joblib.externals.loky import get_reusable_executor
    # get_reusable_executor().shutdown(wait=True)
    para_module = Parallel(n_jobs=-1, verbose=0)
    result = para_module(
        delayed(exponential_moving_weight)
        (data.iloc[start:end], alpha.iloc[start:end], data.iloc[start - 1].values)
        for start, end in start_end
    )

    # Modify the initial state (x_0) can correct the averaged state while keep the error within float eps < 1e-14.
    # y_n - x_n
    # = \sum^{n-1}_{i=1} \prod^{n}_{j=i+1} a_j x_i - \sum^{n}_{i=1} \prod^n_{j=i} a_j x_i + \prod^n_{i=1} a_i x_0
    initial_state = result[0][-1]
    for sub_frame, (start, end) in zip(result, start_end):
        if start > 1:
            alpha_arr = alpha.iloc[start:end].values
            alpha_cumprod = np.cumprod(alpha_arr).reshape(-1, 1)
            sub_frame += alpha_cumprod * np.atleast_2d(initial_state - data.iloc[start - 1].values)

        initial_state = sub_frame[-1]

    data_copy = data.copy()
    data_copy.iloc[1:] = np.concatenate(result, dtype=np.float32)

    return data_copy


def exponential_moving_weight(data: pd.DataFrame, alpha: pd.Series, initial_state: np.ndarray):
    last_state = initial_state.copy()
    result = np.empty(data.shape, dtype=np.float64)
    weight = 1. - alpha
    for idx, (t, signal) in enumerate(data.iterrows()):
        last_state += (signal - last_state) * weight[t]
        result[idx] = last_state
    return result


def estimate_voltage_scale(battery_status: pd.DataFrame, internal_resistance_ohm: float, reference_voltage: float,
                           use_filtered_data: bool = False, clip_scale: bool = True):
    # Use column list to ensure the result is a pd.DataFrame instead of a pd.Series
    middle_name = '_filtered' if use_filtered_data else ''
    result = pd.DataFrame(index=battery_status.index)
    voltage_drop = battery_status[f'current{middle_name}_a'] * internal_resistance_ohm
    voltage_no_load = battery_status[f'voltage{middle_name}_v'] + voltage_drop
    result['scale'] = voltage_no_load / reference_voltage
    if clip_scale:
        result['scale'] = np.minimum(result['scale'], 1.0)

    return result


def estimate_internal_resistance(battery_status: pd.DataFrame):
    battery_time = battery_status.index.total_seconds()
    battery_time = battery_time.values - battery_time[0]

    # Acquire battery voltage after disarmed (current_a = 0)
    end_voltage = battery_status['voltage_v'].iloc[-10:].median()

    X = np.concatenate([battery_time[::-1, np.newaxis], battery_status[['current_a']].values], axis=1)
    y = end_voltage - battery_status['voltage_v']

    model = LinearSVR(loss='squared_epsilon_insensitive', fit_intercept=True, tol=1e-6, dual='auto')

    # When battery voltage > 4.05V, it won't drop linearly with time
    # So we clip data points with voltage > 4.05V
    cell_count = battery_status.cell_count.iloc[0]
    initial_estimate_voltage = battery_status['voltage_v'] + 0.02 * cell_count * battery_status['current_a']
    mask = (initial_estimate_voltage < 4.05 * cell_count)
    X, y = X[mask], y[mask]

    # Fit model again with corrected data
    model.fit(X, y)  # Initial fitting
    print(f"Final Fitting Score: {model.score(X, y)}")
    print(f"Final Coef Importance: {np.round(model.coef_ / np.linalg.norm(model.coef_), decimals=3)}")
    int_r_ohm = model.coef_[-1]
    print(f"Estimated Internal Resistance (Ohm): {int_r_ohm}")

    estimated_voltage_unload = battery_status['voltage_v'] + int_r_ohm * battery_status['current_a']
    mean_error = np.nanmedian(int_r_ohm * battery_status['current_a'])

    fig, ax = plt.subplots(1, sharex=True)
    ax.plot(battery_time, battery_status['voltage_v'])
    ax.plot(battery_time, estimated_voltage_unload)
    ax.plot(battery_time, estimated_voltage_unload - mean_error, alpha=0.8)
    plt.show()

    return int_r_ohm


def calculate_control_input(motor_outputs: pd.DataFrame, voltage_scale: pd.DataFrame = None, tau_s: float = 0.0,
                            pwm_main_min: int = 1000, pwm_main_max: int = 2000):
    motor_outputs: pd.DataFrame = motor_outputs.copy()
    motor_outputs.iloc[1:] = motor_outputs.iloc[:-1]
    scaled_signal: pd.DataFrame = (motor_outputs - pwm_main_min) / (pwm_main_max - pwm_main_min)

    if voltage_scale is not None:
        voltage_scale: pd.DataFrame = voltage_scale.copy()
        voltage_scale.iloc[1:] = voltage_scale.iloc[:-1]
        scaled_signal = scaled_signal * np.atleast_2d(voltage_scale.values)

    if tau_s > 1e-4:
        tau_s = pd.to_timedelta(tau_s, unit='s')
        dt = (scaled_signal.index[1:] - scaled_signal.index[:-1])
        alpha = pd.Series(np.concatenate([[1], np.exp(-dt / tau_s)]), index=scaled_signal.index)
        scaled_signal = fast_ema(scaled_signal, alpha)
        if voltage_scale is not None:
            voltage_scale = fast_ema(voltage_scale, alpha)

    return scaled_signal, voltage_scale
