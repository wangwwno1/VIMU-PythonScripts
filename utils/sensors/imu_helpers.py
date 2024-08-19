import pandas as pd


# Only need to store: measurement, reference, error_ratio, test_ratio
def extract_imu_data(sensor_errors_raw: pd.DataFrame, sensor_noise: float):
    sensor_errors = extract_imu_errors(sensor_errors_raw)
    sensor_errors = sensor_errors.assign(
        x_error_ratio=sensor_errors.loc[:, 'x'] / sensor_noise,
        y_error_ratio=sensor_errors.loc[:, 'y'] / sensor_noise,
        z_error_ratio=sensor_errors.loc[:, 'z'] / sensor_noise
    )

    sensor_errors = sensor_errors.reset_index(drop=True)
    reordered_columns = ['timestamp', 'timestamp_reference']
    for suffix in ('', '_error_ratio'):
        reordered_columns += [f'x{suffix}', f'y{suffix}', f'z{suffix}']
    reordered_columns += ['test_ratio']
    sensor_errors = sensor_errors.loc[:, reordered_columns]
    return sensor_errors


def extract_imu_errors(imu_errors_raw: pd.DataFrame):
    """
    :param imu_errors_raw:
        raw dataframe from topics 'sensor_accel_errors', 'sensor_gyro_errors', or equivalent formats.
        can hold multiple measurements within an interval.
    :return: xyz error and test ratios that are flattened along sample axis
    """
    cols = ['timestamp', 'timestamp_reference', 'device_id', 'x', 'y', 'z', 'test_ratio']

    # Concatenate all samples, ordered by timestamp first, then the arrival within time window.
    # Use 'stable' sort to preserve measurement orders
    output = imu_errors_raw.loc[:, cols].reset_index(drop=True)
    output = output.sort_values('timestamp', kind='stable', ignore_index=True)
    return output


