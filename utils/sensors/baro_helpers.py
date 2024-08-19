import numpy as np
from typing import List
import pandas as pd


def extract_baro_dataframes(baro_error: pd.DataFrame) -> List[pd.DataFrame]:
    """
    :param ref_states_raw: reference state from 'vehicle_reference_states' topic
    :param baro_error: detector records from 'sensor_baro_error' topic
    :return: List of processed dataframes with each frame represent an barometer
    """

    # state vel - 4-6, pos - 7-9
    result_frames = []
    for idx in range(4):
        if baro_error.loc[:, f'device_ids[{idx}]'].iloc[0] == 0:
            break
        ref_label = f'timestamp_reference[{idx}]'
        err_label = f'error[{idx}]'
        variance_label = f'variance[{idx}]'
        test_ratio_label = f'test_ratio[{idx}]'

        sub_frame = baro_error.loc[:, ['timestamp', ref_label, f'device_ids[{idx}]', err_label,
                                       variance_label, test_ratio_label]]
        duplicate_labels = (err_label, variance_label, test_ratio_label)
        sub_frame = sub_frame.drop_duplicates(duplicate_labels).reset_index(drop=True)

        # sub_frame, align_states = sub_frame.align(align_states, 'inner', axis=0)
        sub_frame = sub_frame.assign(
            error_ratio=sub_frame.loc[:, err_label] / np.sqrt(np.maximum(0.0001, sub_frame.loc[:, variance_label]))
        )
        sub_frame = sub_frame.loc[:, ['timestamp', ref_label, f'device_ids[{idx}]', 'error_ratio', test_ratio_label]]
        sub_frame.rename(columns={ref_label: 'timestamp_reference',
                                  f'device_ids[{idx}]': 'device_id',
                                  test_ratio_label: 'test_ratio'},
                         inplace=True)

        result_frames.append(sub_frame)

    return result_frames
