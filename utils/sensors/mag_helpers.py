import numpy as np
import pandas as pd


__all__ = ['get_mag_error_ratio', 'get_mag_test_ratio']


def get_mag_error_ratio(estimator_innovations: pd.DataFrame,
                        estimator_innovation_variances: pd.DataFrame):
    head_tag = ['heading']
    mag_field_labels = [f'mag_field[{idx}]' for idx in range(3)]

    hdg_labels = ['timestamp'] + head_tag
    hdg_ratios = get_error_ratios(estimator_innovations.loc[:, hdg_labels],
                                  estimator_innovation_variances.loc[:, hdg_labels],
                                  head_tag)

    mag_labels = ['timestamp'] + mag_field_labels
    mag_ratios = get_error_ratios(estimator_innovations.loc[:, mag_labels],
                                  estimator_innovation_variances.loc[:, mag_labels],
                                  mag_field_labels)

    return merge_mag_hdg_and_mag3d(hdg_ratios, mag_ratios)


def get_mag_test_ratio(estimator_innovation_test_ratios: pd.DataFrame):
    head_tag = ['heading']
    mag_field_labels = [f'mag_field[{idx}]' for idx in range(3)]
    hdg_labels = ['timestamp'] + head_tag
    mag_labels = ['timestamp'] + mag_field_labels

    hdg_ratios = estimator_innovation_test_ratios.loc[:, hdg_labels].drop_duplicates(head_tag)
    mag_ratios = estimator_innovation_test_ratios.loc[:, mag_labels].drop_duplicates(mag_field_labels)
    result = merge_mag_hdg_and_mag3d(hdg_ratios, mag_ratios)
    return result


def get_error_ratios(innovations, variances, column_labels):
    innovations, variances = innovations.align(variances, 'inner', axis=0, copy=True)
    innovations.loc[:, column_labels] /= np.sqrt(np.maximum(0.0001, variances.loc[:, column_labels]))
    error_ratios = innovations.drop_duplicates(column_labels)
    return error_ratios


def merge_mag_hdg_and_mag3d(hdg_data, mag_data):
    headings = hdg_data.loc[:, 'heading'].values
    hdg_data = hdg_data.assign(heading_y=headings, heading_z=headings)
    hdg_data.rename(columns={'heading': 'mag_field[0]', 'heading_y': 'mag_field[1]',
                             'heading_z': 'mag_field[2]'}, inplace=True)
    result = pd.concat((hdg_data, mag_data), axis=0, ignore_index=True)
    result = result.sort_values('timestamp', kind='stable', ignore_index=True)
    result = result.rename(columns={f'mag_field[{idx}]': f'{ax}_error_ratio' for idx, ax in enumerate(('x', 'y', 'z'))})
    return result
