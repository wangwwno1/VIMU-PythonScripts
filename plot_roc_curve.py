import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

DETECTOR_TYPES = ['L1TW', 'L2TW', 'CUSUM', 'EWMA']
BASELINE_NAME_MAP = {
    'control_invariant': 'L2TW (CI)',
    'software_sensor_sup': 'L1TW (SRR)',
    'savior': 'CUSUM (SAVIOR)',
    'virtual_imu_cusum': 'CUSUM (VIMU-CS)',
    'virtual_imu': 'CS-EMA (VIMU)',
}

ATTACK_NAME_MAP = {
    'gps_overt_attack': r'$OA_{GPS}$',
    'tmr_test_default': r'$OA^{2/3}_{Gyro}$',
    'gyro_overt_attack_default': r'$OA^{3/3}_{Gyro}$',
    'gyro_overt_attack_icm20602': r'$OA^{3/3}_{ICM20602}$',
    'gyro_overt_attack_icm20689': r'$OA^{3/3}_{ICM20689}$',
}

GPS_ATTACK_ORDER = [r'$OA_{GPS}$']
GYRO_ATTACK_ORDER = [r'$OA^{2/3}_{Gyro}$', r'$OA^{3/3}_{Gyro}$', r'$OA^{3/3}_{ICM20689}$']

file_folder = "data/evaluation_reports/IMU Overt Attack ROC 04.03"

def read_threshold_files(folder_path):
    threshold_files = []
    result_columns = ['baseline_name', 'test_case', 'scene_name', 'sensor_type', 'detector_type',
                      'FPR', 'TPR', 'Threshold']
    for detector_type in set(DETECTOR_TYPES):
        fpath = f'{folder_path}/{detector_type}_roc_curve_threshold.csv'
        sub_frame = pd.read_csv(fpath, index_col=False)
        if detector_type == 'EWMA':
            # Discard result on ICM-20602 & ICM-20689
            filter_mask = (sub_frame['test_case'] == 'gyro_overt_attack_icm20602') |\
                          (sub_frame['test_case'] == 'gyro_overt_attack_icm20689') |\
                          (sub_frame['test_case'] == 'gps_overt_attack')
            sub_frame = sub_frame.loc[~filter_mask]
            GYRO_MASK = (sub_frame['sensor_type'] == 'gyroscope')
            GYRO_MASK &= np.isclose(sub_frame['alpha'], 0.01) & np.isclose(sub_frame['cap'], 0.85)
            GPS_MASK = (sub_frame['sensor_type'] == 'gps_position')
            GPS_MASK &= np.isclose(sub_frame['alpha'], 0.01) & np.isclose(sub_frame['cap'], np.inf)
            PARAM_MASK = GPS_MASK | GYRO_MASK
        elif detector_type == 'CUSUM':
            # SAVIOR VIMU+CUSUM VIMU-GPS
            param_name = 'mean_shift'
            gps_param_value = 0.5
            gyro_param_value = 0.5
            sub_frame = sub_frame.loc[sub_frame['baseline_name'] != 'virtual_imu']
            # Discard result on ICM-20602 & ICM-20689
            filter_mask = (sub_frame['test_case'] == 'tmr_test_default') |\
                          (sub_frame['test_case'] == 'gyro_overt_attack_default')
            filter_mask &= sub_frame['baseline_name'] == 'virtual_imu_cusum_angvel'
            sub_frame = sub_frame.loc[~filter_mask]
            sub_frame['baseline_name'] = sub_frame['baseline_name'].replace('virtual_imu_cusum_angvel', 'virtual_imu')

            SOLUTION_MASK = ((sub_frame['baseline_name'] == 'savior') |
                             (sub_frame['baseline_name'] == 'virtual_imu_cusum') |
                             (sub_frame['baseline_name'] == 'virtual_imu'))
            GPS_MASK = (sub_frame['sensor_type'] == 'gps_position') & np.isclose(sub_frame[param_name], gps_param_value)
            GYRO_MASK = (sub_frame['sensor_type'] == 'gyroscope') & np.isclose(sub_frame[param_name], gyro_param_value)
            PARAM_MASK = GPS_MASK | (SOLUTION_MASK & GYRO_MASK)
        else:
            if detector_type == 'L2TW':
                # Control Invariant
                param_name = 'time_window'
                gps_param_value = 16
                gyro_param_value = 20
            elif detector_type == 'L1TW':
                # Software Sensor
                param_name = 'time_window'
                gps_param_value = 10
                gyro_param_value = 10
            else:
                raise RuntimeError(f"Unknown Detector Type: {detector_type}!")

            GPS_MASK = (sub_frame['sensor_type'] == 'gps_position') & np.isclose(sub_frame[param_name], gps_param_value)
            GYRO_MASK = (sub_frame['sensor_type'] == 'gyroscope') & np.isclose(sub_frame[param_name], gyro_param_value)
            PARAM_MASK = GPS_MASK | GYRO_MASK

        threshold_files.append(sub_frame.loc[PARAM_MASK, result_columns])
    return pd.concat(threshold_files)

# fpr_label = 'False Positive Rate'
# tpr_label = 'True Positive Rate'

fpr_label = 'FPR'
tpr_label = 'TPR'
df = read_threshold_files(file_folder)
df = df.rename(columns=dict(
    baseline_name='Solution',
    scene_name='Scene',
    test_case='Attack',
    TPR=tpr_label,
    FPR=fpr_label
))

df['Solution'] = df['Solution'].replace(BASELINE_NAME_MAP)
df['Attack'] = df['Attack'].replace(ATTACK_NAME_MAP)

with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    HEIGHT = 1.75  # inches
    WIDTH = 1.25  # inches
    grid_kwargs = dict(
        sharex=True,
        sharey=True,
        margin_titles=True,
        despine=False
    )

    other_kwargs = dict(
        height=HEIGHT, aspect=WIDTH / HEIGHT,
        linewidth=1.0,
        estimator=None,
    )
    plot = sns.relplot(df, x=fpr_label, y=tpr_label,
                       hue='Solution',
                       hue_order=list(BASELINE_NAME_MAP.values()),
                       row='Scene', row_order=['Hovering', 'Moving'],
                       col='Attack', col_order=GYRO_ATTACK_ORDER,
                       kind="line",
                       facet_kws=grid_kwargs,
                       **other_kwargs)
    plot.legend.set_title(None)
    sns.move_legend(
        plot,
        loc="lower center",
        ncol=3,
    )
    plot.set_titles(
        row_template="{row_name}",
        col_template="{col_name}"
    )

    for ax in plot.axes.flatten():
        # ax.set_title(ax.title.get_text(), pad=10.0)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 0.1, 0.2, 0.3, 0.4, None])
        yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_yticks(np.round(yticks, 2), np.round(yticks, 2))

        ax.set_xlim([-0.0025, 0.195])
        ax.set_ylim([0.0, 1.005])

    plot.tight_layout(pad=0.5, rect=[0.0, 0.12, 1.0, 1.0])
    plot.figure.subplots_adjust(wspace=0.05, hspace=0.15)

plot.figure.show()

with PdfPages('ROC Curves - Gyro.pdf') as pdf_pages:
    pdf_pages.savefig(plot.figure)


file_folder = "data/evaluation_reports/GPS Overt Attack ROC - 01.22"

df = read_threshold_files(file_folder)
df = df.rename(columns=dict(
    baseline_name='Solution',
    scene_name='Scene',
    test_case='Attack',
    TPR=tpr_label,
    FPR=fpr_label
))

df['Solution'] = df['Solution'].replace(BASELINE_NAME_MAP)
df['Attack'] = df['Attack'].replace(ATTACK_NAME_MAP)

with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    HEIGHT = 2.5  # inches
    WIDTH = 1.7  # inches
    grid_kwargs = dict(
        sharex=True,
        sharey=True,
        margin_titles=True,
        despine=False
    )

    other_kwargs = dict(
        height=HEIGHT, aspect=WIDTH / HEIGHT,
        linewidth=1.0,
        estimator=None,
    )
    plot = sns.relplot(df, x=fpr_label, y=tpr_label,
                       hue='Solution',
                       hue_order=list(BASELINE_NAME_MAP.values()),
                       col='Scene', col_order=['Hovering', 'Moving'],
                       row='Attack', row_order=GPS_ATTACK_ORDER,
                       kind="line",
                       facet_kws=grid_kwargs,
                       **other_kwargs)
    plot.legend.set_title(None)
    sns.move_legend(
        plot,
        loc="lower center",
        ncol=3,
        framealpha=1.0,
        columnspacing=0.5,
        labelspacing=0.25
    )
    plot.set_titles(
        row_template="{row_name}",
        col_template="{col_name}"
    )

    for ax in plot.axes.flatten():
        # ax.set_title(ax.title.get_text(), pad=10.0)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 0.1, 0.2, 0.3, 0.4, None])
        yticks = [0.4, 0.6, 0.8, 1.0]
        ax.set_yticks(np.round(yticks, 2), np.round(yticks, 2))

        ax.set_xlim([-0.001, 0.195])
        ax.set_ylim([0.375, 1.005])

    plot.tight_layout(pad=0.25, rect=[0.0, 0.175, 1.0, 1.0])
    plot.figure.subplots_adjust(wspace=0.05, hspace=0.15)

plot.figure.show()

with PdfPages('ROC Curves - GPS.pdf') as pdf_pages:
    pdf_pages.savefig(plot.figure)
