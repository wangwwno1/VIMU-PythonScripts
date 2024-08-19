import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TTD_LABEL = "TTD (ms)"


report_folder = "data/evaluation_reports/Effect of Wind Disturbance - 05.27"

time_to_detection = pd.read_csv(f"{report_folder}/full_detection_reports.csv", index_col=False)

ATTACK_NAME_MAP = {
    'gyro_stealthy_attack_wind_mean_0.0': r'$SA^{3/3}_{Gyro}$',
    'gyro_stealthy_attack_wind_mean_1.0': r'$SA^{3/3}_{Gyro}$',
    'gyro_stealthy_attack_wind_mean_2.0': r'$SA^{3/3}_{Gyro}$',
    'gyro_stealthy_attack_wind_mean_5.0': r'$SA^{3/3}_{Gyro}$',
}

WIND_SETTING_NAME_MAP = {
    'no_attack_or_detection': 'Default',
    'no_attack_or_detection_wind_north_mean_1.0': '1.0',
    'no_attack_or_detection_wind_north_mean_2.0': '2.0',
    'no_attack_or_detection_wind_north_mean_5.0': '5.0',
    'gyro_stealthy_attack_wind_mean_0.0': 'Default',
    'gyro_stealthy_attack_wind_mean_1.0': '1.0',
    'gyro_stealthy_attack_wind_mean_2.0': '2.0',
    'gyro_stealthy_attack_wind_mean_5.0': '5.0',
}

WIND_SETTING_ORDER = ['Default', '1.0', '2.0', '5.0']


def replace_labels(data):
    data = data.rename(columns=dict(
        baseline_name='Solution',
        scene_name='Scene',
        test_case='Attack'
    ))
    data['Setting'] = data['Attack']
    data['Attack'] = data['Attack'].replace(ATTACK_NAME_MAP)
    data['Setting'] = data['Setting'].replace(WIND_SETTING_NAME_MAP)
    return data


time_to_detection = replace_labels(time_to_detection)
time_to_detection = time_to_detection.rename(columns=dict(time_to_detection_ms=TTD_LABEL))
time_to_detection = time_to_detection.loc[time_to_detection.loc[:, 'is_attacked']]
x_axis_column = TTD_LABEL
y_axis_column = 'Setting'
time_to_detection = time_to_detection.loc[:, ['Solution', 'Attack', 'Scene', x_axis_column, y_axis_column]]

SCALE_SEC_TO_MS = 1e3

WIND_VEL_LABEL = "Wind Speed (m/s)"


def mark_recovery_performance(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()

    recovery_durations_s = dataframe['recovery_duration_s']

    dataframe[rating_column] = rating_order[-1]
    dataframe.loc[recovery_durations_s > 300.0, rating_column] = rating_order[0]
    dataframe.loc[recovery_durations_s.between(60, 300.0, 'right'), rating_column] = rating_order[1]
    dataframe.loc[recovery_durations_s.between(10, 60.0, 'right'), rating_column] = rating_order[2]
    return dataframe


with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 1.75))
    sns.stripplot(
        time_to_detection, x=x_axis_column, y=y_axis_column,
        hue='Scene',
        dodge=True,  # Separate the result by class
        alpha=0.25,
        orient='h',
        ax=ax1
    )
    sns.despine(ax=ax1, top=True, right=True, left=True)
    legend = ax1.legend(handlelength=0.5, labelspacing=0.1, handletextpad=0.2, columnspacing=0.5)
    legend.set_title(None)
    # Set legend symbol opacity
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    sns.move_legend(ax1, loc="lower left")
    ax1.set(ylabel=WIND_VEL_LABEL)
    ax1.set_xlim(220, 281)  # in millisecond (ms)

    plt.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 1.0))
    plt.show()

    with PdfPages('Time to Detection - External Disturbance.pdf') as pdf_pages:
        pdf_pages.savefig(fig)

y_axis_column = rating_column = "Duration (s)"
x_axis_column = setting_column = 'Setting'
rating_order = ['> 300', '60~300', '10~60', '< 10']
scene_order = ['Hovering', 'Moving']

recovery_durations = pd.read_csv(f"{report_folder}/recovery_reports.csv",
                                 index_col=False)
recovery_durations = replace_labels(recovery_durations)
recovery_durations = mark_recovery_performance(recovery_durations)
recovery_durations = recovery_durations.loc[:, ['Solution', 'Scene', 'Attack', setting_column, rating_column]]


rating_series = recovery_durations[rating_column]
rating_series = rating_series.astype(str).astype("category")
rating_series = rating_series.cat.set_categories(rating_order[::-1], ordered=True)
recovery_durations[rating_column] = rating_series

setting_series = recovery_durations[setting_column]
setting_series = setting_series.astype(str).astype("category")
setting_series = setting_series.cat.set_categories(WIND_SETTING_ORDER, ordered=True)
recovery_durations[setting_column] = setting_series
recovery_durations.sort_values(by=setting_column, ascending=True, inplace=True)

with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    HEIGHT = 2  # inches
    WIDTH = 2.25  # inches
    grid_kwargs = dict(
        sharex=False,
        sharey=True,
        margin_titles=True,
    )

    duration_histogram = sns.displot(
        recovery_durations,
        y=setting_column,
        hue=rating_column,
        hue_order=rating_order[::-1],
        palette={
            rating_order[0]: 'tab:blue',
            rating_order[1]: 'tab:green',
            rating_order[2]: 'tab:orange',
            rating_order[3]: 'tab:red'
        },
        col='Scene', col_order=scene_order,
        row='Attack',
        kind='hist',
        facet_kws=grid_kwargs,
        multiple='stack',
        height=HEIGHT, aspect=WIDTH / HEIGHT,
        legend=True
    )

    # Remove figure legend and draw a new one inside of the first subplot
    duration_histogram.legend.remove()
    duration_histogram.axes.flatten()[0].legend(handles=duration_histogram.legend.legend_handles,
                                                labels=rating_order[::-1],
                                                title=rating_column, loc='center right', ncol=1,
                                                columnspacing=0.5, labelspacing=0.1,)
    # sns.move_legend(duration_histogram, loc="lower center", ncol=4, bbox_to_anchor=(0.525, 0.0), )
    duration_histogram.despine(top=False, right=False)

    # Plot one super ylabel for all y axis
    supylabel_font = duration_histogram.axes.flatten()[0].yaxis.label.get_font()
    duration_histogram.figure.supylabel(WIND_VEL_LABEL, fontproperties=supylabel_font, ha='center')

    duration_histogram.set_titles(
        row_template="{row_name}",
        col_template="{col_name}"
    )

    duration_histogram.tick_params(axis='x', which='major', grid_color='black', grid_alpha=1)
    duration_histogram.figure.subplots_adjust(wspace=0.0)
    duration_histogram.set_xlabels("#Record")

    for row_idx, row in enumerate(duration_histogram.axes):
        for col_idx, ax in enumerate(row):
            ax.set_xlim(0, 50)
            ax.yaxis.label.set_visible(False)
            for spine in ax.spines.values():
                spine.set(alpha=1.0, color='black')

            if col_idx == 0:
                ax.invert_xaxis()
            else:
                ax.tick_params(axis='y', labelleft=False)

    duration_histogram.tight_layout(pad=0.5, rect=[0, 0.0, 1, 1])
    duration_histogram.figure.subplots_adjust(hspace=0.02, wspace=0.0)

    plt.show()
    with PdfPages('Recovery Duration - External Disturbance.pdf') as pdf_pages:
        pdf_pages.savefig(duration_histogram.fig)


# Plot Param Selection in Wind Presence
DETECTOR_TYPES = ['CUSUM', 'EWMA']
BASELINE_NAME_MAP = {
    'virtual_imu_cusum_angvel': 'CUSUM (VIMU)',
    'virtual_imu': 'EMA (VIMU)',
}

SENSOR_NAME_MAP = {
    'gps_position': 'GPS (Position)',
    'gps_velocity': 'GPS (Velocity)',
    'accelerometer': 'Accelerometer',
    'gyroscope': 'Gyroscope',
}

threshold_files = []
result_columns = ['baseline_name', 'test_case', 'scene_name', 'sensor_type', 'detector_type', 'FPR', 'TPR', 'Threshold']
file_folder = "data/evaluation_reports/Wind Disturbance ROC - 05.27"

# Parameter Selection - Wind
PARAM_COLUMN_LABEL = 'param_label'
for det_type in DETECTOR_TYPES:
    fpath = f'{file_folder}/{det_type}_roc_curve_threshold.csv'
    sub_frame = pd.read_csv(fpath, index_col=False)
    if det_type == 'EWMA':
        SOLUTION_MASK = (sub_frame['baseline_name'] == 'virtual_imu')
        GYRO_MASK = (sub_frame['sensor_type'] == 'gyroscope') &\
                    (sub_frame['cap'] == 0.85) & (sub_frame['alpha'] == 0.01)
        ACCEL_MASK = (sub_frame['sensor_type'] == 'accelerometer') &\
                     (sub_frame['cap'] == 1.1) & (sub_frame['alpha'] == 0.01)
        GPS_P_MASK = (sub_frame['sensor_type'] == 'gps_position') &\
                     (sub_frame['cap'] == 0.85) & (sub_frame['alpha'] == 0.01)
        GPS_V_MASK = (sub_frame['sensor_type'] == 'gps_velocity') &\
                     (sub_frame['cap'] == 1.1) & (sub_frame['alpha'] == 0.01)
        SENSOR_MASK = GYRO_MASK | ACCEL_MASK | GPS_P_MASK |GPS_V_MASK
        sub_frame = sub_frame.loc[SOLUTION_MASK & SENSOR_MASK]
    elif det_type == 'CUSUM':
        # SAVIOR VIMU+CUSUM
        SOLUTION_MASK = ((sub_frame['baseline_name'] == 'virtual_imu_cusum') |
                         (sub_frame['baseline_name'] == 'virtual_imu_cusum_angvel'))
        SOLUTION_MASK2 = (sub_frame['baseline_name'] == 'virtual_imu')
        GYRO_MASK = (sub_frame['sensor_type'] == 'gyroscope') & (sub_frame['mean_shift'] == 0.5)
        ACCEL_MASK = (sub_frame['sensor_type'] == 'accelerometer') & (sub_frame['mean_shift'] == 1.0)
        GPS_P_MASK = (sub_frame['sensor_type'] == 'gps_position') & (sub_frame['mean_shift'] == 0.5)
        GPS_V_MASK = (sub_frame['sensor_type'] == 'gps_velocity') & (sub_frame['mean_shift'] == 1.0)
        SENSOR_MASK = (GYRO_MASK | ACCEL_MASK | GPS_P_MASK | GPS_V_MASK)
        sub_frame = sub_frame.loc[SOLUTION_MASK & SENSOR_MASK]
        sub_frame['baseline_name'] = "virtual_imu_cusum_angvel"
        mean_shift = np.round(sub_frame['mean_shift'], decimals=2).astype(str)
    else:
        continue

    threshold_files.append(sub_frame)

fpr_label = 'FPR'
tpr_label = 'TPR'

df = pd.concat(threshold_files)
df = replace_labels(df)
df['Threshold'] = df['Threshold'].replace(np.inf, np.nan).bfill(axis=0)
df['Solution'] = df['Solution'].replace(BASELINE_NAME_MAP)
df['sensor_type'] = df['sensor_type'].replace(SENSOR_NAME_MAP)
df = df.rename(columns=dict(
    TPR=tpr_label,
    FPR=fpr_label
))

with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    HEIGHT = 2.0  # inches
    WIDTH = 2.55  # inches
    grid_kwargs = dict(
        height=HEIGHT, aspect=WIDTH / HEIGHT,
        despine=False,
        sharex=False,
        sharey=True,
    )

    facet_grid = sns.FacetGrid(
        df,
        row='sensor_type', row_order=list(SENSOR_NAME_MAP.values()),
        col='Solution', col_order=list(BASELINE_NAME_MAP.values()),
        margin_titles=True,
        **grid_kwargs
    )

    facet_grid.map_dataframe(
        sns.lineplot,
        x='Threshold', y=fpr_label,
        hue='Setting', hue_order=WIND_SETTING_ORDER
    )

    facet_grid.set_titles(
        row_template="{row_name}",
        col_template="{col_name}"
    )
    for row_idx, row in enumerate(facet_grid.axes):
        for col_idx, ax in enumerate(row):
            ax.axhline(0.05, color="red", linewidth=0.5)
            if col_idx == 0:
                if row_idx == 1:
                    ax.axvline(3.5, color="blue", linewidth=0.5)
                else:
                    ax.axvline(3.0, color="blue", linewidth=0.5)
            elif col_idx == 1:
                if row_idx == 0:
                    ax.axvline(0.45, color="blue", linewidth=0.5)
                elif row_idx == 1:
                    ax.axvline(0.50, color="blue", linewidth=0.5)
                elif row_idx == 2:
                    ax.axvline(0.95, color="blue", linewidth=0.5)
                elif row_idx == 3:
                    ax.axvline(0.235, color="blue", linewidth=0.5)

            x_label = "Normalized Threshold"
            ax.set_xlabel(x_label)
            ax.set_ylim(0.0, 1.01)
            x_min, x_max = ax.get_xlim()
            if x_min < 0.05:
                ax.set_xlim(0.0)

            if col_idx == 0:
                ax.set_xlim(right=4.5)
                new_ticks = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
                tick_labels = new_ticks.astype(str)
                ax.set_xticks(ticks=new_ticks, labels=tick_labels)

            ax.legend(
                loc="upper right",
                handlelength=1.0,
                labelspacing=0.1,
                handletextpad=0.2,
            )

    facet_grid.tight_layout(pad=0.5, rect=[0.0, 0.0, 1.0, 1.0])
    facet_grid.figure.subplots_adjust(wspace=0.05)
    facet_grid.figure.show()
    with PdfPages('Param Selection - External Disturbance.pdf') as pdf_pages:
        pdf_pages.savefig(facet_grid.fig)
