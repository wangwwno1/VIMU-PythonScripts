import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils import AttackType

BASELINE_NAME_MAP = {
    'control_invariant': 'CI',
    'software_sensor_sup': 'SRR',
    'savior': 'SAVIOR',
    'virtual_imu_cusum': 'VIMU-CS',
    'virtual_imu': 'VIMU',
}

ATTACK_NAME_MAP = {
    'gps_overt_attack': r'$OA_{GPS}$',
    'tmr_test_default': r'$OA^{2/3}_{Gyro}$',
    # 'tmr_test_icm20602': r'$OA^{2/3}_{ICM20602}$',
    # 'tmr_test_icm20689': r'$OA^{2/3}_{ICM20689}$',
    'gyro_overt_attack_default': r'$OA^{3/3}_{Gyro}$',
    'gyro_overt_attack_icm20602': r'$OA^{3/3}_{ICM20602}$',
    'gyro_overt_attack_icm20689': r'$OA^{3/3}_{ICM20689}$',
}

GPS_ATTACK_ORDER = [r'$OA_{GPS}$']
GYRO_ATTACK_ORDER = [r'$OA^{2/3}_{Gyro}$', r'$OA^{3/3}_{Gyro}$', r'$OA^{3/3}_{ICM20602}$', r'$OA^{3/3}_{ICM20689}$']

SENSOR_NAME_MAP = {
    'gps': 'GPS',
    'barometer': 'BARO',
    'magnetometer': 'MAG',
    'accelerometer': 'ACC',
    'gyroscope': 'GYR',
}


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot_table(index=args[1], columns=args[0], values=args[2], sort=False)
    sns.heatmap(d, **kwargs)


def draw_grid_map(facet_grid, x_col_name, y_col_name, x_grid_label, y_grid_label, value_column, cbar_ax, cmap):
    facet_grid.map_dataframe(
        draw_heatmap, x_col_name, y_col_name, value_column,
        vmin=0., vmax=1.,
        center=0.5,
        annot=True,
        annot_kws=dict(),
        linewidth=0.5,
        cmap=cmap,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"},
        fmt='.2f',
    )

    facet_grid.set_titles(
        row_template="{row_name}",
        col_template="{col_name}"
    )

    facet_grid.set_xlabels(x_grid_label)
    # facet_grid.set_ylabels(y_grid_label)

    max_row_idx = len(facet_grid.axes) - 1
    for row_idx, row in enumerate(facet_grid.axes):
        for col_idx, ax in enumerate(np.atleast_1d(row)):
            if col_idx == 0:
                ax.set_ylabel(None)

            plt.setp(ax.xaxis.get_ticklabels(), ha="center", rotation=0)
            ax.tick_params(axis="x", labelbottom=row_idx == max_row_idx)

            plt.setp(ax.yaxis.get_ticklabels(), ha="center", va="center", rotation=0)
            ax.tick_params(axis="y", direction="out", pad=20)


def get_attack_deviation_(dataframe: pd.DataFrame):
    dataframe.insert(1, DEVIATION_LABEL, np.nan)

    gps_attack_mask = dataframe.loc[:, 'ATK_APPLY_TYPE'] == AttackType.GpsPosition
    dataframe.loc[gps_attack_mask, DEVIATION_LABEL] = dataframe.loc[gps_attack_mask, "ATK_GPS_P_IV"]

    gyro_attack_mask = dataframe.loc[:, 'ATK_APPLY_TYPE'] == AttackType.Gyroscope
    dataframe.loc[gyro_attack_mask, DEVIATION_LABEL] = dataframe.loc[gyro_attack_mask, "ATK_GYR_AMP"]

    dataframe.loc[:, DEVIATION_LABEL] = np.round(dataframe.loc[:, DEVIATION_LABEL].values, decimals=2)
    return dataframe


def mark_positive_sample_(dataframe: pd.DataFrame, detect_time_threshold_ms: float = 0.0):
    if detect_time_threshold_ms > 0.:
        # Do not count TP/FP that the time exceed threshold
        is_exceeded = dataframe['time_to_detection_ms'] > detect_time_threshold_ms
        dataframe.loc[is_exceeded, 'time_to_detection_ms'] = np.nan

    is_attacked = dataframe.loc[:, 'is_attacked'].values
    has_alarm = dataframe.loc[:, 'time_to_detection_ms'].notna().values
    dataframe['true_positive'] = is_attacked & has_alarm
    dataframe['false_positive'] = (~is_attacked) & has_alarm

    return dataframe


def sort_entries_(dataframe: pd.DataFrame):
    dataframe['Solution'] = dataframe['Solution'].astype("category")
    dataframe['Scene'] = dataframe['Scene'].astype("category")

    dataframe['Solution'] = dataframe['Solution'].cat.set_categories(list(BASELINE_NAME_MAP.values()))
    dataframe['Scene'] = dataframe['Scene'].cat.set_categories(['Hovering', 'Moving'])

    dataframe.sort_values(['Scene', 'Solution', 'deviation'], axis=0, inplace=True)
    # Align the scene text length
    dataframe['Scene'] = dataframe['Scene'].astype(str)
    dataframe['Solution'] = dataframe['Solution'].astype(str)

    return dataframe


def replace_names(dataframe: pd.DataFrame):
    dataframe = dataframe.rename(columns=dict(
        baseline_name='Solution',
        scene_name='Scene',
        test_case='Attack',
        sensor='Sensor'
    ))
    dataframe['Solution'] = dataframe['Solution'].replace(BASELINE_NAME_MAP)
    dataframe['Attack'] = dataframe['Attack'].replace(ATTACK_NAME_MAP)
    dataframe['Sensor'] = dataframe['Sensor'].replace(SENSOR_NAME_MAP)
    return dataframe


def preprocess_data(dataframe: pd.DataFrame, detect_time_threshold_ms: float = 0.0):
    result_0 = mark_positive_sample_(dataframe.copy(), detect_time_threshold_ms)
    result_0 = get_attack_deviation_(result_0)

    group_keys = ['Solution', 'Attack', 'Scene', 'deviation', 'Sensor', 'is_attacked']
    result_1 = result_0.loc[:, group_keys + ['true_positive', 'false_positive']]
    result_1['instance_counts'] = 1
    output = result_1.groupby(group_keys, group_keys=True).sum().reset_index()
    sort_entries_(output)

    return output


def get_false_positive_rate(dataframe: pd.DataFrame, detect_time_threshold_ms: float = 0.0, round_digits: int = 2):
    result_2 = preprocess_data(dataframe, detect_time_threshold_ms)
    output = result_2.loc[~result_2['is_attacked']].copy()
    fpr = np.round(output['false_positive'] / output['instance_counts'], round_digits)
    output['fpr_pct'] = fpr

    return output


def get_true_positive_rate(dataframe: pd.DataFrame, detect_time_threshold_ms: float = 0.0, round_digits: int = 2):
    result_2 = preprocess_data(dataframe, detect_time_threshold_ms)
    output = result_2.loc[result_2['is_attacked']].copy()
    tpr = np.round(output['true_positive'] / output['instance_counts'], round_digits)
    output['tpr_pct'] = tpr

    return output


if __name__ == "__main__":
    DEVIATION_LABEL = 'deviation'
    SEC_TO_MS = 1e3
    GPS_TPR_TTD_THRESHOLD_MS = 20.0 * SEC_TO_MS

    csv_path = 'data/evaluation_reports/GPS Overt Attacks TPR - 01.22/full_detection_reports.csv'
    df = pd.read_csv(csv_path, index_col=False)
    df = replace_names(df)

    tp_data = get_true_positive_rate(df, GPS_TPR_TTD_THRESHOLD_MS)
    with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
        x_axis_column = DEVIATION_LABEL
        y_axis_column = 'Solution'
        x_axis_label = None
        y_axis_label = "Solution"
        cmap = sns.color_palette("blend:#FF9999,#FFE699,#C6E0B4", as_cmap=True)

        HEIGHT = 2.75  # inches
        WIDTH = 2.7  # inches
        grid_kwargs = dict(
            height=HEIGHT, aspect=WIDTH / HEIGHT,
            despine=False,
            sharex=False,
            sharey=True,
            margin_titles=True
        )
        # Plot TPR graph
        tpr_graph = sns.FacetGrid(
            tp_data,
            row='Attack', row_order=GPS_ATTACK_ORDER,
            col='Scene', col_order=['Hovering', 'Moving'],
            **grid_kwargs
        )

        # cbar_ax = tpr_graph.fig.add_axes([.95, .1, .01, .8])
        cbar_ax = tpr_graph.fig.add_axes([0.125, 0.08, .825, .025])
        draw_grid_map(tpr_graph,
                      x_axis_column, y_axis_column, x_axis_label, y_axis_label,
                      'tpr_pct', cbar_ax, cmap)
        max_row_idx = len(tpr_graph.axes) - 1
        for row_idx, row in enumerate(tpr_graph.axes):
            for col_idx, ax in enumerate(np.atleast_1d(row)):
                # Gyro Attack
                x_label = "GPS Deviation (m)"
                ax.set_xlabel(None)
                if row_idx == max_row_idx:
                    ax.set_xlabel(x_label)
        tpr_graph.tight_layout(pad=0.25, rect=[0, 0.1, 1.0, 1.0])
        tpr_graph.figure.subplots_adjust(wspace=0.0, hspace=0.01)
        tpr_graph.figure.show()

        with PdfPages('TPR_heatmap Overt GPS Attack - 1x2.pdf') as pdf_pages:
            pdf_pages.savefig(tpr_graph.fig)

