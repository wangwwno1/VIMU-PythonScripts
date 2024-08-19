import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter

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
}


def fill_deviation(df, label_name):
    df.insert(1, label_name, np.nan)
    gps_attack_mask = df.loc[:, 'ATK_APPLY_TYPE'] == AttackType.GpsPosition
    df.loc[gps_attack_mask, label_name] = df.loc[gps_attack_mask, "ATK_GPS_P_IV"]
    gyro_attack_mask = df.loc[:, 'ATK_APPLY_TYPE'] == AttackType.Gyroscope
    df.loc[gyro_attack_mask, label_name] = df.loc[gyro_attack_mask, "ATK_GYR_AMP"]
    df.loc[:, label_name] = np.round(df.loc[:, label_name], decimals=2)


if __name__ == "__main__":
    GYRO_TTD_THRESHOLD_S = 1.0
    GPS_TTD_THRESHOLD_S = 20.0
    SEC_TO_MS = 1e3

    CSV_PATH = 'data/evaluation_reports/GPS Overt Attacks TPR - 01.22/full_detection_reports.csv'
    FILENAME = 'Time to Detect - GPS Overt Attack.pdf'

    df = pd.read_csv(CSV_PATH, index_col=False)
    df = df.rename(columns=dict(
        baseline_name='Solution',
        scene_name='Scene',
        test_case='Attack',
        time_to_detection_ms="TTD (ms)"
    ))
    df['Solution'] = df['Solution'].replace(BASELINE_NAME_MAP)
    df['Attack'] = df['Attack'].replace(ATTACK_NAME_MAP)
    # Select attacked sensor instance only
    df = df.loc[df.loc[:, 'is_attacked']]
    x_axis_column = 'deviation'
    y_axis_column = 'TTD (ms)'
    y_axis_label = "TTD (ms)"
    fill_deviation(df, x_axis_column)
    result_0 = df.copy()
    result_1 = result_0.loc[:, ['Solution', 'Attack', 'Scene', x_axis_column, y_axis_column]]

    # TTD greater than threshold is equal to False Negative
    result_1[y_axis_column] = result_1[y_axis_column].where(result_1[y_axis_column] < GPS_TTD_THRESHOLD_S * SEC_TO_MS)
    result_1.loc[:, y_axis_column] = result_1.loc[:, y_axis_column].fillna(np.inf)

    # Minus 4 because the autopilot occasionally count one more sample in ttd calculation.
    result_1.loc[:, y_axis_column] = np.maximum(1.1, result_1.loc[:, y_axis_column] - 4)
    result_1.loc[:, y_axis_column] = np.minimum(40000, result_1.loc[:, y_axis_column])
    # result_1[y_axis_column] /= SCALE_SEC_TO_MS

    with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
        HEIGHT = 3  # inches
        WIDTH = 3.75  # inches

        plot = sns.catplot(
            result_1,
            x=x_axis_column, y=y_axis_column,
            hue='Solution',
            hue_order=list(BASELINE_NAME_MAP.values()),
            col='Attack', col_order=list(ATTACK_NAME_MAP.values()),
            row='Scene', row_order=['Hovering', 'Moving'],
            kind="strip",
            sharey=False,
            sharex=False,
            dodge=True,  # Separate the result by class
            margin_titles=True,
            alpha=0.25,
            height=HEIGHT,
            aspect=WIDTH / HEIGHT,
            orient='v'
        )

        # plot.despine(top=True, right=True, left=True)
        plot.legend.set_title(None)  # Remove title
        # Set legend symbol opacity
        for lh in plot.legend.legend_handles:
            lh.set_alpha(1)

        sns.move_legend(
            plot,
            loc="lower center",
            ncol=3,
        )

        plot.set_titles(
            row_template="{row_name}",
            col_template="{col_name}"
        )

        plot.set_ylabels(y_axis_label)
        for row_idx, row in enumerate(plot.axes):
            for col_idx, ax in enumerate(row):
                ax.tick_params(axis='y', which='major', grid_color='black', grid_alpha=1)
                ax.tick_params(axis='y', which='minor', gridOn=True)  # Enable minor grid line

                if row_idx < len(plot.axes) - 1:
                    # Remove xticks for space
                    ax.set(xlabel=None, xticklabels=[])
                    ax.tick_params(axis='x', bottom=False)
                    pass
                else:
                    x_label = r"GPS Deviation (m)"
                    ax.xaxis.label.set_text(x_label)

                # Plot vertical line to distinguish each category
                num_categories = len(ax.get_xticks())
                min_xlim, max_xlim = ax.get_xbound()
                width_per_categories = (max_xlim - min_xlim) / num_categories
                for cat_idx in range(num_categories):
                    ax.axvline(min_xlim + (cat_idx+1) * width_per_categories, color='black', linewidth=0.5)
                ax.set_xbound(min_xlim, max_xlim)

                for s in ax.spines.values():
                    s.set_visible(True)

                ax.set_yscale('log')
                ax.set_ylim(1, GPS_TTD_THRESHOLD_S * 3 * SEC_TO_MS)  # in millisecond (ms)
                ax.axhline(GPS_TTD_THRESHOLD_S * SEC_TO_MS, color="red", linewidth=1)

                ticks = np.asarray([1.1, 10, 100, 1000, 10000, 20000, 40000])
                ticks_label = np.array(ticks).astype(int).astype(str)
                ticks_label[0] = '0'
                ticks_label[-1] = 'N/A'

                ax.yaxis.set_major_formatter(ScalarFormatter())
                # ax.get_yticklabels()[3].set_color("green")
                ax.get_yticklabels()[-3].set_color("red")
                ax.set_yticks(ticks, labels=ticks_label)
                if col_idx > 0:
                    ax.set(yticklabels=[])

        plot.tight_layout(pad=0.5, rect=[0.0, 0.075, 1.00, 1.00])
        plot.figure.subplots_adjust(wspace=0.005, hspace=0.025)

    plot.figure.show()
    with PdfPages(FILENAME) as pdf_pages:
        pdf_pages.savefig(plot.fig)
