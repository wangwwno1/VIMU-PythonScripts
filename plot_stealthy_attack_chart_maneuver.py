import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter

from utils import AttackType, StealthyType

BASELINE_NAME_MAP = {
    'savior': 'SAVIOR',
    'virtual_imu': 'VIMU',
}

PALATTE = {
    "CI": 'tab:blue',
    'SRR': 'tab:orange',
    "SAVIOR": 'tab:green',
    'VIMU-CS': 'tab:red',
    'VIMU': 'tab:purple'
}

# BASELINE_NAME_MAP.pop('virtual_imu_cusum')

ATTACK_NAME_MAP = {
    'gyro_stealthy_attack_default': r'$SA^{3/3}_{Gyro}$'
}


def fill_deviation(df, label_name):
    df.insert(1, label_name, np.nan)
    gps_attack_mask = df.loc[:, 'ATK_APPLY_TYPE'] == AttackType.GpsPosition
    df.loc[gps_attack_mask, label_name] = df.loc[gps_attack_mask, "ATK_GPS_P_IV"]
    gyro_attack_mask = df.loc[:, 'ATK_APPLY_TYPE'] == AttackType.Gyroscope
    df.loc[gyro_attack_mask, label_name] = df.loc[gyro_attack_mask, "ATK_GYR_AMP"]
    df.loc[:, label_name] = np.round(df.loc[:, label_name], decimals=3)

    mask = df.loc[:, label_name] > 0.025
    df.loc[mask, label_name] = np.round(df.loc[mask, label_name], decimals=2)


if __name__ == "__main__":
    GYRO_TTD_THRESHOLD_S = 1.0
    GPS_TTD_THRESHOLD_S = 20.0
    SEC_TO_MS = 1e3

    csv_path = 'data/evaluation_reports/Moving vs Maneuver Gyro Stealthy Attack - 08.15/full_detection_reports.csv'

    df = pd.read_csv(csv_path, index_col=False)
    TIME_TO_DETECTION_LABEL = "TTD (ms)"

    df = df.rename(columns=dict(
        baseline_name='Solution',
        scene_name='Scene',
        test_case='Attack',
        time_to_detection_ms=TIME_TO_DETECTION_LABEL
    ))

    df['Solution'] = df['Solution'].replace(BASELINE_NAME_MAP)
    df['Attack'] = df['Attack'].replace(ATTACK_NAME_MAP)

    # Select attacked sensor instance only
    df = df.loc[df.loc[:, 'is_attacked']]
    x_axis_column = 'Deviation'
    y_axis_column = TIME_TO_DETECTION_LABEL
    y_axis_label = TIME_TO_DETECTION_LABEL
    fill_deviation(df, x_axis_column)

    result_0 = df.copy()
    result_1 = result_0.loc[:, ['Solution', 'Attack', 'Scene', x_axis_column, y_axis_column]]
    for attack_type in (AttackType.GpsPosition, AttackType.Gyroscope):
        if attack_type == AttackType.Gyroscope:
            ttd_threshold = GYRO_TTD_THRESHOLD_S if attack_type == AttackType.Gyroscope else GPS_TTD_THRESHOLD_S
            not_applicable_ttd = 2.0
        else:
            ttd_threshold = GPS_TTD_THRESHOLD_S
            not_applicable_ttd = 30.0

        mask = df['ATK_APPLY_TYPE'] == attack_type
        sub_frame = result_1.loc[mask, y_axis_column]

        # TTD greater than threshold is equal to False Negative
        sub_frame = sub_frame.where(sub_frame < ttd_threshold * SEC_TO_MS)
        sub_frame.fillna(np.inf, inplace=True)

        # Minus 4 because the autopilot occasionally count one more sample in ttd calculation.
        sub_frame = np.maximum(1.1, sub_frame - 4)
        sub_frame = np.minimum(not_applicable_ttd * SEC_TO_MS, sub_frame)

        result_1.loc[mask, y_axis_column] = sub_frame

    with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
        # HEIGHT = 3.25  # inches
        # HEIGHT = 2.75  # inches
        HEIGHT = 2.5  # inches
        WIDTH = 2  # inches

        plot = sns.catplot(
            result_1,
            x=x_axis_column, y=y_axis_column,
            hue='Solution',
            hue_order=list(BASELINE_NAME_MAP.values()),
            palette=PALATTE,
            row='Attack', row_order=list(ATTACK_NAME_MAP.values()),
            col='Scene', col_order=['Moving', 'Maneuver'],
            kind="strip",
            sharey=False,
            sharex=False,
            dodge=True,  # Separate the result by class
            alpha=0.25,
            margin_titles=True,
            height=HEIGHT,
            aspect=WIDTH / HEIGHT,
            orient='v',
        )

        plot.legend.set_title(None)  # Remove title
        # Set legend symbol opacity
        for lh in plot.legend.legend_handles:
            lh.set_alpha(1)

        sns.move_legend(
            plot,
            loc="lower center",
            ncol=10,
            # frameon=True, edgecolor="black"
        )
        plot.despine(top=False, right=False)
        plot.set_titles(
            row_template="{row_name}",
            col_template="{col_name}"
        )

        plot.set_ylabels(y_axis_label)
        GPS_ROW_IDX = 0
        NOT_APPLICABLE_LABEL = 'N/A'
        for row_idx, row in enumerate(plot.axes):
            for col_idx, ax in enumerate(row):
                ax.tick_params(axis='y', which='major', grid_color='black', grid_alpha=1)
                ax.tick_params(axis='y', which='minor', gridOn=True)  # Enable minor grid line

                # Plot vertical line to distinguish each category
                num_categories = len(ax.get_xticks())
                min_xlim, max_xlim = ax.get_xbound()
                width_per_categories = (max_xlim - min_xlim) / num_categories
                for cat_idx in range(num_categories):
                    ax.axvline(min_xlim + (cat_idx+1) * width_per_categories, color='black', linewidth=0.5)
                ax.set_xbound(min_xlim, max_xlim)

                ax.set_yscale('log')

                ax.xaxis.label.set_text(r"Absolute Residual (rad/s)")
                ax.set_ylim(0.8, GYRO_TTD_THRESHOLD_S * 3 * SEC_TO_MS)

                ax.yaxis.set_major_formatter(ScalarFormatter())
                ticks = np.asarray([1, 10, 100, 500, 1000, 2000])
                ax.set_yticks(ticks, [0, 10, '100', '500', '1000', NOT_APPLICABLE_LABEL])
                ax.axhline(1000, color="red", linewidth=1)
                ax.axhline(500, color="green", linewidth=1)
                if col_idx == 0:
                    ax.get_yticklabels()[-3].set_color("green")
                    ax.get_yticklabels()[-2].set_color("red")

                if col_idx != 0:
                    ax.set(yticklabels=[])

    plot.tight_layout(pad=0.5, rect=[0, 0.1, 1.0, 1.0])
    plot.figure.subplots_adjust(wspace=0.01)
    plot.figure.show()

    with PdfPages('Stealthy Time to Detect - Maneuver.pdf') as pdf_pages:
        pdf_pages.savefig(plot.fig)
