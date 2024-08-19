import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter

from utils import AttackType

TTD_THRESHOLD_S = 15.0
df = pd.read_csv('data/evaluation_reports/Recovery Duration vs. SRR/recovery_reports.csv',
                 index_col=False)

BASELINE_NAME_MAP = {
    'software_sensor': 'SRR',
    'software_sensor_sup': 'SRR',
    'virtual_imu': 'VIMU',
}

MA_GPS_GYRO_NAME = r'$MA_{GPS|Gyro}$'
MA_GPS_BARO_NAME = r'$MA_{GPS|Baro}$'
MA_GPS_BARO_GYRO_NAME = r'$MA_{GPS|Baro|Gyro}$'
MA_MAG_ACC_GYRO_NAME = r'$MA_{Mag|Accel|Gyro}$'

ATTACK_TYPE_MAP = {
    AttackType.GpsPosition: r'$OA_{GPS}$',
    AttackType.Barometer: r'$OA^{2/2}_{Baro}$',
    AttackType.Gyroscope: r'$OA^{3/3}_{Gyro}$',
    AttackType.Gyroscope | AttackType.GpsPosition: MA_GPS_GYRO_NAME,
    AttackType.GpsPosition | AttackType.Barometer: MA_GPS_BARO_NAME,
    AttackType.Gyroscope | AttackType.GpsPosition | AttackType.Barometer: MA_GPS_BARO_GYRO_NAME,
    AttackType.Gyroscope | AttackType.Accelerometer | AttackType.Magnetometer: MA_MAG_ACC_GYRO_NAME,
}

ATTACK_TYPE_ORDER = [
    r'$OA^{2/2}_{Baro}$',
    r'$OA_{GPS}$',
    r'$OA^{3/3}_{Gyro}$',
    MA_GPS_BARO_NAME,
    MA_GPS_GYRO_NAME,
    MA_GPS_BARO_GYRO_NAME,
    MA_MAG_ACC_GYRO_NAME
]

x_axis_column = 'Attack'
y_axis_column = "Duration (s)"
y_axis_label = "Duration (s)"

df = df.rename(columns=dict(
    baseline_name='Solution',
    scene_name='Scene',
    test_case=x_axis_column,
    recovery_duration_s=y_axis_column
))

df['Solution'] = df['Solution'].replace(BASELINE_NAME_MAP)
df['ATK_APPLY_TYPE'] = df['ATK_APPLY_TYPE'].astype(int)
df['Attack'] = df['ATK_APPLY_TYPE'].replace(ATTACK_TYPE_MAP).astype("category")
df['Attack'] = df['Attack'].cat.set_categories(ATTACK_TYPE_ORDER, ordered=True)
df.loc[df.loc[:, 'tp_ttd_s'] > TTD_THRESHOLD_S, y_axis_column] = np.nan

result_0 = df.dropna(subset=[y_axis_column])
result_1 = result_0.loc[:, ['Solution', 'Scene', 'false_positive', 'fp_ttd_s', x_axis_column, y_axis_column]]
result_1.loc[:, y_axis_column] = result_1.loc[:, y_axis_column].clip(0., 345.)

with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    HEIGHT = 1.25  # inches
    WIDTH = 1.4  # inches
    grid_kwargs = dict(
        sharex=True,
        sharey=True,
        margin_titles=True
    )

    plot = sns.displot(
        result_1,
        y=y_axis_column,
        log_scale=True,
        hue='Solution',
        hue_order=[BASELINE_NAME_MAP['software_sensor'], BASELINE_NAME_MAP['virtual_imu']],
        palette={
            BASELINE_NAME_MAP['software_sensor']: 'tab:orange',
            BASELINE_NAME_MAP['virtual_imu']: 'tab:purple'
        },
        col='Attack', col_order=ATTACK_TYPE_ORDER,
        row='Scene', row_order=['Hovering', 'Moving'],
        kind='hist',
        facet_kws=grid_kwargs,
        element="step",
        height=HEIGHT, aspect=WIDTH / HEIGHT,
    )

    plot.set_xlabels("#Record")

    # Remove figure legend and draw a new one inside of the first subplot
    plot.legend.remove()
    plot.axes[1, 0].legend(handles=plot.legend.legend_handles,
                           labels=[BASELINE_NAME_MAP['software_sensor'], BASELINE_NAME_MAP['virtual_imu']],
                           title='Solution',
                           loc="lower left", ncol=1, framealpha=1.0,
                           columnspacing=0.5, labelspacing=0.1, )

    plot.despine(top=False, right=False)
    plot.set_titles(
        row_template="{row_name}",
        col_template="{col_name}"
    )

    sup_label_font = plot.axes.flatten()[0].yaxis.label.get_font()
    plot.figure.supylabel(y_axis_label, x=0.015, y=0.55, fontproperties=sup_label_font, ha='center')

    for row_idx, row in enumerate(plot.axes):
        for col_idx, ax in enumerate(row):
            x_ticks = np.arange(0, 51, 10)
            x_ticklabels = x_ticks.astype(str)
            ax.set_xticks(x_ticks, x_ticklabels)
            ax.set_xlim(0, 50)

            ax.yaxis.label.set_visible(False)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.tick_params(axis='y', which='major', grid_color='black', grid_alpha=1)

            if col_idx == 0:
                ax.set_yticks([3, 10, 100, 300], [3, 10, 100, 300])
                ax.get_yticklabels()[0].set_color("red")
                ax.get_yticklabels()[-1].set_color("green")

            ax.axhline(300, color="green", linewidth=1)
            ax.axhline(3, color="red", linewidth=1)

    # plot.tight_layout(pad=0.5, rect=[0, 0.125, 1, 1])
    plot.tight_layout(pad=0.5, rect=[0.0, 0.0, 1.0, 1.0])
    plot.figure.subplots_adjust(wspace=0.15, hspace=0.05)
plot.figure.show()

with PdfPages('Recovery Time Histogram.pdf') as pdf_pages:
    pdf_pages.savefig(plot.fig)
