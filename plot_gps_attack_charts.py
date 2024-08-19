import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plot_tpr_fpr_heatmap import get_true_positive_rate

DETECTOR_TYPES = ['L1TW', 'L2TW', 'CUSUM', 'EWMA']
BASELINE_NAME_MAP = {
    'control_invariant': 'CI',
    'software_sensor_sup': 'SRR',
    'savior': 'SAVIOR',
    'virtual_imu_cusum': 'VIMU-CS',
    'virtual_imu': 'VIMU',
}

ATTACK_NAME_MAP = {
    'gps_joint_attack': r'$OA_{GPS-PV}$',
}

threshold_files = []
result_columns = ['baseline_name', 'test_case', 'scene_name', 'sensor_type', 'detector_type', 'FPR', 'TPR', 'Threshold']
file_folder = "data/evaluation_reports/GPS Joint Attack"

tpr_label = 'TPR'

SEC_TO_MS = 1e3
TTD_THRESHOLD_MS = 10.0 * SEC_TO_MS

df = pd.read_csv(f"{file_folder}/full_detection_reports.csv", index_col=False)
df = df.rename(columns=dict(
    baseline_name='Solution',
    scene_name='Scene',
    test_case='Attack',
    sensor='Sensor',
))
df['Solution'] = df['Solution'].replace(BASELINE_NAME_MAP)
df['Attack'] = df['Attack'].replace(ATTACK_NAME_MAP)

# This gps attack is only applied to hovering case
select_condition = (df['Attack'] == ATTACK_NAME_MAP['gps_joint_attack'])\
                   & (df['Scene'] == 'Hovering') & (df['Sensor'] == 'gps')
df = df.loc[select_condition]

with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    fig, ttd_ax = plt.subplots(1, 1, figsize=(5, 1.75))

    # Select attacked sensor instance only
    time_to_detection = df.loc[df.loc[:, 'is_attacked']]
    time_to_detection = time_to_detection.rename(columns=dict(time_to_detection_ms="TTD (ms)"))
    x_axis_column = 'Attack'
    y_axis_column = 'TTD (ms)'
    y_axis_label = "TTD (ms)"

    time_to_detection = time_to_detection.loc[:, list({'Solution', 'Attack', 'Scene', x_axis_column, y_axis_column})]
    # TTD greater than threshold is equal to False Negative
    time_to_detection[y_axis_column] = time_to_detection[y_axis_column].where(time_to_detection[y_axis_column] < TTD_THRESHOLD_MS)
    time_to_detection.loc[:, y_axis_column] = time_to_detection.loc[:, y_axis_column].fillna(np.inf)

    # Minus 4 because the autopilot occasionally count one more sample in ttd calculation.
    time_to_detection.loc[:, y_axis_column] = np.maximum(1.1, time_to_detection.loc[:, y_axis_column] - 4)
    time_to_detection.loc[:, y_axis_column] = np.minimum(22000, time_to_detection.loc[:, y_axis_column])

    sns.stripplot(
        time_to_detection,
        x='TTD (ms)',  # TTD
        hue='Solution',
        hue_order=list(BASELINE_NAME_MAP.values()),
        dodge=True,  # Separate the result by class
        alpha=0.25,
        orient='h',
        ax=ttd_ax
    )

    # Set legend symbol opacity
    for lh in ttd_ax.get_legend().legend_handles:
        lh.set_alpha(1)

    fig.legend(
        handles=ttd_ax.get_legend().legend_handles,
        labels=list(BASELINE_NAME_MAP.values()),
        loc='lower center',
        ncol=5,
        frameon=False,
        columnspacing=1.0
    )

    ttd_ax.get_legend().remove()

    ttd_ax.axvline(20000, color="red", linewidth=1)
    ttd_ax.xaxis.set_label_position('top')

    tick_labels = ttd_ax.get_xticklabels()
    tick_labels[-2] = 'N/A'
    ttd_ax.set_xticks(ticks=ttd_ax.get_xticks(), labels=tick_labels)
    ttd_ax.get_xticklabels()[-3].set_color("red")

    ttd_ax.set_xlim(7250, 22500)

    plt.tight_layout(rect=(0.0, 0.125, 1.0, 1.0))
    plt.show()

with PdfPages('Real GPS Attack TTD.pdf') as pdf_pages:
    pdf_pages.savefig(fig)
