import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DETECTOR_TYPES = ['CUSUM', 'EWMA']
BASELINE_NAME_MAP = {
    'savior': 'CUSUM (SAVIOR)',
    'virtual_imu_cusum_angvel': 'CUSUM (VIMU)',
    'virtual_imu': 'EMA (VIMU)',
}

threshold_files = []
result_columns = ['baseline_name', 'test_case', 'scene_name', 'sensor_type', 'detector_type', 'FPR', 'TPR', 'Threshold']
file_folder = "data/evaluation_reports/Maneuver Param Selection - 08.15"

# General Parameter Selection
PARAM_COLUMN_LABEL = 'param_label'
for det_type in DETECTOR_TYPES:
    fpath = f'{file_folder}/{det_type}_roc_curve_threshold.csv'
    sub_frame = pd.read_csv(fpath, index_col=False)
    GYRO_MASK = (sub_frame['sensor_type'] == 'gyroscope')
    sub_frame = sub_frame.loc[GYRO_MASK]
    if det_type == 'EWMA':
        # gyro_alpha = 0.01
        # gyro_cap = 0.85
        SOLUTION_MASK = (sub_frame['baseline_name'] == 'virtual_imu')
        sub_frame = sub_frame.loc[SOLUTION_MASK]
        sub_frame = sub_frame.loc[sub_frame['cap'] <= 1.0]
        alpha = np.round(sub_frame['alpha'], decimals=2).astype(str)
        cap = np.round(sub_frame['cap'], decimals=2).astype(str)
        # Add tail spaces to ensure the text is within the legend box
        sub_frame[PARAM_COLUMN_LABEL] = r'$\lambda|R$=' + alpha + '|' + cap  # + '    '
    elif det_type == 'CUSUM':
        # SAVIOR VIMU+CUSUM
        # gps_param_value = 0.5
        # gyro_param_value = 0.5
        SOLUTION_MASK = ((sub_frame['baseline_name'] == 'savior') |
                         (sub_frame['baseline_name'] == 'virtual_imu_cusum') |
                         (sub_frame['baseline_name'] == 'virtual_imu_cusum_angvel'))
        sub_frame = sub_frame.loc[SOLUTION_MASK]
        sub_frame = sub_frame.loc[(sub_frame['mean_shift'] <= 0.8)]
        mean_shift = np.round(sub_frame['mean_shift'], decimals=2).astype(str)
        # Add tail spaces to ensure the text is within the legend box
        sub_frame[PARAM_COLUMN_LABEL] = r'$b$=' + mean_shift  # + '  '
    else:
        raise RuntimeError(f"Unknown Detector Type: {det_type}!")

    threshold_files.append(sub_frame)

fpr_label = 'FPR'
tpr_label = 'TPR'

df = pd.concat(threshold_files)
df['Threshold'] = df['Threshold'].replace(np.inf, np.nan)
df = df.rename(columns=dict(
    baseline_name='Solution',
    scene_name='Scene',
    test_case='Attack',
    TPR=tpr_label,
    FPR=fpr_label
))

df['Solution'] = df['Solution'].replace(BASELINE_NAME_MAP)

with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    HEIGHT = 1.8  # inches
    WIDTH = 5  # inches
    grid_kwargs = dict(
        height=HEIGHT, aspect=WIDTH / HEIGHT,
        despine=False,
        sharex=False,
        sharey=True,
    )

    facet_grid = sns.FacetGrid(
        df, row='Solution', row_order=list(BASELINE_NAME_MAP.values()),
        margin_titles=True,
        **grid_kwargs
    )

    facet_grid.map_dataframe(
        sns.lineplot,
        x='Threshold', y=fpr_label, hue='param_label',
    )

    facet_grid.set_titles(
        row_template="{row_name}",
        col_template="{col_name}"
    )

    for k, ax in facet_grid.axes_dict.items():
        x_label = "Normalized Threshold"
        ax.set_xlabel(x_label)
        ax.set_ylim(0.0, 1.0)
        x_min, x_max = ax.get_xlim()
        if x_min < 0.05:
            ax.set_xlim(0.0)

        x_max_new = None
        if 'L2TW' in k:
            x_max_new = 2.25
        # elif 'L1TW' in k:
        #     x_max_new = 100.0
        elif 'CUSUM' in k:
            x_max_new = 4.25
        elif 'EWMA' in k:
            x_max_new = 0.65
        ax.set_xlim(xmax=x_max_new)

        ax.legend(
            loc="upper right",
            # borderpad=1.0,
            ncols=3,
            columnspacing=0.5,
            handlelength=1.0,
            labelspacing=0.1,
            handletextpad=0.2,
        )

    facet_grid.tight_layout(pad=0.5, rect=[0.0, 0.0, 1.0, 1.0])
    facet_grid.figure.subplots_adjust(
        wspace=0.02,
        hspace=0.4
    )

    facet_grid.figure.show()
    with PdfPages('Param Selection Maneuver.pdf') as pdf_pages:
        pdf_pages.savefig(facet_grid.fig)
