import numpy as np
import pandas as pd
import pyulog
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils.ulog_extractors import extract_ulog_dict, find_attack_timestamp, extract_recovery_status
from utils import quaternion_to_euler
import seaborn as sns


def clip(dataframe, attack_start):
    start = attack_start - CLIP_START_BEFORE_ATK_S * SECOND_TO_MILLISECONDS
    stop = attack_start + CLIP_STOP_AFTER_ATK_S * SECOND_TO_MILLISECONDS
    full_timestamp = dataframe.index
    mask = (full_timestamp >= start) & (full_timestamp <= stop)
    dataframe = dataframe.loc[mask, :]
    dataframe.index = (dataframe.index.astype(int) - attack_start) / SECOND_TO_MILLISECONDS
    return dataframe


def to_euler_angle(dataframe):
    result = pd.DataFrame(
        quaternion_to_euler(dataframe.values) / np.pi * 180,
        index=dataframe.index,
        columns=EULER_COLUMN
    )
    result.reset_index(names=TIME_LABEL, inplace=True)
    return result


def process_attitude(log_path, solution_name):
    ulog_obj = pyulog.ULog(log_path)
    ulog_dict = extract_ulog_dict(ulog_obj)
    attack_info = find_attack_timestamp(ulog_obj)
    attack_start, attack_type, attack_end = attack_info[0]

    estimate, groundtruth = extract_recovery_status(ulog_dict)

    estimate_quaternions = clip(estimate.attitude, attack_start)
    estimate_euler_angles = to_euler_angle(estimate_quaternions)
    estimate_euler_angles = estimate_euler_angles.melt(id_vars=TIME_LABEL,
                                                       var_name='AngleAxis', value_name='EulerAngle')
    estimate_euler_angles['DataType'] = solution_name

    groundtruth_quaternions = clip(groundtruth.attitude, attack_start)
    groundtruth_euler_angles = to_euler_angle(groundtruth_quaternions)
    groundtruth_euler_angles = groundtruth_euler_angles.melt(id_vars=TIME_LABEL,
                                                             var_name='AngleAxis', value_name='EulerAngle')
    groundtruth_euler_angles['DataType'] = HUE_ORDER[-1]

    estimate_groundtruth = pd.concat([estimate_euler_angles, groundtruth_euler_angles])
    estimate_groundtruth['Solution'] = solution_name

    return estimate_groundtruth


if __name__ == '__main__':
    CLIP_START_BEFORE_ATK_S = 2
    CLIP_STOP_AFTER_ATK_S = 1
    SECOND_TO_MILLISECONDS = 1e6

    SRR_W_COMP_LOG = "data/evaluation_reports/sup_compensation/logs/SRR with ACC-Compensation 2023-05-12_102852.ulg"
    SRR_WO_COMP_LOG = "data/evaluation_reports/sup_compensation/logs/SRR without ACC-Compensation 2023-03-07_023209.ulg"
    VIMU_WO_COMP_LOG = "data/evaluation_reports/sup_compensation/logs/VIMU without ACC-Compensation 2023-05-10_123208.ulg"

    SOLUTION_ORDER = ['SRR w/o SC', 'SRR', 'VIMU']
    SOLUTION_LOG_DICT = dict(zip(SOLUTION_ORDER, [SRR_WO_COMP_LOG, SRR_W_COMP_LOG, VIMU_WO_COMP_LOG]))
    HUE_ORDER = list(SOLUTION_LOG_DICT.keys()) + ['Ground Truth']
    EULER_COLUMN = ['Roll', 'Pitch', 'Yaw']
    TIME_LABEL = 'Time to Attack (s)'


    result_dataframes = []
    for sol_name, fpath in SOLUTION_LOG_DICT.items():
        result_dataframes.append(process_attitude(fpath, sol_name))
    result_dataframes = pd.concat(result_dataframes)

    with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
        HEIGHT = 1.5  # inches
        WIDTH = 1.5  # inches
        plot = sns.relplot(
            result_dataframes,
            x=TIME_LABEL,
            y='EulerAngle',
            hue='DataType',
            hue_order=HUE_ORDER,
            palette={
                HUE_ORDER[0]: 'tab:red',
                HUE_ORDER[1]: 'tab:orange',
                HUE_ORDER[2]: 'tab:purple',
                HUE_ORDER[3]: 'black'
            },
            row='AngleAxis',
            row_order=EULER_COLUMN[:-1],  # Not include Yaw
            col='Solution',
            kind='line',
            facet_kws=dict(margin_titles=True),
            height=HEIGHT,
            aspect=WIDTH / HEIGHT,
        )

        plot.despine(top=True, right=True, left=True)
        plot.legend.set_title(None)  # Remove title
        sns.move_legend(
            plot,
            ncol=4,
            loc="lower center",
            columnspacing=0.5,
            handlelength=2.0,
        )

        for row_idx, row in enumerate(plot.axes):
            for col_idx, ax in enumerate(np.atleast_1d(row)):
                rect = plt.Rectangle((0, -180), 1.01, 360, alpha=0.1, edgecolor=None, facecolor='red')
                ax.add_patch(rect)
                ax.set_xlim([-1.8, 1.01])
                ax.set_ylim([-15, 10])

                if col_idx == 0:
                    ax.yaxis.label.set_text(f"{plot.row_names[row_idx]} (Deg)")

        plot.set_titles(
            row_template="",
            col_template="{col_name}"
        )

        plot.tight_layout(pad=0.5, rect=[0.0, 0.1, 1.0, 1.0])
        plot.figure.subplots_adjust(wspace=0.05, hspace=0.1)

        plot.fig.show()

    with PdfPages('Supplementary Compensation Error.pdf') as pdf_pages:
        pdf_pages.savefig(plot.fig)
