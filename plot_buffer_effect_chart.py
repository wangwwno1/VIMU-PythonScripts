import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TTD_LABEL = "TTD (ms)"

report_folder = "data/evaluation_reports/Effect of Buffer Length"

ATTACK_NAME_MAP = {
    't_buffer_test': r'$SA^{3/3}_{Gyro}$',
}

SETTING_NAME_MAP = {
    'virtual_imu_t_buf_250': '250',
    'virtual_imu_t_buf_500': '500',
    'virtual_imu_t_buf_750': '750',
}

WIND_SETTING_ORDER = ['Default', '1.0', '2.0']


def replace_labels(data):
    data = data.rename(columns=dict(
        baseline_name='Solution',
        scene_name='Scene',
        test_case='Attack'
    ))
    data['Setting'] = data['Solution']
    data['Attack'] = data['Attack'].replace(ATTACK_NAME_MAP)
    data['Setting'] = data['Setting'].replace(SETTING_NAME_MAP)
    return data

SCALE_SEC_TO_MS = 1e3

SETTING_LABEL = r'T$_{Buf}$ (ms)'

def mark_recovery_performance(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()

    recovery_durations_s = dataframe['recovery_duration_s']

    dataframe[rating_column] = rating_order[-1]
    dataframe.loc[recovery_durations_s > 300.0, rating_column] = rating_order[0]
    dataframe.loc[recovery_durations_s.between(60, 300.0, 'right'), rating_column] = rating_order[1]
    dataframe.loc[recovery_durations_s.between(10, 60.0, 'right'), rating_column] = rating_order[2]
    return dataframe

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

with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    HEIGHT = 1.75  # inches
    WIDTH = 2.25  # inches
    grid_kwargs = dict(
        sharex=False,
        sharey=True,
        margin_titles=True
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
    duration_histogram.despine(top=False, right=False)

    # Plot one super ylabel for all y axis
    supylabel_font = duration_histogram.axes.flatten()[0].yaxis.label.get_font()
    duration_histogram.figure.supylabel(SETTING_LABEL, fontproperties=supylabel_font, ha='center')

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
    with PdfPages('Recovery Duration - Buffer Length.pdf') as pdf_pages:
        pdf_pages.savefig(duration_histogram.fig)
