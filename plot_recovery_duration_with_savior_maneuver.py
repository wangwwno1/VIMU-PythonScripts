import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

TTD_THRESHOLD_S = 15.0
df = pd.read_csv('data/evaluation_reports/Complex Maneuver Recovery - 08.15/recovery_reports.csv', index_col=False)

BASELINE_NAME_MAP = {
    'savior': 'SAVIOR',
    'virtual_imu': 'VIMU'
}
COL_ORDER = [BASELINE_NAME_MAP[k] for k in ['savior', 'virtual_imu']]

df['Solution'] = df['Solution'].replace(BASELINE_NAME_MAP)

y_axis_column = rating_column = "Duration (s)"
x_axis_column = deviation_column = 'Deviation (rad/s)'
rating_order = ['> 300', '60~300', '10~60', '4~10', '< 4']
scene_order = ['Maneuver']

df = df.rename(
    columns=dict(
        baseline_name='Solution',
        scene_name='Scene',
        ATK_GYR_BIAS=deviation_column,
        ATK_GYR_AMP=deviation_column,
    )
)

df = df.loc[df.loc[:, deviation_column] >= 0.079]


def mark_recovery_performance(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()

    recovery_durations_s = dataframe['recovery_duration_s']

    dataframe[rating_column] = rating_order[-1]
    dataframe.loc[recovery_durations_s > 300.0, rating_column] = rating_order[0]
    dataframe.loc[recovery_durations_s.between(60.0, 300.0, 'right'), rating_column] = rating_order[1]
    dataframe.loc[recovery_durations_s.between(10.0, 60.0, 'right'), rating_column] = rating_order[2]
    dataframe.loc[recovery_durations_s.between(4.0, 10.0, 'right'), rating_column] = rating_order[3]
    return dataframe


result_0 = mark_recovery_performance(df)
result_1 = result_0.loc[:, ['Solution', 'Scene', deviation_column, rating_column]]

result_1[deviation_column] = result_1[deviation_column].round(2)
time_to_detect_cat = result_1[deviation_column].unique()
time_to_detect_cat.sort()
time_to_detect_cat = time_to_detect_cat.astype(str)
result_1[deviation_column] = result_1[deviation_column].astype(str).astype("category")
result_1[deviation_column] = result_1[deviation_column].cat.set_categories(time_to_detect_cat, ordered=True)

result_1[rating_column] = result_1[rating_column].astype(str).astype("category")
result_1[rating_column] = result_1[rating_column].cat.set_categories(rating_order[::-1], ordered=True)

with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    HEIGHT = 2.25  # inches
    WIDTH = 2.25  # inches
    grid_kwargs = dict(
        sharex=False,
        sharey=True,
        margin_titles=True
    )

    plot = sns.displot(
        result_1,
        y=deviation_column,
        hue=rating_column,
        hue_order=rating_order[::-1],
        palette={
            rating_order[0]: 'tab:blue',
            rating_order[1]: 'tab:green',
            rating_order[2]: 'tab:olive',
            rating_order[3]: 'tab:orange',
            rating_order[4]: 'tab:red'
        },
        row='Scene', row_order=scene_order,
        shrink=.8,
        col='Solution',
        col_order=COL_ORDER,
        kind='hist',
        facet_kws=grid_kwargs,
        multiple='stack',
        height=HEIGHT, aspect=WIDTH / HEIGHT,
    )

    plot.legend.remove()
    plot.axes[0, 0].legend(handles=plot.legend.legend_handles, labels=rating_order[::-1],
                           title=rating_column, loc='upper right', framealpha=1.0,
                           columnspacing=0.5, labelspacing=0.1)

    plot.despine(top=False, right=False)
    plot.set_titles(
        row_template="{row_name}",
        col_template="{col_name}"
    )

    plot.tick_params(axis='x', which='major', grid_color='black', grid_alpha=1)
    plot.figure.subplots_adjust(wspace=0.0)

    # Plot one super label for x & y axis
    suplabel_font = plot.axes.flatten()[0].yaxis.label.get_font()
    plot.figure.supylabel(plot.axes.flatten()[0].yaxis.label.get_text(),
                          fontproperties=suplabel_font, ha='center')
    # plot.figure.supxlabel("Log Counts",
    #                       fontproperties=suplabel_font, ha='center')
    plot.set_xlabels("#Record")

    for row_idx, row in enumerate(plot.axes):
        for col_idx, ax in enumerate(row):
            ax.set_xlim(0, 50)
            ax.yaxis.label.set_visible(False)
            for spine in ax.spines.values():
                spine.set(alpha=1.0, color='black')

            if row_idx == 0 and len(plot.axes) > 1:
                ax.tick_params(axis='x', labelbottom=False)
            if col_idx != 0:
                ax.invert_xaxis()
                ax.tick_params(axis='y', labelleft=False)

    plot.tight_layout(pad=0.5, rect=[0, 0.0, 1, 1])
    plot.figure.subplots_adjust(hspace=0.02, wspace=0.0)

plot.figure.show()

with PdfPages('Recovery Duration with SAVIOR - Maneuver.pdf') as pdf_pages:
    pdf_pages.savefig(plot.fig)
