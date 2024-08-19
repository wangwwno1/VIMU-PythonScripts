import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

TTD_THRESHOLD_S = 15.0
df = pd.read_csv('data/evaluation_reports/Recovery Duration by TTD/recovery_reports.csv',
                 index_col=False)

BASELINE_NAME_MAP = {
    'virtual_imu': 'VIMU',
    'virtual_imu_no_buffer': 'VIMU-NoBuffer',
}

df['Solution'] = df['Solution'].replace(BASELINE_NAME_MAP)

x_axis_column = rating_column = "Duration (s)"
y_axis_column = 'Time to Detect (ms)'
rating_order = ['> 300', '60~300', '10~60', '< 10']

df = df.rename(
    columns=dict(
        baseline_name='Solution',
        # recovery_duration_s=x_axis_column,
        IV_TTD_DELAY_MS=y_axis_column
    )
)


def mark_recovery_performance(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()

    recovery_durations_s = dataframe['recovery_duration_s']

    dataframe[rating_column] = rating_order[-1]
    dataframe.loc[recovery_durations_s > 300.0, rating_column] = rating_order[0]
    dataframe.loc[recovery_durations_s.between(60, 300.0, 'right'), rating_column] = rating_order[1]
    dataframe.loc[recovery_durations_s.between(10, 60.0, 'right'), rating_column] = rating_order[2]
    return dataframe


result_0 = mark_recovery_performance(df)
result_1 = result_0.loc[:, ['Solution', x_axis_column, y_axis_column]]

time_to_detect_cat = result_1[y_axis_column].unique().astype(int)
time_to_detect_cat.sort()
time_to_detect_cat = time_to_detect_cat.astype(str)
result_1[y_axis_column] = result_1[y_axis_column].astype(int).astype(str).astype("category")
result_1[y_axis_column] = result_1[y_axis_column].cat.set_categories(time_to_detect_cat, ordered=True)

result_1[x_axis_column] = result_1[x_axis_column].astype(str).astype("category")
result_1[x_axis_column] = result_1[x_axis_column].cat.set_categories(rating_order[::-1], ordered=True)

with sns.axes_style("whitegrid"), sns.plotting_context(font_scale=1.6):
    HEIGHT = 2.5  # inches
    WIDTH = 2.25  # inches
    grid_kwargs = dict(
        sharex=False,
        sharey=True,
        margin_titles=True
    )

    plot = sns.displot(
        result_1,
        y=y_axis_column,
        hue=x_axis_column,
        hue_order=rating_order[::-1],
        palette={
            rating_order[0]: 'tab:blue',
            rating_order[1]: 'tab:green',
            rating_order[2]: 'tab:orange',
            rating_order[3]: 'tab:red'
        },
        # col='Attack', col_order=ATTACK_TYPE_ORDER,
        shrink=.9,
        col='Solution',
        col_order=[BASELINE_NAME_MAP['virtual_imu_no_buffer'], BASELINE_NAME_MAP['virtual_imu']],
        kind='hist',
        facet_kws=grid_kwargs,
        multiple='stack',
        height=HEIGHT, aspect=WIDTH / HEIGHT,
    )

    plot.legend.remove()
    plot.axes.flatten()[0].legend(handles=plot.legend.legend_handles,
                                  labels=rating_order[::-1],
                                  title=rating_column, loc='upper left', ncol=1,
                                  framealpha=1.0, columnspacing=0.5, labelspacing=0.1)
    plot.despine(top=False, right=False)
    plot.set_titles(
        row_template="{row_name}",
        col_template="{col_name}"
    )

    plot.tick_params(axis='x', which='major', grid_color='black', grid_alpha=1)
    plot.figure.subplots_adjust(wspace=0.0)
    plot.set_xlabels("#Record")

    for row_idx, row in enumerate(plot.axes):
        for col_idx, ax in enumerate(row):

            ax.set_xlim(0, 100)
            if col_idx == 0:
                pass
                ax.invert_xaxis()
            else:
                ax.tick_params(axis='y', labelleft=False)

            for spine in ax.spines.values():
                spine.set(alpha=1.0, color='black')

            # if row_idx == 0:
            #     ax.set_xlabel(None)
            #     ax.tick_params(axis='x', labelbottom=False)

    plot.tight_layout(pad=0.5, rect=[0, 0.0, 1, 1])
    plot.figure.subplots_adjust(wspace=0.)

plot.figure.show()

with PdfPages('Effect of Buffer and TTD.pdf') as pdf_pages:
    pdf_pages.savefig(plot.fig)
