import plotly.express as px
from tbparse import SummaryReader


def generate_pivot_table(logdir, columns=None):
    if columns is None:
        columns = ["dropout", "lr", "swa_start", "pooling", "decay", "label_smoothing"]
    reader = SummaryReader(logdir, extra_columns={'dir_name'})  # 'wall_time',
    hparams = reader.hparams
    scalars = reader.scalars
    hparams['run'] = hparams['dir_name'].str.split('-').str[0].str[3:].apply(int)
    scalars['run'] = scalars['dir_name'].str.split('-').str[0].str[3:].apply(int)
    # scalars['wall_clock'] = pd.to_datetime(scalars['wall_time'], unit='s')
    hparams = hparams.set_index("run")
    scalars = scalars.set_index("run")
    scalars = scalars.loc[scalars['tag'].str.contains("hparams")]
    hparams = hparams.pivot(columns="tag", values="value")
    scalars = scalars.pivot(columns="tag", values="value")
    # Select only interesting columns
    hparams = hparams.loc[:, columns]
    # Merge tables to get metrics together with params
    hparams = hparams.join(scalars)
    return hparams


def parallel_coord_plot(df, metric="hparams/val_acc"):
    fig = px.parallel_coordinates(df, color=metric)
    fig.show()


if __name__ == "__main__":
    df = generate_pivot_table("tensorboard_logs.notxfm/cnn")
    parallel_coord_plot(df)
