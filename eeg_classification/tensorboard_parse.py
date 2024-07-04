# import plotly.graph_objects as go
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


def _cleanup_df(df):
    df = df.infer_objects()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")
    return df


def parallel_coord_plot(df, metric="hparams/val_acc"):
    if "acc" in metric:
        cmin = 0
        cmax = 1
    else:
        cmin = df[metric].min()
        cmax = df[metric].max()
    df = _cleanup_df(df)
    fig = px.parallel_coordinates(df, color=metric, color_continuous_scale=px.colors.sequential.Viridis)
    fig.show()
    # fig = go.Figure(data=
    #     go.Parcoords(
    #         line=dict(color=df[metric],
    #                   colorscale='Viridis',
    #                   showscale=True,
    #                   cmin=cmin,
    #                   cmax=cmax),
    #         dimensions=list([
    #             dict(range=
    #         ])



if __name__ == "__main__":
    df = generate_pivot_table("tensorboard_logs.notxfm/cnn")
    parallel_coord_plot(df)
