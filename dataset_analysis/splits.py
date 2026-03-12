import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})

def plot_split_balance(splits: dict):
    """
    splits = {
        'Train': (y_train, sex_train, age_train),
        'Test':  (y_test,  sex_test,  age_test),
        'Valid': (y_valid, sex_valid, age_valid),
    }
    """
    COLORS = {'Train': '#6366f1', 'Test': '#22d3ee', 'Valid': '#f59e0b'}
    names = list(splits.keys())

    all_y   = splits[names[0]][0]
    class_vals = sorted(all_y.unique())
    sex_vals   = sorted(splits[names[0]][1].unique())

    class_counts = {s: splits[s][0].value_counts(normalize=True) for s in names}
    sex_counts   = {s: splits[s][1].value_counts(normalize=True) for s in names}
    age_data     = {s: splits[s][2].values for s in names}

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Class distribution", "Sex distribution", "Age distribution"],
        horizontal_spacing=0.10,
    )

    for s in names:
        fig.add_trace(go.Bar(
            name=s, x=class_vals,
            y=[class_counts[s].get(c, 0) for c in class_vals],
            marker_color=COLORS[s], showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name=s, x=sex_vals,
            y=[sex_counts[s].get(sv, 0) for sv in sex_vals],
            marker_color=COLORS[s], showlegend=False,
        ), row=1, col=2)

        fig.add_trace(go.Box(
            name=s, y=age_data[s],
            marker_color=COLORS[s], boxmean=True, showlegend=False,
        ), row=1, col=3)

    fig.update_layout(
        barmode='group',
        title={"text": "Train / Test / Valid split balance"},
        legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5),
    )
    fig.update_yaxes(title_text="Proportion", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Proportion", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(title_text="Age (years)", row=1, col=3)

    fig.show()

X_train, X_test, X_valid, y_train, y_test, y_valid = ds.get_train_test_valid_split(ds.X, ds.y)

plot_split_balance({
    'Train': (y_train, ds.sex.loc[X_train.index], ds.age.loc[X_train.index]),
    'Test':  (y_test,  ds.sex.loc[X_test.index],  ds.age.loc[X_test.index]),
    'Valid': (y_valid, ds.sex.loc[X_valid.index],  ds.age.loc[X_valid.index]),
})
