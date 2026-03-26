from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


def test_covariate_bias_reductor():
    ds = load_dataset(r"../../data/counts_pancreatic.csv",
                      r"../../data/samples_pancreatic.xlsx", label_col="Group")

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

    X_train, X_test, X_valid, y_train, y_test, y_valid = (
        ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25))

    # filtracja genów skorelowanych z wiekiem (fit na train, z diagnozą)
    reductor =  CovariatesBiasReductor(covariate=ds.age, beta_thresh=0.00005)

    reductor.fit(X_train, y_train)
    n_before, n_after = X_train.shape[1], len(reductor.selected_genes_)
    print(f"Geny: {n_before} → {n_after}")

    # tylko zdrowi z dostępnym wiekiem — do regresji
    def healthy_with_age(X, y_enc, ds):
        healthy = y_enc[y_enc == le.transform([HEALTHY])[0]].index
        age = ds.age.dropna().astype(int)
        idx = X.index.intersection(healthy).intersection(age.index)
        return X.loc[idx], age.loc[idx]

    X_tr, y_tr = healthy_with_age(X_train, y_train, ds)
    X_te, y_te = healthy_with_age(X_test, y_test, ds)

    X_tr_deb, X_te_deb = reductor.transform(X_tr), reductor.transform(X_te)

    # modele
    pipes = {
        "Baseline": (Pipeline([
            ('const', ConstantExpressionReductor()),
            ('scaler', MinMaxScaler()),
            ('model', LinearRegression()
    )
        ]), X_tr, X_te),
        f"Debiased ({n_after})": (Pipeline([
            ('const', ConstantExpressionReductor()),
            ('scaler', MinMaxScaler()),
            ('model', LinearRegression())
        ]), X_tr_deb, X_te_deb),
    }

    # ewaluacja + wykresy
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, (pipe, X_fit, X_ev)) in zip(axes, pipes.items()):
        pipe.fit(X_fit, y_tr)
        y_pred = pipe.predict(X_ev)
        mae = mean_absolute_error(y_te, y_pred)
        r2 = pipe.score(X_ev, y_te)
        print(f"\n=== {name} ===\nMAE: {mae:.3f}  R²: {r2:.4f}")

        lim = [min(y_te.min(), y_pred.min()) - 2, max(y_te.max(), y_pred.max()) + 2]
        ax.scatter(y_te, y_pred, alpha=0.5)
        ax.plot(lim, lim, 'r--', label='ideał')
        ax.set(xlim=lim, ylim=lim, xlabel="Prawdziwy wiek",
               ylabel="Przewidywany wiek", title=f"{name}\nMAE={mae:.2f}, R²={r2:.3f}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f"Wpływ usunięcia genów skorelowanych z wiekiem\n"
                 f"(p_thresh=0.05, FDR-BH, {n_before} do{n_after} genów)", y=2)
    plt.tight_layout()
    plt.show()

def test_high_variance_reductor():

    ds = load_dataset(r"../../data/counts_pancreatic.csv",
                      r"../../data/samples_pancreatic.xlsx", label_col="Group")

    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

    X_train, X_test, X_valid, y_train, y_test, y_valid = (
        ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25))

    const = ConstantExpressionReductor()
    X_tr = const.fit_transform(X_train, y_train)

    hv = AnovaReductor(percentile=95)
    X_tr_hv = hv.fit_transform(X_tr)

    var_before = X_tr.var(axis=0)
    var_after = X_tr_hv.var(axis=0)
    threshold = np.percentile(var_before, 95)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Histogram
    axes[0].hist(var_before, bins=100, alpha=0.6, label=f'Przed ({len(var_before)})')
    axes[0].hist(var_after, bins=100, alpha=0.6, label=f'Po ({len(var_after)})')
    axes[0].axvline(threshold, color='red', ls='--', label=f'próg = {threshold:.3f}')
    axes[0].set(xlabel='Wariancja', ylabel='Liczba genów', title='Rozkład wariancji')
    axes[0].legend()

    # 2. Log-skala
    axes[1].hist(np.log1p(var_before), bins=100, alpha=0.6, label='Przed')
    axes[1].hist(np.log1p(var_after), bins=100, alpha=0.6, label='Po')
    axes[1].axvline(np.log1p(threshold), color='red', ls='--', label='próg')
    axes[1].set(xlabel='log(1 + wariancja)', ylabel='Liczba genów', title='Rozkład wariancji (log)')
    axes[1].legend()

    # 3. Posortowane wariancje
    sorted_var = var_before.sort_values()
    colors = ['tab:blue' if v <= threshold else 'tab:red' for v in sorted_var]
    axes[2].scatter(range(len(sorted_var)), sorted_var, c=colors, s=1, alpha=0.5)
    axes[2].axhline(threshold, color='red', ls='--', label=f'próg = {threshold:.3f}')
    axes[2].set(xlabel='Geny (posortowane)', ylabel='Wariancja',
                title='Niebieskie = zachowane, Czerwone = odrzucone')
    axes[2].legend()

    plt.suptitle(f'HighVarianceReductor (percentile=95): {len(var_before)} → {len(var_after)} genów')
    plt.tight_layout()
    plt.show()

test_covariate_bias_reductor()