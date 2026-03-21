from utilz.preprocessing_utilz import *
from utilz.helpers import *

from utilz.Dataset import load_dataset
from utilz.helpers import plot_pca
from utilz.constans import DISEASE, HEALTHY

meta_path = r"../../data/samples_pancreatic.xlsx"
data_path = r"../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
ds.y = ds.y.replace({DISEASE: HEALTHY})
meta = ds.meta

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)
X_train, X_test, X_valid, y_train, y_test, y_valid = (
    ds.get_train_test_valid_split(ds.X, y_encoded, test_size=0.25, valid_size=0.25))
sex_numeric = ds.sex.map({"F": 0, "M": 1})

num_pca_components = 4
plot_pca(ds.X, y_encoded, num_pca_components, le)

pipeline = Pipeline([
    ('ConstantExpressionReductor', ConstantExpressionReductor()),
    ('AnovaReductor', AnovaReductor(percentile=80)),
    ('MeanExpressionReductor', MeanExpressionReductor(percentile=25)),
    ('AgeBiasReductor',  CovariatesBiasReductor(covariate=ds.age)),
    ('SexBiasReductor',  CovariatesBiasReductor(covariate=sex_numeric)),
])
pipeline.fit(X_train, y_train)
X_test = pipeline.transform(X_test)

plot_pca(X_test, y_test, num_pca_components, le)