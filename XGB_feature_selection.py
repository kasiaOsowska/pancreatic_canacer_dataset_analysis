from Dataset import load_dataset

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from utilz import *
from xgboost import XGBClassifier, plot_tree


meta_path = r"../data/samples_pancreatic_filtered.xlsx"
data_path = r"../data/counts_pancreatic_filtered.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
X = ds.X
meta = ds.meta
y = ds.y
y = y.replace({DISEASE: HEALTHY})


print(X.shape)

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y), index=y.index)

bst = XGBClassifier(n_estimators=220, max_depth=3, learning_rate=0.06, objective='binary:logistic',
                    colsample_bytree = 0.8, reg_lambda = 3.0, gamma = 0, min_child_weight = 1,
                    subsample = 0.9)
bst.fit(ds.X, y_encoded)
importance_dict = bst.get_booster().get_score(importance_type='gain')
importance_df = (
    pd.DataFrame.from_dict(importance_dict, orient='index', columns=['importance'])
    .sort_values(by='importance', ascending=False)
)

print(importance_df.head(20))




