from Dataset import load_dataset

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from utilz import *
from xgboost import XGBClassifier


meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
X = ds.X
meta = ds.meta
y = ds.y
y = y.replace({DISEASE: HEALTHY})


print(X.shape)

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y), index=y.index)

bst = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, objective='binary:logistic',
                    subsample = 0.9)
bst.fit(ds.X, y_encoded)
importance_dict = bst.get_booster().get_score(importance_type='gain')
importance_df = (
    pd.DataFrame.from_dict(importance_dict, orient='index', columns=['importance'])
    .sort_values(by='importance', ascending=False)
)

print(importance_df.head(20))


