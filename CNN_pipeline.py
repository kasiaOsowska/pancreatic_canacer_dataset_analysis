from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import class_weight

from Dataset import load_dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilz import *
from collections import Counter
import os, random, numpy as np, tensorflow as tf

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)


meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

X = ds.X.to_numpy()
print(Counter(y_encoded))
rus = RandomUnderSampler(random_state=42)
X, y_encoded = rus.fit_resample(X, y_encoded)
print(Counter(y_encoded))
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5,
                                                    random_state=42, stratify=y_encoded)


x_test, x_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5,
                                                    random_state=42, stratify=y_test)

X_train = X_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
x_valid = x_valid[..., np.newaxis]
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1], 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(8, kernel_size=3, activation='relu', padding='SAME'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding='SAME'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


def f1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.round(y_pred), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

model.compile(
    optimizer='adam',
    loss="binary_crossentropy",
    metrics=[f1]
)

batch_size = 4
epochs = 15
learning_rate = 0.0002

classes = np.unique(y_train)
cls_wgts = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

history_simple = model.fit(X_train, y_train, epochs=epochs,
                           batch_size=batch_size, validation_data=(x_valid, y_valid))

plt.figure()
plt.plot(history_simple.history['f1'], label='F1 train')
plt.plot(history_simple.history['val_f1'], label='F1 val')
plt.legend()
plt.show()

y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype("int32").reshape(-1)
cm = tf.math.confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=le.classes_))
disp = ConfusionMatrixDisplay(confusion_matrix=cm.numpy(), display_labels=le.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.show()