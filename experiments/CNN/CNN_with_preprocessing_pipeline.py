from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os, random
import tensorflow as tf

from utilz.constans import DISEASE, HEALTHY
from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)


meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")
y_containing_disease = ds.y

# combine healthy and disease into one class
ds.y = ds.y.replace({DISEASE: HEALTHY})


le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(ds.y), index=ds.y.index)

"""
rus = RandomUnderSampler(random_state=42)
ds.X, y_encoded = rus.fit_resample(ds.X, y_encoded)
"""

X_train, X_test, y_train, y_test = train_test_split(ds.X, y_encoded, test_size=0.5,
                                                    random_state=42, stratify=y_encoded)

x_test, x_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5,
                                                    random_state=42, stratify=y_test)


pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('scaler', StandardScaler()),
])

X_train = pipeline.fit_transform(X_train, y_train)
x_test = pipeline.transform(x_test)
x_valid = pipeline.transform(x_valid)


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


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryFocalCrossentropy(alpha = 0.1, from_logits = True,
            gamma = 3, reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
    metrics=[tf.keras.metrics.SensitivityAtSpecificity(specificity=0.8)]
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
]

batch_size = 16
epochs = 100
learning_rate = 1e-3

history_simple = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_valid, y_valid),
    callbacks=callbacks
)

plt.figure()
plt.plot(history_simple.history['sensitivity_at_specificity'], label='sensitivity_at_specificity train')
plt.plot(history_simple.history['val_sensitivity_at_specificity'], label='sensitivity_at_specificity val')
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