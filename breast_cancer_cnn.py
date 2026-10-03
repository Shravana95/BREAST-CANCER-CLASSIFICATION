import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

DATASET_DIR = "dataset"
IMG_SIZE = (50, 50)
BATCH_SIZE = 16
SEED = 42

print("TensorFlow version:", tf.__version__)
print()

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

class_names = train_dataset.class_names

print("Classes:", class_names)
print()

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)

model = models.Sequential([
    layers.Rescaling(
        1.0 / 255,
        input_shape=(50, 50, 3)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("CNN Model:")
model.summary()
print()

print("Training CNN for 5 epochs")

history_5 = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5
)

print()
print("Continuing training to 10 epochs")

history_10 = model.fit(
    train_dataset,
    validation_data=test_dataset,
    initial_epoch=5,
    epochs=10
)

accuracy = (
    history_5.history["accuracy"]
    + history_10.history["accuracy"]
)

val_accuracy = (
    history_5.history["val_accuracy"]
    + history_10.history["val_accuracy"]
)

loss = (
    history_5.history["loss"]
    + history_10.history["loss"]
)

val_loss = (
    history_5.history["val_loss"]
    + history_10.history["val_loss"]
)

os.makedirs("output", exist_ok=True)

plt.figure()

plt.plot(
    range(1, 11),
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    range(1, 11),
    val_accuracy,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy")
plt.legend()

plt.savefig(
    "output/cnn_accuracy.png"
)

plt.close()

plt.figure()

plt.plot(
    range(1, 11),
    loss,
    label="Training Loss"
)

plt.plot(
    range(1, 11),
    val_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")
plt.legend()

plt.savefig(
    "output/cnn_loss.png"
)

plt.close()

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predictions = (
        predictions.flatten() >= 0.5
    ).astype(int)

    y_pred.extend(predictions)

    y_true.extend(
        labels.numpy().astype(int).flatten()
    )

accuracy_score = np.mean(
    np.array(y_true) == np.array(y_pred)
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

print()
print("Final Results")
print("Accuracy :", round(accuracy_score, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

cm = confusion_matrix(
    y_true,
    y_pred
)

print()
print("Confusion Matrix:")
print(cm)

plt.figure()

plt.imshow(cm)

plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.colorbar()

plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.savefig(
    "output/confusion_matrix.png"
)

plt.close()

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print()
print("Classification Report:")
print(report)

with open(
    "output/classification_report.txt",
    "w"
) as file:
    file.write(report)

model.save(
    "output/breast_cancer_cnn.keras"
)

print()
print("Project completed successfully!")
print("Results saved in the output folder.")