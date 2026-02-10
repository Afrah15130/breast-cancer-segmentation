# breast-cancer-segmentation
Streamlit-style breast cancer tumor segmentation system with automatic detection and classification

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2lab
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10  # keep small to avoid overfitting & timeouts


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "/content/drive/MyDrive/dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    "/content/drive/MyDrive/dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

cnn_model = Model(
    inputs=base_model.input,
    outputs=GlobalAveragePooling2D()(base_model.output)
)


def extract_graph_features(image):
    """
    image: RGB image (0–255)
    returns: graph-level feature vector
    """
    segments = slic(image, n_segments=100, compactness=10, start_label=0)
    lab_img = rgb2lab(image)

    node_features = []
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        mean_color = lab_img[mask].mean(axis=0)
        node_features.append(mean_color)

    node_features = np.array(node_features)

    # ---- GNN-style aggregation (mean pooling) ----
    graph_embedding = node_features.mean(axis=0)

    return graph_embedding

X_features = []
y_labels = []

for i in range(len(train_data)):
    imgs, labels = train_data[i]
    for img, lbl in zip(imgs, labels):
        img_uint8 = (img * 255).astype(np.uint8)

        cnn_feat = cnn_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        graph_feat = extract_graph_features(img_uint8)

        combined_feat = np.concatenate([cnn_feat, graph_feat])
        X_features.append(combined_feat)
        y_labels.append(lbl)

X_features = np.array(X_features)
y_labels = np.array(y_labels)

classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(X_features.shape[1],)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

classifier.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

classifier.fit(X_features, y_labels, epochs=EPOCHS, verbose=1)


X_test = []
y_true = []

for i in range(len(val_data)):
    imgs, labels = val_data[i]
    for img, lbl in zip(imgs, labels):
        img_uint8 = (img * 255).astype(np.uint8)

        cnn_feat = cnn_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        graph_feat = extract_graph_features(img_uint8)

        combined_feat = np.concatenate([cnn_feat, graph_feat])
        X_test.append(combined_feat)
        y_true.append(lbl)

X_test = np.array(X_test)
y_true = np.array(y_true)

y_pred_prob = classifier.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
dice = (2 * precision * recall) / (precision + recall + 1e-6)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("Dice     :", dice)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Healthy", "Unhealthy"]))


cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()


sample_img, _ = val_data[0]
sample_img = (sample_img[0] * 255).astype(np.uint8)

segments = slic(sample_img, n_segments=100, compactness=10)
plt.figure(figsize=(6,6))
plt.imshow(mark_boundaries(sample_img, segments))
plt.title("Superpixel Visualization")
plt.axis("off")
plt.show()
