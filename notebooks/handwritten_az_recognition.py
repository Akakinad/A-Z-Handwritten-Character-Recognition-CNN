# Az Handwritten Character Recognition using CNN
# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.patches import Rectangle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set style
plt.style.use('ggplot')

# Step 2: Load dataset
print("Loading dataset...")
data = pd.read_csv('dataset/A_Z Handwritten Data.csv')
data.columns = ['label'] + [f'pixel{i}' for i in range(784)]

# Step 3: Shuffle and split into training/testing
print("Shuffling and splitting dataset...")
data = data.sample(frac=1).reset_index(drop=True)
train = data.iloc[:250000, :]
test = data.iloc[250000:350000, :].reset_index(drop=True)

# Step 4: Visualize class distribution
letters = [chr(i) for i in range(65, 91)]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.barh(letters, train['label'].value_counts().sort_index(), color=sns.color_palette('viridis', 26))
ax1.set_title('Training Frequency')
ax2.barh(letters, test['label'].value_counts().sort_index(), color=sns.color_palette('viridis', 26))
ax2.set_title('Testing Frequency')
plt.tight_layout()
plt.show()

# Step 5: Prepare features and labels
train_x = train.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
train_y = to_categorical(train['label'], 26)
test_x = test.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
test_y = test['label']

# Step 6: Visualize some samples
random_indices = np.random.choice(len(train), 24, replace=False)
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(train[train.columns[1:]].iloc[random_indices[i]].values.reshape(28, 28), cmap='Greys')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Step 7: Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(26, activation='softmax')
])
model.summary()

# Step 8: Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
history = model.fit(train_x, train_y, validation_split=0.2, epochs=10, batch_size=50, callbacks=[early_stopping])

# Step 9: Plot training history
history_df = pd.DataFrame(history.history)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(history_df['accuracy'], label='Training Accuracy', color='blue')
ax1.plot(history_df['val_accuracy'], label='Validation Accuracy', color='green')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history_df['loss'], label='Training Loss', color='orange')
ax2.plot(history_df['val_loss'], label='Validation Loss', color='red')
ax2.set_title('Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

# Step 10: Evaluate on test set
test_y_cat = to_categorical(test_y, 26)
test_loss, test_accuracy = model.evaluate(test_x, test_y_cat, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Step 11: Predict on test set
preds = model.predict(test_x)
pred_labels = np.argmax(preds, axis=1)

# Step 12: Classification report and per-class accuracy
print("\nClassification Report:")
print(classification_report(test_y, pred_labels, target_names=letters))

class_accuracies = {}
for i in range(26):
    correct = np.sum((test_y == i) & (pred_labels == i))
    total = np.sum(test_y == i)
    acc = correct / total if total > 0 else 0
    class_accuracies[letters[i]] = acc

print("\nPer-Class Accuracy:")
for label, acc in class_accuracies.items():
    print(f"{label}: {acc * 100:.2f}%")


# Step 13: Visualize a random prediction
index = np.random.randint(len(test))
pixels = test.iloc[index, 1:].values.reshape(28, 28)
plt.imshow(pixels, cmap='Greys_r')
plt.axis('off')
plt.title(f"Predicted: {chr(pred_labels[index] + 65)}")
plt.show()

# Step 14: Count correct and incorrect predictions
wrong_preds = np.where(pred_labels != test_y)[0]
right_preds = np.where(pred_labels == test_y)[0]
print(f"\nCorrect predictions: {len(right_preds)} / {len(test_y)}")
print(f"Incorrect predictions: {len(wrong_preds)}")

# Step 15: Visualize correct predictions
np.random.shuffle(right_preds)
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(test.iloc[right_preds[i], 1:].values.reshape(28, 28), cmap='Greys')
    ax.axis('off')
    ax.set_title(f"Pred: {chr(pred_labels[right_preds[i]] + 65)}")
plt.tight_layout()
plt.show()

# Step 16: Plot confusion matrix
conf_mat = confusion_matrix(test_y, pred_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=letters, yticklabels=letters)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()