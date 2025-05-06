import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)

# Load the MNIST dataset
print("Loading MNIST dataset...")
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

print(f"Original training data shape: {X_train_full.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of original training samples: {X_train_full.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train_full[i], cmap='gray')
    plt.title(f"Label: {y_train_full[i]}")
    plt.axis('off')
plt.suptitle("Sample MNIST Images")
plt.tight_layout()
plt.savefig('sample_images.png')
plt.show()

# Check class distribution
unique_train, counts_train = np.unique(y_train_full, return_counts=True)
plt.figure(figsize=(10, 5))
plt.bar(unique_train, counts_train)
plt.title('Class Distribution in Training Set')
plt.xlabel('Digit')
plt.ylabel('Count')
plt.xticks(range(10))
plt.grid(axis='y', alpha=0.3)
plt.savefig('class_distribution.png')
plt.show()

print("Preprocessing data and creating train/validation/test splits...")

# Normalize pixel values to be between 0 and 1
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train_full
)

print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

# Reshape data for the models
X_train_mlp = X_train.reshape(-1, 28*28)
X_val_mlp = X_val.reshape(-1, 28*28)
X_test_mlp = X_test.reshape(-1, 28*28)

X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_val_cnn = X_val.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

# One-hot encode y
y_train_one_hot = to_categorical(y_train, 10)
y_val_one_hot = to_categorical(y_val, 10)
y_test_one_hot = to_categorical(y_test, 10)

# Build and train a MLP model
print("Building and training MLP model...")
mlp_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

mlp_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nMLP Model Architecture:")
mlp_model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

mlp_history = mlp_model.fit(
    X_train_mlp, y_train_one_hot,
    batch_size=128,
    epochs=15,
    validation_data=(X_val_mlp, y_val_one_hot),  # Use explicit validation set
    callbacks=[early_stopping],
    verbose=1
)

# Build and train a CNN model
print("\nBuilding and training CNN model...")
cnn_model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nCNN Model Architecture:")
cnn_model.summary()

cnn_history = cnn_model.fit(
    X_train_cnn, y_train_one_hot,
    batch_size=128,
    epochs=10,
    validation_data=(X_val_cnn, y_val_one_hot),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate both models on all datasets
print("\nFinal evaluation metrics:")

# Training set evaluation
print("Evaluating on training set...")
mlp_train_loss, mlp_train_acc = mlp_model.evaluate(X_train_mlp, y_train_one_hot, verbose=1)
cnn_train_loss, cnn_train_acc = cnn_model.evaluate(X_train_cnn, y_train_one_hot, verbose=1)

# Validation set evaluation
print("\nEvaluating on validation set...")
mlp_val_loss, mlp_val_acc = mlp_model.evaluate(X_val_mlp, y_val_one_hot, verbose=1)
cnn_val_loss, cnn_val_acc = cnn_model.evaluate(X_val_cnn, y_val_one_hot, verbose=1)

# Test set evaluation
print("\nEvaluating on test set...")
mlp_test_loss, mlp_test_acc = mlp_model.evaluate(X_test_mlp, y_test_one_hot, verbose=1)
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test_one_hot, verbose=1)

print("\nMLP Model Performance:")
print(f"{'Dataset':<10} {'Accuracy':<10} {'Loss':<10}")
print(f"{'-'*30}")
print(f"{'Training':<10} {mlp_train_acc:.4f}     {mlp_train_loss:.4f}")
print(f"{'Validation':<10} {mlp_val_acc:.4f}     {mlp_val_loss:.4f}")
print(f"{'Test':<10} {mlp_test_acc:.4f}     {mlp_test_loss:.4f}")

print("\nCNN Model Performance:")
print(f"{'Dataset':<10} {'Accuracy':<10} {'Loss':<10}")
print(f"{'-'*30}")
print(f"{'Training':<10} {cnn_train_acc:.4f}     {cnn_train_loss:.4f}")
print(f"{'Validation':<10} {cnn_val_acc:.4f}     {cnn_val_loss:.4f}")
print(f"{'Test':<10} {cnn_test_acc:.4f}     {cnn_test_loss:.4f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
datasets = ['Training', 'Validation', 'Test']
mlp_accs = [mlp_train_acc, mlp_val_acc, mlp_test_acc]
cnn_accs = [cnn_train_acc, cnn_val_acc, cnn_test_acc]

x = np.arange(len(datasets))
width = 0.35

plt.bar(x - width/2, mlp_accs, width, label='MLP')
plt.bar(x + width/2, cnn_accs, width, label='CNN')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xticks(x, datasets)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Loss comparison
plt.subplot(1, 2, 2)
mlp_losses = [mlp_train_loss, mlp_val_loss, mlp_test_loss]
cnn_losses = [cnn_train_loss, cnn_val_loss, cnn_test_loss]

plt.bar(x - width/2, mlp_losses, width, label='MLP')
plt.bar(x + width/2, cnn_losses, width, label='CNN')
plt.ylabel('Loss')
plt.title('Model Loss Comparison')
plt.xticks(x, datasets)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# Training history for both models
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(mlp_history.history['accuracy'], label='MLP Train')
plt.plot(mlp_history.history['val_accuracy'], label='MLP Validation')
plt.plot(cnn_history.history['accuracy'], label='CNN Train')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(mlp_history.history['loss'], label='MLP Train')
plt.plot(mlp_history.history['val_loss'], label='MLP Validation')
plt.plot(cnn_history.history['loss'], label='CNN Train')
plt.plot(cnn_history.history['val_loss'], label='CNN Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("\nGenerating predictions and analyzing results for the CNN model...")
y_pred_cnn = cnn_model.predict(X_test_cnn)
y_pred_classes_cnn = np.argmax(y_pred_cnn, axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes_cnn)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (CNN Model)')
plt.xticks(np.arange(10) + 0.5, np.arange(10))
plt.yticks(np.arange(10) + 0.5, np.arange(10))
plt.savefig('confusion_matrix.png')
plt.show()

print("\nClassification Report (CNN Model):")
report = classification_report(y_test, y_pred_classes_cnn, digits=4)
print(report)

misclassified_idx = np.where(y_pred_classes_cnn != y_test)[0]
correctly_classified_idx = np.where(y_pred_classes_cnn == y_test)[0]

# Display some correctly classified examples
plt.figure(figsize=(12, 4))
plt.suptitle("Correctly Classified Examples", y=1.05)
for i in range(5):
    idx = correctly_classified_idx[i]
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f"True: {y_test[idx]}\nPred: {y_pred_classes_cnn[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('correct_examples.png')
plt.show()

# Display some misclassified examples
if len(misclassified_idx) > 0:
    plt.figure(figsize=(12, 4))
    plt.suptitle("Misclassified Examples", y=1.05)
    for i in range(min(5, len(misclassified_idx))):
        idx = misclassified_idx[i]
        plt.subplot(1, 5, i+1)
        plt.imshow(X_test[idx], cmap='gray')
        plt.title(f"True: {y_test[idx]}\nPred: {y_pred_classes_cnn[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    plt.show()

# Examples where the model was less confident
pred_probs = cnn_model.predict(X_test_cnn)
max_probs = np.max(pred_probs, axis=1)
uncertain_samples = np.argsort(max_probs)[:5]

plt.figure(figsize=(15, 8))
plt.suptitle("Examples with Low Confidence Predictions", y=1.05)
for i, idx in enumerate(uncertain_samples):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f"True: {y_test[idx]}\nPred: {y_pred_classes_cnn[idx]}")
    plt.axis('off')

    plt.subplot(2, 5, i+6)
    plt.bar(range(10), pred_probs[idx])
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.title(f"Confidence: {max_probs[idx]:.2f}")
plt.tight_layout()
plt.savefig('uncertain_examples.png')
plt.show()

# Check for overfitting
plt.figure(figsize=(10, 6))
plt.title('Error Differences (Training vs. Validation vs. Test)')
plt.bar(['MLP Train-Val', 'MLP Val-Test', 'CNN Train-Val', 'CNN Val-Test'], 
        [abs(mlp_train_loss - mlp_val_loss), 
         abs(mlp_val_loss - mlp_test_loss),
         abs(cnn_train_loss - cnn_val_loss),
         abs(cnn_val_loss - cnn_test_loss)])
plt.ylabel('Absolute Error Difference')
plt.grid(axis='y', alpha=0.3)
plt.savefig('error_differences.png')
plt.show()

# Saving models
print("\nSaving models...")
mlp_model.save('mnist_mlp_model.h5')
cnn_model.save('mnist_cnn_model.h5')

print("\nProject completed successfully! Models saved as 'mnist_mlp_model.h5' and 'mnist_cnn_model.h5'")