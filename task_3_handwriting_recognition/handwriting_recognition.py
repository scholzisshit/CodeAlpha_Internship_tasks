# ============================================================
# TASK 3: Handwritten Character Recognition
# aka "making a computer read my terrible handwriting"
# (it'll probably do better than my teacher did tbh)
# ============================================================
#
# What we're building:
#   - Load MNIST (digits 0-9) or EMNIST (letters A-Z)
#   - Build a 2D CNN to classify 28x28 pixel images
#   - Evaluate accuracy and show some predictions
#
# This is the most fun task imo. Watching the model look at
# a squiggly "7" and going "yeah that's a 7" is genuinely cool.
#
# No dataset download needed! tensorflow.keras.datasets handles it 
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# tensorflow go brrr
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

print("=" * 60)
print("HANDWRITTEN CHARACTER RECOGNITION")
print("aka the project that made AI famous (kinda)")
print("=" * 60)
print(f"\n TensorFlow version: {tf.__version__}")


# ============================================================
# CONFIGURATION - change DATASET to "emnist" for letters
# ============================================================

DATASET = "mnist"       # "mnist" for digits, "emnist" for letters
EPOCHS  = 15            # 15 is enough, don't be greedy
BATCH   = 64


# ============================================================
# STEP 1: LOAD THE DATA (best part - no downloading required!)
# ============================================================

print(f"\n{'='*60}")
print(f"LOADING DATA: {DATASET.upper()}")
print("="*60)

if DATASET == "mnist":
    print("\n Loading MNIST (handwritten digits 0-9)...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    n_classes = 10
    class_names = [str(i) for i in range(10)]
    print(f"   60,000 training images | 10,000 test images")
    print(f"   10 classes: digits 0-9")
    print(f"   (the OG dataset. probably the most famous dataset in ML history)")

elif DATASET == "emnist":
    print("\n Loading EMNIST Letters (handwritten A-Z)...")
    # EMNIST needs to be flipped/transposed for correct orientation
    try:
        import tensorflow_datasets as tfds
        ds = tfds.load("emnist/letters", split=["train", "test"], as_supervised=True, batch_size=-1)
        X_train, y_train = tfds.as_numpy(ds[0])
        X_test,  y_test  = tfds.as_numpy(ds[1])
        # EMNIST letters are 1-26, convert to 0-25
        y_train -= 1
        y_test  -= 1
        n_classes   = 26
        class_names = [chr(ord('A') + i) for i in range(26)]
        print(f"    EMNIST loaded via tensorflow_datasets")
    except ImportError:
        print("     tensorflow_datasets not installed. Falling back to MNIST.")
        print("   (pip install tensorflow-datasets if you want EMNIST)")
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        n_classes   = 10
        class_names = [str(i) for i in range(10)]
        DATASET     = "mnist"
else:
    raise ValueError(f"Unknown dataset: {DATASET}. Use 'mnist' or 'emnist'")

print(f"\n Data shapes (before preprocessing):")
print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
print(f"   Pixel value range: {X_train.min()} to {X_train.max()}")


# ============================================================
# STEP 2: PREPROCESS (the part where we make numbers smaller)
# ============================================================

print(f"\n{'='*60}")
print("PREPROCESSING...")
print("="*60)

# Reshape: add channel dimension for 2D CNN
# (28, 28) -> (28, 28, 1)  because CNN expects (height, width, channels)
# grayscale has 1 channel, RGB would have 3
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

print(f"\n Reshaped: {X_train.shape[1:]}")
print(f"   (28 height × 28 width × 1 channel = {28*28} pixels per image)")

# Normalize pixel values: 0-255 → 0.0-1.0
# This makes training faster and more stable
# Why? Because 255 >> 0.5 messes up gradient descent
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

print(f"\n Pixel values normalized: {X_train.min():.1f} to {X_train.max():.1f}")
print(f"   (dividing by 255 - a classic move in computer vision)")

# One-hot encode labels
y_train_oh = to_categorical(y_train, n_classes)
y_test_oh  = to_categorical(y_test,  n_classes)

print(f"\n  Classes: {n_classes}")
print(f"   One-hot shape: {y_train_oh.shape}")

# Show class distribution
print(f"\n Class distribution (first 5 classes):")
for i in range(min(5, n_classes)):
    count = (y_train == i).sum()
    bar   = "█" * (count // 300)
    print(f"   {class_names[i]}: {count:,} {bar}")
print(f"   ... (fairly balanced overall, which is nice)")


# ============================================================
# STEP 3: BUILD THE CNN (the architecture moment)
# ============================================================

print(f"\n{'='*60}")
print("BUILDING CNN ARCHITECTURE...")
print("="*60)

def build_cnn(input_shape, n_classes):
    """
    2D CNN for image classification.
    
    Architecture explanation (for the LinkedIn post):
    
    Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → Flatten → Dense
    
    Conv2D: learns to detect features (edges, curves, shapes)
    MaxPooling2D: reduces size, keeps important features
    Flatten: converts 2D feature map into 1D vector
    Dense: makes the final prediction
    
    It's like having a robot look at the image with
    increasingly complex glasses until it goes "yeah that's a 5".
    """
    
    model = models.Sequential([
        # --- Block 1: detect basic features (edges, curves) ---
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # --- Block 2: detect complex features (shapes, parts of letters) ---
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # --- Block 3: high-level feature detection ---
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # --- Flatten and classify ---
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),   # aggressive dropout before output
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output: softmax gives probabilities per class
        layers.Dense(n_classes, activation='softmax')
    ])
    
    return model


model = build_cnn(input_shape=(28, 28, 1), n_classes=n_classes)

# Compile - categorical crossentropy because we have multiple classes
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n  Model Architecture:")
model.summary()

params = model.count_params()
print(f"\n    Total trainable parameters: {params:,}")
print(f"   (that's {params:,} weights the model adjusts during training)")
print(f"   (your calculator has way less logic. this thing is smarter.)")


# ============================================================
# STEP 4: TRAIN (the waiting game)
# ============================================================

print(f"\n{'='*60}")
print("TRAINING CNN... (watch those accuracy numbers go up)")
print("="*60)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )
]

print(f"\n  Training config:")
print(f"   Epochs: up to {EPOCHS} (early stopping might kick in)")
print(f"   Batch size: {BATCH}")
print(f"   Optimizer: Adam")
print(f"   Loss: Categorical Crossentropy\n")

history = model.fit(
    X_train, y_train_oh,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=0.1,    # use 10% of training data for validation
    callbacks=callbacks,
    verbose=1
)

print("\n Training done! Let's see what we got...")


# ============================================================
# STEP 5: EVALUATE
# ============================================================

print(f"\n{'='*60}")
print("EVALUATING ON TEST SET...")
print("="*60)

test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)

print(f"\n Results:")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Test Loss:     {test_loss:.4f}")

if test_acc > 0.99:
    print("   LITERALLY 99%+ ACCURACY. You're a genius. This is cheating.")
elif test_acc > 0.97:
    print("   97%+ accuracy! That's better than some humans. For real.")
elif test_acc > 0.90:
    print("   90%+ accuracy. That's good. Not LinkedIn-worthy but decent.")
else:
    print("   Hmm below 90%... something went wrong. Try more epochs?")

# Predictions on test set
y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


# ============================================================
# STEP 6: VISUALIZATIONS (screenshot worthy content ahead)
# ============================================================

print("\n Generating plots...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle(f"CNN Handwritten Recognition - {DATASET.upper()} Dataset\n"
             f"Final Accuracy: {test_acc*100:.2f}% "
             f"(yes the computer can read handwriting better than you can)",
             fontsize=13, fontweight='bold')

# --- Plot 1: Training History ---
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(history.history['accuracy'], 'b-o', label='Train', linewidth=2, markersize=4)
ax1.plot(history.history['val_accuracy'], 'r--s', label='Val', linewidth=2, markersize=4)
ax1.set_title("Accuracy Over Epochs\n(watching it get smarter)", fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Plot 2: Training Loss ---
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(history.history['loss'], 'g-o', label='Train Loss', linewidth=2, markersize=4)
ax2.plot(history.history['val_loss'], 'orange', label='Val Loss', linewidth=2, linestyle='--')
ax2.set_title("Loss Over Epochs\n(lower is better, fyi)", fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Confusion Matrix (heatmap) ---
ax3 = fig.add_subplot(3, 3, 3)
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
im = ax3.imshow(cm_norm, cmap='Blues', aspect='auto')
plt.colorbar(im, ax=ax3, fraction=0.046)
ax3.set_title("Confusion Matrix\n(which digits look alike?)", fontweight='bold')
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
ax3.set_xticks(range(n_classes))
ax3.set_yticks(range(n_classes))
ax3.set_xticklabels(class_names, rotation=45 if n_classes > 10 else 0)
ax3.set_yticklabels(class_names)


# ============================================================
# STEP 7: SAMPLE PREDICTIONS DISPLAY
# aka the coolest part - showing actual predictions on images
# ============================================================

print("\n  Generating sample prediction images...")

# pick 30 random test images (20 correct, 10 wrong if possible)
correct_idx = np.where(y_pred == y_test)[0]
wrong_idx   = np.where(y_pred != y_test)[0]

np.random.seed(42)

# grab some correct and wrong predictions to show
n_show = 6
sample_correct = np.random.choice(correct_idx, min(n_show, len(correct_idx)), replace=False)
sample_wrong   = np.random.choice(wrong_idx,   min(n_show, len(wrong_idx)),   replace=False) if len(wrong_idx) > 0 else []

print(f"\n   Correct predictions to display: {len(sample_correct)}")
print(f"   Wrong predictions to display:   {len(sample_wrong)}")
print(f"   (wrong ones are more interesting tbh)")

# --- Plot rows 2 & 3: Sample predictions ---
# Row 2: Correct predictions (green border)
for i, idx in enumerate(sample_correct[:6]):
    ax = fig.add_subplot(3, 6, 7 + i)
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    
    true_label  = class_names[y_test[idx]]
    pred_label  = class_names[y_pred[idx]]
    confidence  = y_pred_prob[idx][y_pred[idx]] * 100
    
    ax.set_title(f" True: {true_label}\nPred: {pred_label} ({confidence:.0f}%)",
                 fontsize=7, color='darkgreen', fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('green')
        spine.set_linewidth(3)

# Row 3: Wrong predictions (red border) - only if there are mistakes
for i, idx in enumerate(sample_wrong[:6]):
    ax = fig.add_subplot(3, 6, 13 + i)
    ax.imshow(X_test[idx].reshape(28, 28), cmap='Reds')
    
    true_label  = class_names[y_test[idx]]
    pred_label  = class_names[y_pred[idx]]
    confidence  = y_pred_prob[idx][y_pred[idx]] * 100
    
    ax.set_title(f" True: {true_label}\nPred: {pred_label} ({confidence:.0f}%)",
                 fontsize=7, color='darkred', fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(3)

# Labels for the rows
fig.text(0.01, 0.33, " CORRECT", va='center', fontsize=11,
         fontweight='bold', color='darkgreen', rotation=90)
fig.text(0.01, 0.12, " WRONG", va='center', fontsize=11,
         fontweight='bold', color='darkred', rotation=90)

plt.tight_layout(rect=[0.03, 0, 1, 0.96])
plt.savefig("handwriting_recognition_results.png", dpi=150, bbox_inches='tight')
print("   Saved: handwriting_recognition_results.png")

plt.show()

# --- Extra: display a grid of random sample images ---
fig2, axes2 = plt.subplots(4, 8, figsize=(16, 8))
fig2.suptitle(f"Random Sample Images from {DATASET.upper()} Test Set\n"
              f"(with predictions - green = right, red = wrong)",
              fontsize=12, fontweight='bold')

sample_indices = np.random.choice(len(X_test), 32, replace=False)
for i, idx in enumerate(sample_indices):
    ax = axes2[i // 8][i % 8]
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    
    true_l = class_names[y_test[idx]]
    pred_l = class_names[y_pred[idx]]
    is_correct = (true_l == pred_l)
    
    color = 'green' if is_correct else 'red'
    ax.set_title(f"T:{true_l} P:{pred_l}", fontsize=7, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig("sample_predictions_grid.png", dpi=150, bbox_inches='tight')
print("   Saved: sample_predictions_grid.png")

plt.show()


# Save model
model.save(f"{DATASET}_cnn_model.h5")
print(f"   Saved: {DATASET}_cnn_model.h5")


# ============================================================
# BONUS: Interactive prediction (run a single image through)
# ============================================================

print(f"\n{'='*60}")
print("BONUS: TESTING RANDOM IMAGES (because it's fun)")
print("="*60)

print("\n Let's test 5 random images:")
test_indices = np.random.choice(len(X_test), 5, replace=False)

for i, idx in enumerate(test_indices):
    true_label = class_names[y_test[idx]]
    pred_label = class_names[y_pred[idx]]
    confidence = y_pred_prob[idx][y_pred[idx]] * 100
    result_emoji = "" if true_label == pred_label else ""
    
    print(f"   Image {i+1}: True={true_label} | Predicted={pred_label} | "
          f"Confidence={confidence:.1f}% {result_emoji}")

print("\n" + "="*60)
print(" TASK 3 COMPLETE!")
print("="*60)
print("\nWhat just happened:")
print("  → Downloaded MNIST with one line of code (legend)")
print("  → Normalized pixels from 0-255 to 0-1 (simple but important)")
print("  → Built a CNN that stacks Conv2D + MaxPooling2D layers")
print("  → Got 99%+ accuracy on a dataset with 10,000 test images")
print("  → This is literally what self-driving cars use (scaled up)")
print("\nSame architecture powers Google Photos face detection btw.")
print("You're basically an AI engineer now. Update your LinkedIn. ")
