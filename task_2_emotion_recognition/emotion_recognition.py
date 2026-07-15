# ============================================================
# TASK 2: Emotion Recognition from Speech
# aka "teaching a computer to feel things (it can't, but we try)"
# ============================================================
#
# What we're doing here:
#   - Load audio files from RAVDESS dataset
#   - Extract MFCCs (Mel-Frequency Cepstral Coefficients)
#     (sounds fancy, it's basically audio fingerprints)
#   - Throw it into a 1D CNN and hope for the best
#
# Dataset: RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song
# Download from: https://zenodo.org/record/1188976
# Extract the Audio_Speech_Actors_01-24 folder here
#
# NOTE: If you don't have the dataset, we generate dummy data.
# It won't learn anything useful, but it'll run and that's what matters
# for submission right? 😂 (jk actually download the dataset)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")  # shhh

# check if librosa is installed
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
    print("✅ librosa found! We're cooking.")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("❌ librosa not installed! Run: pip install librosa")
    print("   (or just pip install -r requirements.txt you absolute legend)")

# tensorflow - the big boy library
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, Dense, Dropout,
        Flatten, BatchNormalization, GlobalAveragePooling1D
    )
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix, classification_report
    print("✅ TensorFlow version:", tf.__version__)
    print("   (if it's below 2.0, we're gonna have a bad time)")
except ImportError:
    print("❌ TensorFlow not found! Run: pip install tensorflow")
    raise


# ============================================================
# EMOTION LABELS (RAVDESS uses numeric codes, so we decode them)
# ============================================================

# RAVDESS filename format: 03-01-06-01-02-01-12.wav
# The 3rd number (position [2]) is the emotion:
# 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# we'll focus on these 6 emotions (dropping calm and surprised for cleaner results)
# (calm and surprised are kinda mid emotions anyway)
TARGET_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

print(f"\n🎭 Targeting {len(TARGET_EMOTIONS)} emotions: {TARGET_EMOTIONS}")


# ============================================================
# STEP 1: LOAD AND EXTRACT MFCC FEATURES
# ============================================================

def extract_features(file_path, n_mfcc=40, max_len=200):
    """
    Extract MFCC features from an audio file.
    
    MFCCs are basically the audio equivalent of pixels for images.
    The model learns patterns from these instead of the raw waveform
    because raw waveforms are just too much data for our poor CPU.
    
    Args:
        file_path: path to .wav file
        n_mfcc: number of MFCC coefficients (40 is the sweet spot)
        max_len: pad/truncate to this length (so all arrays are same size)
    
    Returns:
        numpy array of shape (max_len, n_mfcc)
    """
    if not LIBROSA_AVAILABLE:
        return None
    
    try:
        # load audio, sample rate 22050 Hz by default
        audio, sr = librosa.load(file_path, sr=22050)
        
        # Extract MFCCs - the bread and butter of audio ML
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # transpose so shape is (time_steps, n_mfcc) - easier for CNN
        mfcc = mfcc.T
        
        # Pad or truncate to fixed length
        if len(mfcc) < max_len:
            # pad with zeros at the end
            pad_width = max_len - len(mfcc)
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            # just cut it off
            mfcc = mfcc[:max_len]
        
        return mfcc
    
    except Exception as e:
        print(f"   Couldn't load {file_path}: {e}")
        return None


def load_ravdess_data(data_folder="Audio_Speech_Actors_01-24"):
    """
    Walk through the RAVDESS dataset folder and load all audio files.
    Expects folder structure: Actor_XX/03-01-XX-XX-XX-XX-XX.wav
    """
    
    print(f"\n📂 Looking for RAVDESS data in: {data_folder}/")
    
    X = []
    y = []
    
    if not os.path.exists(data_folder):
        print(f"   ⚠️  Folder not found! Generating dummy data instead...")
        print(f"   (Please download RAVDESS from https://zenodo.org/record/1188976)\n")
        return None, None
    
    found_files = 0
    skipped_files = 0
    
    for actor_folder in sorted(os.listdir(data_folder)):
        actor_path = os.path.join(data_folder, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        
        for file in sorted(os.listdir(actor_path)):
            if not file.endswith(".wav"):
                continue
            
            # parse emotion from filename
            parts = file.split("-")
            if len(parts) < 3:
                continue
            
            emotion_code = parts[2]
            emotion = EMOTION_MAP.get(emotion_code)
            
            if emotion not in TARGET_EMOTIONS:
                skipped_files += 1
                continue
            
            file_path = os.path.join(actor_path, file)
            features = extract_features(file_path)
            
            if features is not None:
                X.append(features)
                y.append(emotion)
                found_files += 1
    
    print(f"   ✅ Loaded {found_files} audio files")
    print(f"   ⏭️  Skipped {skipped_files} files (wrong emotion category)")
    
    return np.array(X), np.array(y)


# Try to load real data first, fall back to dummy
X, y = load_ravdess_data()

if X is None:
    print("🎲 Generating dummy MFCC data for demonstration...")
    print("   Shape will be: (samples, 200, 40) - [batch, time_steps, mfcc_features]")
    print("   The model will train but won't learn anything useful with random data")
    print("   (story of my life honestly)\n")
    
    n_samples = 1440   # RAVDESS has ~1440 speech samples
    n_timesteps = 200
    n_mfccs = 40
    
    # Generate random MFCC-like data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_timesteps, n_mfccs).astype(np.float32)
    
    # Random labels from our target emotions
    y = np.random.choice(TARGET_EMOTIONS, n_samples)
    
    print(f"   Generated {n_samples} fake samples. Please don't submit this to Kaggle.")


# ============================================================
# STEP 2: PREPROCESS (encode labels, split data)
# ============================================================

print("\n" + "="*60)
print("PREPROCESSING...")
print("="*60)

# Encode emotion strings to integers
# "happy" -> 0, "angry" -> 1, etc.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)

print(f"\n🏷️  Emotion classes ({n_classes} total):")
for idx, emotion in enumerate(le.classes_):
    count = (y_encoded == idx).sum()
    print(f"   {idx}: {emotion:12s} - {count} samples")

# Convert to one-hot for categorical crossentropy
y_onehot = to_categorical(y_encoded, n_classes)

print(f"\n📐 Data shapes:")
print(f"   X shape: {X.shape}  (samples, time_steps, mfcc_features)")
print(f"   y shape: {y_onehot.shape}  (samples, n_classes)")

# Normalize features (z-score normalization per feature)
X_mean = X.mean(axis=(0, 1), keepdims=True)
X_std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
X_norm = (X - X_mean) / X_std

# Train / Val / Test split - 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X_norm, y_onehot, test_size=0.30, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"\n✂️  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# ============================================================
# STEP 3: BUILD THE CNN MODEL
# ============================================================

print("\n" + "="*60)
print("BUILDING THE CNN... (architecture time baby)")
print("="*60)

def build_cnn_model(input_shape, n_classes):
    """
    1D CNN for sequential audio data.
    
    Why 1D CNN and not 2D? Because our data is (time_steps, features),
    not a 2D image. 1D conv slides across the time dimension.
    
    Why not LSTM? LSTMs are slower and for this task CNNs work fine.
    Plus I don't want to wait 3 hours for training. We have memes to scroll.
    """
    
    model = Sequential([
        # Block 1 - learn local patterns in the audio
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),  # drop 30% of neurons to prevent overfitting
        
        # Block 2 - learn higher-level patterns
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Block 3 - even higher patterns (like "this sounds angry")
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling1D(),  # cleaner than Flatten for sequences
        Dropout(0.4),
        
        # Fully connected layers - the brain of the operation
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        # Output layer - one neuron per emotion
        Dense(n_classes, activation='softmax')
    ])
    
    return model


input_shape = (X_train.shape[1], X_train.shape[2])
model = build_cnn_model(input_shape, n_classes)

model.compile(
    optimizer='adam',               # adam optimizer - the reliable choice
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🏗️  Model Architecture:")
model.summary()

total_params = model.count_params()
print(f"\n   Total parameters: {total_params:,}")
print(f"   ({total_params:,} numbers the model needs to learn. Yeah it's a lot.)")


# ============================================================
# STEP 4: TRAIN THE MODEL
# ============================================================

print("\n" + "="*60)
print("TRAINING... (go make a coffee, this takes a minute)")
print("="*60)

# Callbacks to make training smarter
callbacks = [
    # Stop early if val_loss doesn't improve (saves time)
    EarlyStopping(
        monitor='val_loss',
        patience=10,              # wait 10 epochs before giving up
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when stuck (like pressing harder on the gas)
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,               # halve the learning rate
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    epochs=50,                    # max 50 epochs (early stopping will kick in earlier)
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete! Let's see how bad it is...")


# ============================================================
# STEP 5: EVALUATE THE MODEL
# ============================================================

print("\n" + "="*60)
print("EVALUATING...")
print("="*60)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")
print(f"   Test Loss: {test_loss:.4f}")

if test_acc > 0.70:
    print("   Not bad! That's above 70% - intern of the month material 💪")
elif test_acc > 0.50:
    print("   Over 50%! Better than random guessing. We take those wins.")
else:
    print("   Under 50%... you're just generating random data aren't you 👀")

# Predictions
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))


# ============================================================
# STEP 6: VISUALIZATIONS (the part that makes the report look good)
# ============================================================

print("\n📸 Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Emotion Recognition from Speech - Model Results\n"
             "(Teaching machines to feel things since 2024)",
             fontsize=14, fontweight='bold')

# --- Plot 1: Training History - Accuracy ---
ax1 = axes[0][0]
ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#3498db', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#e74c3c', linewidth=2, linestyle='--')
ax1.set_title("Training vs Validation Accuracy\n(watching the model get smarter... hopefully)", fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Plot 2: Training History - Loss ---
ax2 = axes[0][1]
ax2.plot(history.history['loss'], label='Train Loss', color='#2ecc71', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val Loss', color='#f39c12', linewidth=2, linestyle='--')
ax2.set_title("Training vs Validation Loss\n(lower = better, that's literally it)", fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Confusion Matrix ---
ax3 = axes[1][0]
cm = confusion_matrix(y_true, y_pred)
# normalize it so it's percentages
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=ax3,
    linewidths=0.5
)
ax3.set_title("Confusion Matrix (Normalized)\n(which emotions confuse the model the most?)", fontweight='bold')
ax3.set_xlabel("Predicted Emotion")
ax3.set_ylabel("Actual Emotion")
ax3.tick_params(axis='x', rotation=45)

# --- Plot 4: Per-class Accuracy Bar Chart ---
ax4 = axes[1][1]
per_class_acc = cm_normalized.diagonal()
emotion_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
bars = ax4.bar(le.classes_, per_class_acc, color=emotion_colors[:n_classes], edgecolor='black', linewidth=0.8)
ax4.set_title("Per-Emotion Accuracy\n(some emotions are harder than others)", fontweight='bold')
ax4.set_ylabel("Accuracy")
ax4.set_ylim(0, 1.15)
ax4.tick_params(axis='x', rotation=30)
ax4.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, per_class_acc):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{acc:.1%}", ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig("emotion_recognition_results.png", dpi=150, bbox_inches='tight')
print("   Saved: emotion_recognition_results.png")

plt.show()

# Save the model (optional but looks professional)
model.save("emotion_cnn_model.h5")
print("   Saved: emotion_cnn_model.h5 (the trained brain)")

print("\n" + "="*60)
print("✅ TASK 2 COMPLETE!")
print("="*60)
print("\nFun facts you learned today:")
print("  → MFCCs convert audio to numbers that CNNs can understand")
print("  → Humans confuse neutral/calm too, so the model isn't dumb")
print("  → 1D CNN > LSTM for this task (faster + often better)")
print("  → EarlyStopping saves us from training for 10 hours")
print("\nThis is legitimately cool tech. Siri uses this stuff.")
print("Go brag on LinkedIn. You deserve it. 🎤")
