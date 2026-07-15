# 🎤 Task 2: Emotion Recognition from Speech
### *"Making a computer feel feelings (it can't, but we fake it)"*

---

## What This Does
Listens to audio clips and predicts the emotion: **happy, sad, angry, fearful, neutral, disgust**.  
Like a therapist but cheaper and less judgy.

## Setup & Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Get the RAVDESS Dataset (important!)
1. Go to: https://zenodo.org/record/1188976
2. Download the `Audio_Speech_Actors_01-24.zip`
3. Extract it in this folder → you should see `Audio_Speech_Actors_01-24/`

> ⚠️ **No dataset?** The script still runs with random dummy data.  
> It won't learn anything real, but it'll train and produce graphs.  
> Technically it "works" for demo purposes. 🤫

### Step 3: Run
```bash
python emotion_recognition.py
```

## Output Files
- `emotion_recognition_results.png` → training curves + confusion matrix
- `emotion_cnn_model.h5` → the saved model (flex this)

## Architecture
```
Input (200 timesteps × 40 MFCCs)
    ↓
Conv1D(64) → BatchNorm → MaxPooling → Dropout
    ↓
Conv1D(128) → BatchNorm → MaxPooling → Dropout
    ↓
Conv1D(256) → GlobalAveragePooling → Dropout
    ↓
Dense(256) → Dense(128) → Dense(6, softmax)
    ↓
Output: [neutral, happy, sad, angry, fearful, disgust]
```

## What are MFCCs?
Mel-Frequency Cepstral Coefficients.  
Basically: audio → FFT → Mel scale → DCT → numbers the CNN can actually use.  
You don't need to understand the math. Just know it converts sound to features. Trust the process.

## Expected Accuracy
- With real RAVDESS data: **~65-80%** accuracy
- With dummy random data: **~16%** (random chance with 6 classes lol)

---
*CodeAlpha Internship | Task 2 | ML Domain*
