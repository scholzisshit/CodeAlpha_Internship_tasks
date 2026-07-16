#  Task 3: Handwritten Character Recognition  
### *"Teaching pixels to have meaning since MNIST dropped in 1998"*

---

## What This Does
Builds a CNN that looks at 28×28 pixel images of handwritten digits  
and correctly identifies them with **~99% accuracy**.  
Yeah. 99%. On a laptop. Without a PhD. We live in incredible times.

## How to Run

```bash
# Step 1: install dependencies
pip install -r requirements.txt

# Step 2: run (dataset downloads automatically, no Kaggle needed!)
python handwriting_recognition.py
```

That's it. Seriously. TensorFlow downloads MNIST for you.  
The future is now old man.

## Want Letters Instead of Digits?
Change line 56 in the script:
```python
DATASET = "emnist"   # was "mnist"
```
Then install tensorflow-datasets:
```bash
pip install tensorflow-datasets
```

## Output Files
- `handwriting_recognition_results.png` → training curves, confusion matrix, sample predictions
- `sample_predictions_grid.png` → grid of 32 images with predictions
- `mnist_cnn_model.h5` → the trained model

## Architecture Breakdown
```
Input Image (28×28×1 pixels)
    ↓
[Block 1] Conv2D(32) → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓  detects: edges, basic curves
[Block 2] Conv2D(64) → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓  detects: corners, partial shapes  
[Block 3] Conv2D(128) → Dropout(0.25)
    ↓  detects: full digit shapes
Flatten → Dense(256) → Dense(128) → Dense(10, softmax)
    ↓
Output: probabilities for digits 0-9
```

## Why It Works So Well
- **Conv2D** learns which pixel patterns matter (corners, curves, lines)
- **MaxPooling2D** reduces size while keeping important info
- **BatchNormalization** keeps training stable
- **Dropout** prevents memorizing the training data (overfitting)
- **Softmax** gives a clean probability for each class

## Dataset Stats
| Dataset | Images | Classes | Accuracy |
|---------|--------|---------|----------|
| MNIST | 70,000 | 10 (digits) | ~99% |
| EMNIST | 145,600 | 26 (letters) | ~90% |

---
*CodeAlpha Internship | Task 3 | ML Domain*
