#  CodeAlpha ML Internship Tasks
### *"Three ML projects that'll make your resume actually interesting"*

> Built for CodeAlpha Internship | Machine Learning Domain  
> By someone who definitely didn't procrastinate on these

---

##  Folder Structure

```
internship_tasks/
│
├── task_1_credit_scoring/
│   ├── credit_scoring.py       ← main script
│   ├── requirements.txt        ← install these first
│   └── README.md
│
├── task_2_emotion_recognition/
│   ├── emotion_recognition.py  ← main script  
│   ├── requirements.txt        ← includes librosa
│   └── README.md
│
├── task_3_handwriting_recognition/
│   ├── handwriting_recognition.py  ← main script
│   ├── requirements.txt            ← tensorflow + sklearn
│   └── README.md
│
└── README.md                   ← you are here
```

---

##  Quick Start

### Task 1 - Credit Scoring (easiest, start here)
```bash
cd task_1_credit_scoring
pip install -r requirements.txt
python credit_scoring.py
```

### Task 2 - Emotion Recognition (needs RAVDESS dataset)
```bash
cd task_2_emotion_recognition
pip install -r requirements.txt
# download RAVDESS from https://zenodo.org/record/1188976
# extract Audio_Speech_Actors_01-24/ here
python emotion_recognition.py
```

### Task 3 - Handwriting Recognition (easiest setup, most impressive results)
```bash
cd task_3_handwriting_recognition
pip install -r requirements.txt
python handwriting_recognition.py   # downloads MNIST automatically!
```

---

##  What Each Task Covers

| Task | Model | Dataset | Key Concepts |
|------|-------|---------|-------------|
| **Credit Scoring** | Random Forest + Logistic Regression | Credit Risk CSV | Precision, Recall, F1, ROC-AUC |
| **Emotion Recognition** | 1D CNN | RAVDESS audio | MFCCs, audio processing, confusion matrix |
| **Handwriting Recognition** | 2D CNN | MNIST/EMNIST | Conv2D, MaxPooling, image normalization |

---

##  Environment Setup (do this once)

```bash
# recommended: use a virtual environment
python -m venv venv
source venv/bin/activate      # mac/linux
# venv\Scripts\activate       # windows (you poor soul)

# then install task-specific requirements in each folder
```

---

##  Expected Results

| Task | Metric | Expected |
|------|--------|----------|
| Credit Scoring | ROC-AUC | 0.85 - 0.95 |
| Emotion Recognition | Accuracy | 65% - 80% |
| Handwriting Recognition | Accuracy | 98% - 99.5% |

---

##  Troubleshooting

**"Module not found" errors?**  
→ Did you `pip install -r requirements.txt`? Do that first.

**librosa installation failing?**  
→ Try `pip install librosa soundfile` separately.

**TensorFlow not finding GPU?**  
→ It'll use CPU, it's fine. Just slower. Make a coffee.

**Credit dataset not found?**  
→ The script auto-generates synthetic data. You're fine.

---

##  Notes for Submission
- All scripts are heavily commented (because we care about readability apparently)
- Each task produces output PNG files for screenshots
- Models are saved as `.h5` files after training
- Make sure to actually download the RAVDESS dataset for Task 2 (the fake data won't cut it)

---

*CodeAlpha Internship | Machine Learning | 2026*  
*"Machine learning is just statistics in a trenchcoat" - some professor probably*
