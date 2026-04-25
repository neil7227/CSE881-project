# CSE881-project

Overview
Classifies AEROBIC, ANAEROBIC, and STRESS states using Empatica E4 physiological signals (EDA, HR, TEMP, ACC) with machine learning.

Dataset
-100 participants (31 aerobic, 32 anaerobic, 37 stress)
-Signals: EDA (4 Hz), HR (1 Hz), TEMP (4 Hz), ACC (32 Hz)

Method
-Preprocessing: alignment, filtering, normalization
-Features: 16 statistical features (mean, std, median, P95 × 4 signals)
-Models: LR, KNN, DT, SVM (RBF), MLP
