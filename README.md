# 🎬 IMDB Sentiment Analysis using RNN

> NLP + Deep Learning project to classify movie reviews as Positive or Negative using a Recurrent Neural Network (RNN) built with PyTorch.

---

## 📌 Problem Statement

Given an IMDB movie review (text), predict whether the sentiment is **Positive** or **Negative** — a binary text classification problem.

**Applications:** Product reviews, social media monitoring, customer feedback analysis

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Dataset | IMDB Movie Reviews |
| Total Reviews | 50,000 |
| Classes | Positive / Negative (balanced) |
| Train/Test Split | 80% / 20% |

---

## 🔧 Text Preprocessing Pipeline

Raw text → Clean text → Numbers → Model

| Step | What it does |
|------|-------------|
| Lowercase | `"The Movie"` → `"the movie"` |
| Remove URLs | `https://...` → removed |
| Remove Punctuation | `"good!"` → `"good"` |
| Remove HTML tags | `<br>` → removed |
| Remove Stopwords | `"the", "is", "a"` → removed |
| Stemming | `"running"` → `"run"` |
| TF-IDF Vectorization | Text → 5000 numerical features |
| Label Encoding | `"positive"` → `1`, `"negative"` → `0` |

---

## 🏗️ Model Architecture

```
Input (5000 TF-IDF features)
    │
Unsqueeze → (batch, 1, 5000)
    │
RNN Layer (hidden_size=128, num_layers=1)
    │
Last time step output
    │
Linear(128 → 1)
    │
Sigmoid → Probability (0 to 1)
    │
Output: Positive (>0.5) / Negative (≤0.5)
```

| Component | Details |
|-----------|---------|
| RNN | 1 layer, hidden_size=128 |
| Loss Function | BCELoss (Binary Cross Entropy) |
| Optimizer | Adam |
| Epochs | 10 |

---

## 🚀 How to Run

### Google Colab (Recommended)
1. Upload `RNN_for_sentimentanalysis.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload `IMDB Dataset.csv` when prompted
3. `Runtime → Run all`

### Local
```bash
pip install torch pandas scikit-learn nltk
jupyter notebook RNN_for_sentimentanalysis.ipynb
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![NLTK](https://img.shields.io/badge/NLTK-3.x-yellow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-blue)

---

## 📚 Key Concepts Covered

- Text preprocessing (cleaning, tokenization, stemming, stopword removal)
- TF-IDF vectorization for text → numbers
- RNN architecture for sequential data
- Binary classification using BCELoss + Sigmoid
- Training loop (forward pass, backpropagation, weight update)

---

*Built with PyTorch — NLP + Deep Learning*
