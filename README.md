# Common Voice Exploratory Data Analysis

This project explores the [Mozilla Common Voice 13.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) dataset using Python, pandas, and seaborn. The goal is to better understand the structure, quality, and demographics of publicly available speech data as a foundation for building a custom, dialect-aware text-to-speech (TTS) model.

My ultimate goal is to build a custom, dialect-aware TTS model, and this exploration helped me understand the structure and diversity of real-world speech corpora.

---

## 📌 Project Goals

- Perform exploratory data analysis (EDA) on the `train.tsv` split of Common Voice 13.0
- Understand speaker demographics (age, gender, accent)
- Analyze sentence length distribution
- Identify potential biases and data quality issues
- Build intuition for data preprocessing needs in TTS pipelines

---

## Setup Instructions (Using pip)
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

---

## 📂 Project Structure

```bash
common-voice-eda/
├── data/
│   └── train.tsv              # Sourced from Common Voice 13.0 (English)
├── analysis.ipynb            # Main Jupyter Notebook for EDA
├── main.py   				  # Reusable script version
├── plots/                    # Saved charts and figures
└── README.md                 # You're here!
```

---

## 🛠️ Tools Used

- Python 3.13
- pandas
- matplotlib
- seaborn
- Jupyter Notebook

---

## Date Source & License
This project includes data from Mozilla Common Voice 13.0.
Data is licensed under CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/).
Source: https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0