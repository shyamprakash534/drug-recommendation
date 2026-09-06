# Drug Recommendation System

A Flask-based machine-learning application that demonstrates end-to-end model training, inference, treatment-rule logic, and a web interface for educational exploration of drug recommendations.

> **Important:** This project is for educational and software-engineering demonstration purposes only. It is **not medical advice** and must not be used to select or dose medication for a real patient. Medication decisions require a qualified clinician and patient-specific medical information.

## ✨ What the project demonstrates

- Python + Flask web application development
- Machine-learning model training with scikit-learn
- Random Forest classification for drug prediction
- Gradient Boosting for dosage prediction
- Label encoding and train/test splitting
- Model persistence with Joblib
- Dataset-driven inference
- Rule-based treatment scheduling and safety checks
- Matplotlib-based visual/report generation
- PDF generation with FPDF
- Gunicorn deployment configuration

## 🧠 ML Pipeline

```text
Patient / condition inputs
          ↓
Feature encoding
          ↓
Random Forest ─────────→ Drug prediction
          ↓
Gradient Boosting ─────→ Dosage prediction
          ↓
Rule-based checks
          ↓
Treatment information / educational output
```

The application trains models from the dataset when the persisted model files are not available, then loads the generated artifacts for inference.

## 🛠️ Technology Stack

| Area | Technology |
| --- | --- |
| Language | Python |
| Web | Flask, HTML templates |
| ML | scikit-learn |
| Data | pandas, NumPy |
| Models | Random Forest, Gradient Boosting |
| Persistence | Joblib |
| Visualization | Matplotlib |
| Reports | FPDF |
| Deployment | Gunicorn / Procfile |

## 📁 Project Structure

```text
drug-recommendation/
├── app.py
├── dataset/
│   └── real_drug_dataset.csv
├── templates/
├── requirements.txt
├── runtime.txt
├── Procfile
└── README.md
```

Runtime-generated model artifacts are intentionally kept out of source control.

## 🚀 Run Locally

```bash
git clone https://github.com/shyamprakash534/drug-recommendation.git
cd drug-recommendation
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

For production-style execution:

```bash
gunicorn app:app
```

## 🔐 Configuration

Set a strong `FLASK_SECRET_KEY` environment variable for deployed environments. Never commit production secrets to the repository.

Example:

```bash
export FLASK_SECRET_KEY="replace-with-a-random-secret"
```

## 📊 Engineering Notes

The project intentionally combines ML inference with explicit rule-based logic. This makes it useful for demonstrating how a software application can integrate trained models, deterministic business rules, data processing, and a web UI rather than treating an ML model as the entire application.

## ⚠️ Responsible Use

The recommendations, dosage values, treatment schedules, and drug information in this repository are implementation/demo data and should **not** be interpreted as clinical guidance. Do not use the application to diagnose, prescribe, substitute medication, or determine treatment for yourself or another person.

## 👨‍💻 Author

**Shyam Prakash**

- GitHub: https://github.com/shyamprakash534
- LinkedIn: https://www.linkedin.com/in/shyam-prakash-269a74208/

---

⭐ Built as an AI/ML + Python engineering project to demonstrate practical application development.
