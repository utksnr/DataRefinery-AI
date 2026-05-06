# DataRefinery AI | Preprocess Engine

**DataRefinery AI** is an autonomous data preprocessing pipeline designed to transform raw datasets into "Golden Standard" formats optimized for Machine Learning models. This project integrates complex statistical transformations and data engineering workflows into a seamless, high-performance interface.

## 🚀 Key Technical Features

- **Yeo-Johnson Gaussian Normalization:** Automatically forces data into a normal distribution to enhance model predictive power.
- **Temporal Synthesis:** Detects date-time columns and synthesizes meaningful categorical and numerical features.
- **SOTA Feature Selection:** Utilizes Random Forest importance metrics to eliminate noise and retain high-impact variables.
- **Privacy Guard (GDPR Compliance):** Automatically detects and hashes sensitive Personally Identifiable Information (PII).
- **Memory Optimization:** Implements automated data-type downcasting to minimize memory footprint for large datasets.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI
- **Data Science:** Pandas, NumPy, Scikit-learn, SciPy
- **Frontend:** HTML5, Tailwind CSS (Glassmorphism UI)
- **Deployment:** Uvicorn

## 📦 Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/DataRefinery-AI.git](https://github.com/yourusername/DataRefinery-AI.git)
