# Vigyaan

Vigyaan is an all-in-one application for rapid exploratory data analysis, interactive visualizations, and baseline machine learning model training (Logistic Regression, SVC, RandomForest) with downloadable trained artifacts and preprocessing pipeline. It is built using Streamlit + FastAPI.

## ✨ Features

- CSV upload with automatic temporary storage
- Automatic feature / label detection
- Data summary: file info, schema, statistical description
- Visualizations (Plotly): scatter, histogram, line, correlation heatmap, box, pair, area
- Data quality & profiling panels (missing values, dtypes, categorical vs numerical split)
- One-click model training (LogisticRegression, SVC, RandomForestClassifier)
- Performance comparison (accuracy, F1) + charts
- Download trained models and preprocessing artifacts (scaler, imputer, feature names)
- Built-in minimal model usage template snippet
- Rate limiting via slowapi (internal API)
- Single-container deployment (Streamlit UI + internal FastAPI backend)
- Cloud Run / container-friendly (respects dynamic $PORT)

## 🏗 Architecture

```
+--------------------+            (internal HTTP, localhost only)
|  Streamlit Frontend | <--------------------------------+
|  UI + user session  |                                |
+--------------------+                                v
									  +---------------------------+
									  | FastAPI Backend (internal)|
									  | Data summary, plotting,   |
									  | ML training, prediction   |
									  +---------------------------+
```

- Both processes started by `run_all.sh`
- FastAPI bound to 127.0.0.1 (`BACKEND_PORT`, default 8000) and NOT exposed externally
- Streamlit exposed on `$PORT` or 8501 locally
- Models + preprocessing artifacts stored in `/app/models`

## 📂 Project Structure (key paths)

```
streamlit_frontend/
	app.py                 # Orchestrates layout, sidebar, and tabs
	api_client.py          # HTTP client for backend endpoints
	session_state.py       # Centralized Streamlit session state setup
	config.py              # Streamlit page config & constants
	views/
		overview.py          # Data overview (preview, file info, describe)
		visualizations.py    # All charts (scatter, hist, line, corr, box, pair, area)
		analysis.py          # Data analysis (missing, types, feature analysis)
		machine_learning.py  # Train, metrics, downloads, usage snippet

src/api/main.py          # FastAPI app + routers registration
src/api/routes/          # API endpoints (csv_file, data_summary, data_science, machine_learning)
src/machinelearning/     # Training & prediction pipeline code
models/                  # Persisted trained model and preprocessing artifacts
run_all.sh               # Supervisor script launching both services
Dockerfile               # Single-image container build
requirements.txt / pyproject.toml
```

## 🚀 Quick Start (Local, no Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
export API_BASE_URL="http://localhost:8000"  # optional; script sets it
./run_all.sh
```

Then open: http://localhost:8501

## 🐳 Docker (Single Container)

```bash
docker build -t vigyaan .
docker run -p 8501:8501 vigyaan
```

Open: http://localhost:8501

### Environment Variables

| Variable        | Purpose                                   | Default |
|-----------------|--------------------------------------------|---------|
| BACKEND_PORT    | Internal FastAPI port                      | 8000    |
| FRONTEND_PORT   | Streamlit external port (overrides $PORT)  | 8501    |
| PORT            | Cloud provider injected port (Cloud Run)   | (unset) |
| API_BASE_URL    | Derived automatically by `run_all.sh`      |         |

Cloud Run sets `PORT`; `run_all.sh` makes Streamlit listen there automatically.

## 🧠 Model Training Pipeline

For each model:
1. Load CSV
2. Impute missing values (mean for numeric, mode for categorical)
3. Encode categorical (label encoding)
4. Scale numerical features (StandardScaler)
5. Train models (LogisticRegression, SVC, RandomForestClassifier)
6. Persist: model pickles + `scaler.pkl`, `imputer.pkl`, `feature_names.pkl`
7. Return metrics (accuracy, f1_score)

## 📥 Downloaded Artifacts

| File                    | Description                                |
|-------------------------|--------------------------------------------|
| LogisticRegression.pkl  | Trained Logistic Regression model          |
| SVC.pkl                 | Trained Support Vector Classifier          |
| RandomForestClassifier.pkl | Trained Random Forest model           |
| scaler.pkl              | StandardScaler fitted on training data     |
| imputer.pkl             | Imputer object (numeric + categorical)     |
| feature_names.pkl       | Ordered feature names used in training     |

## 🛠 Using a Downloaded Model (Example Snippet)

```python
import os, pickle, numpy as np

def predict(features: dict, model_path: str):
	d = os.path.dirname(model_path) or "."
	model = pickle.load(open(model_path, "rb"))
	scaler = pickle.load(open(os.path.join(d, "scaler.pkl"), "rb"))
	imputer = pickle.load(open(os.path.join(d, "imputer.pkl"), "rb"))
	try:
		names = pickle.load(open(os.path.join(d, "feature_names.pkl"), "rb"))
	except Exception:
		names = list(features.keys())
	ordered = [features.get(n) for n in names]
	X = np.array(ordered).reshape(1, -1)
	X = scaler.transform(imputer.transform(X))
	pred = model.predict(X)[0]
	probs = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
	return {"prediction": pred, "probabilities": probs}

res = predict({"feature1": 5.1, "feature2": 3.5, "feature3": 1.4}, "./LogisticRegression.pkl")
print(res)
```

## 🔌 API (Internal Only)

Base URL (inside container / local): `http://localhost:8000`

| Method | Endpoint                               | Purpose                               |
|--------|-----------------------------------------|---------------------------------------|
| GET    | /csv_file/upload_csv (multipart)        | (Handled via Streamlit form)          |
| GET    | /csv_file/extract_features              | Detect features / label               |
| GET    | /data_summary/file_info                 | File size / name                      |
| GET    | /data_summary/data_info                 | Pandas info text                      |
| GET    | /data_summary/data_description          | Statistical describe()                |
| GET    | /data_science/scatter_plot              | Scatter data (two features)           |
| GET    | /data_science/histogram_plot            | Histogram numeric data                |
| GET    | /data_science/line_plot                 | Line chart data                       |
| GET    | /data_science/correlation_matrix        | Correlation matrix                    |
| GET    | /data_science/box_plot                  | Single-feature distribution           |
| GET    | /data_science/pair_plot                 | Pairwise numeric sample               |
| GET    | /data_science/area_plot                 | Area plot data                        |
| GET    | /machine_learning/train                 | Train models                          |
| GET    | /machine_learning/download?model_name=  | Download model / preprocessing file   |
| POST   | /machine_learning/predict               | Predict given model + feature values  |
| GET    | /health                                 | Backend health                        |

> Note: Backend not exposed publicly in container runtime; endpoints accessed through Streamlit via `requests`.

## 🧪 Rate Limiting

Configured with `slowapi` (e.g., 20/minute on health and root). Adjust limits in `src/api/main.py`.

## 🧹 Housekeeping / Temp Files

- Uploaded CSVs stored as temporary files (not persisted long-term)
- Consider adding a cron / background cleanup if deploying long-running multi-user instance
- Model directory may grow; implement retention or manual pruning for production

## 🔐 Security Notes

- No auth layer currently; add API keys or OAuth if exposing externally
- Internal FastAPI binding prevents direct external access in single-container mode
- Remove unused database / auth code if not needed to slim image & dependencies

## 🗺 Roadmap Ideas

- Direct in-process calls (remove HTTP hop) for extra speed
- Add model registry & versioning
- Add SHAP / feature importance visualization
- Async task queue for long-running training
- User auth + multi-tenant storage

## 🤝 Contributions

Feel free to open issues or submit PRs with improvements (tests, docs, features).

## 📄 License

Licensed under the MIT License. See `LICENSE` file for full text.

---
Happy experimenting! 🚀
