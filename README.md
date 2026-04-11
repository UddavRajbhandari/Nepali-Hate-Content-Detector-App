# Nepali Hate Content Detector — Web Application

> Production web interface for Nepali hate speech classification.  
> Built with **FastAPI** backend + **React/Vite** frontend, powered by **XLM-RoBERTa-large** fine-tuned on Nepali social media data.

[![Backend](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb?logo=react)](https://vitejs.dev)
[![Model](https://img.shields.io/badge/model-XLM--RoBERTa--large-orange?logo=huggingface)](https://huggingface.co/UDHOV/xlm-roberta-large-nepali-hate-classification)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## 📎 Related Repository

This repository contains **only the web application** (API + frontend).

The full research project — including Jupyter notebooks, training pipeline, dataset analysis, model evaluation, and the original Streamlit demo — lives in the companion repository:

**[→ uddavrajbhandari/major-project](https://github.com/UddavRajbhandari/major-project)**

| What | Where |
|---|---|
| Model training & fine-tuning | `major-project` repo |
| Dataset & EDA notebooks | `major-project` repo |
| Streamlit demo app | `major-project` repo |
| Preprocessing & XAI scripts | Both repos (`scripts/` folder) |
| FastAPI backend | **This repo** |
| React frontend | **This repo** |
| Vercel / Railway deployment | **This repo** |

---

## 🗂️ Project Structure

```
nepali-hate-detector-app/
├── backend/                        FastAPI application
│   ├── app/
│   │   ├── main.py                 App entry point, CORS, lifespan
│   │   ├── services/
│   │   │   └── model_service.py    Singleton model + XAI methods
│   │   ├── routers/
│   │   │   ├── predict.py          POST /api/predict
│   │   │   ├── explain.py          POST /api/explain/{lime|shap|captum}
│   │   │   ├── batch.py            POST /api/batch  (streaming NDJSON)
│   │   │   ├── history.py          GET/DELETE /api/history
│   │   │   ├── status.py           GET /api/status
│   │   │   └── analyze.py          POST /api/analyze
│   │   ├── models/schemas.py       Pydantic v2 request/response schemas
│   │   └── utils/history.py        Thread-safe JSONL history store
│   ├── requirements.txt
│   └── Dockerfile
│
├── scripts/                        Preprocessing & XAI (shared with major-project)
│   ├── transformer_data_preprocessing.py
│   ├── explainability.py
│   └── captum_explainer.py
│
├── frontend-build/                 React + Vite frontend
│   ├── src/
│   │   ├── App.tsx                 Tab routing + model boot polling
│   │   ├── app.css                 Global dark theme
│   │   ├── pages/
│   │   │   ├── SinglePredict.tsx   Single text classification
│   │   │   ├── ExplainPage.tsx     LIME / SHAP / Captum explanations
│   │   │   ├── BatchAnalysis.tsx   Batch with live streaming progress
│   │   │   └── HistoryPage.tsx     Saved predictions + charts
│   │   ├── components/
│   │   │   ├── ProbabilityBars.tsx Class probability bars
│   │   │   └── TokenViz.tsx        Token attribution chip visualization
│   │   └── utils/
│   │       ├── api.ts              All fetch calls + NDJSON stream client
│   │       └── labels.ts           Label metadata and colors
│   ├── package.json
│   ├── vite.config.ts              Proxy: /api/* → localhost:8000
│   └── tsconfig.json
│
├── .gitignore
└── README.md
```

---

## 🧠 Model

| Property | Value |
|---|---|
| Architecture | XLM-RoBERTa-large |
| Task | 4-class hate speech classification |
| Languages | Nepali (Devanagari + Romanized), English, mixed |
| HuggingFace | [`UDHOV/xlm-roberta-large-nepali-hate-classification`](https://huggingface.co/UDHOV/xlm-roberta-large-nepali-hate-classification) |

**Classes:**

| Label | Meaning |
|---|---|
| `NO` | Non-offensive |
| `OO` | Other-offensive (general) |
| `OR` | Offensive-Racist (race / ethnicity / religion) |
| `OS` | Offensive-Sexist (gender / sexual orientation) |

The model is downloaded automatically from HuggingFace Hub on first backend startup — no manual model download required.

---

## 🚀 Running Locally

### Prerequisites

- Python 3.8+
- Node.js 18+
- Git

### 1. Clone and set up

```bash
git clone https://github.com/UddavRajbhandari/nepali-hate-detector-app.git
cd nepali-hate-detector-app
```

### 2. Backend

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the API server (run from repo root)
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

On first start the model downloads from HuggingFace (~1–2 GB). Wait for:
```
Model ready.
INFO: Application startup complete.
```

### 3. Frontend

Open a **second terminal**:

```bash
cd frontend-build
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server + model status |
| `GET` | `/api/status` | XAI package availability |
| `POST` | `/api/predict` | Single text classification |
| `POST` | `/api/analyze` | Script detection + emoji info |
| `POST` | `/api/explain/lime` | LIME word attribution |
| `POST` | `/api/explain/shap` | SHAP word attribution |
| `POST` | `/api/explain/captum` | Captum Integrated Gradients |
| `POST` | `/api/batch` | Batch classify (streaming NDJSON) |
| `GET` | `/api/history` | Saved predictions |
| `GET` | `/api/history/stats` | Prediction statistics |
| `DELETE` | `/api/history` | Clear history |

Full API documentation: **http://localhost:8000/docs** (Swagger UI, available when backend is running).

---

## ☁️ Deployment

### Frontend → Vercel

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → **New Project** → import this repo
3. Set **Root Directory** to `frontend-build`
4. Set **Build Command** to `npm run build`
5. Set **Output Directory** to `dist`
6. Add environment variable:
   ```
   VITE_API_BASE=https://your-backend.up.railway.app/api
   ```
7. Click **Deploy**

### Backend → Railway

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Select this repo
3. Set **Root Directory** to `backend`
4. Railway auto-detects the `Dockerfile` — no extra config needed
5. Add environment variables in Railway dashboard:
   ```
   HF_MODEL_ID=UDHOV/xlm-roberta-large-nepali-hate-classification
   FRONTEND_URL=https://your-app.vercel.app
   ```
6. Copy the Railway URL and paste it into Vercel's `VITE_API_BASE` env var

### Backend → HuggingFace Spaces (Docker)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Docker** as the SDK
3. Push the repo — the `Dockerfile` inside `backend/` will be used
4. Set Space secrets:
   ```
   FRONTEND_URL=https://your-app.vercel.app
   ```

---

## ⚙️ Environment Variables

### Backend

| Variable | Default | Description |
|---|---|---|
| `HF_MODEL_ID` | `UDHOV/xlm-roberta-large-nepali-hate-classification` | HuggingFace model ID |
| `MODEL_PATH` | `models/saved_models/xlm_roberta_results/large_final` | Local model path (optional) |
| `HISTORY_FILE` | `data/prediction_history.jsonl` | History file location |
| `FRONTEND_URL` | `""` | Additional CORS origin for deployed frontend |

### Frontend

| Variable | Description |
|---|---|
| `VITE_API_BASE` | Backend API URL. Leave empty for local dev (Vite proxy handles it) |

Copy `.env.example` to `.env` and fill in values for production.

---

## 🔬 Explainability Methods

| Method | Description | Speed |
|---|---|---|
| **LIME** | Perturbs preprocessed tokens, fits local linear model | Fast |
| **SHAP** | Shapley values via text masker, occlusion fallback | Medium |
| **Captum IG** | Layer Integrated Gradients on subword embeddings | Slow / memory-intensive |

All methods operate on the **preprocessed text** (transliterated to unified Devanagari) so token labels always align with what the model actually received.

---

## 📦 Dependencies

### Backend (key packages)

```
fastapi
uvicorn
transformers
torch
scikit-learn
lime
shap
captum
huggingface-hub
```

Full list: [`backend/requirements.txt`](backend/requirements.txt)

### Frontend

```
react 18
vite 5
recharts
papaparse
typescript
```

---

## 🙏 Acknowledgements

- Model trained on the [Nepali Hate Speech dataset](https://aclanthology.org/2021.woah-1.7.pdf) (WOAH 2021)
- Preprocessing pipeline built using `indic-transliteration` and `deep-translator`
- XAI implementations adapted from LIME, SHAP, and Captum libraries
- Full training pipeline and research notebooks in the [companion repository](https://github.com/UddavRajbhandari/major-project)
