# StackSense

ML-powered Tech Stack Recommender
================================

StackSense is a research prototype web application that recommends a technology stack (frontend framework, backend technology, and database) based on a project's functional and non-functional requirements. It uses a trained machine-learning model to map free-text requirements and structured project attributes to stack recommendations.

This repository contains the inference server (Flask) and serialized model artifacts used for predictions.

Highlights
----------
- Language: Python
- Web framework: Flask (single-file prototype)
- Machine learning: scikit-learn and XGBoost (model artifacts saved with joblib)
- Text processing: TF-IDF vectorization for requirement text
- Frontend: server-rendered HTML (templates/index.html) with minimal JavaScript

What this repo contains
-----------------------
- app.py — Flask application and inference logic (main entrypoint). See [app.py].
- templates/index.html — Minimal UI served by Flask. See [templates/].
- best_model.pkl — Main trained model bundle (may contain multiple sub-models or an ensemble).
- tfidf_vectorizer.pkl — TF-IDF vectorizer used to convert requirement text into features.
- label_encoder_frontend.pkl, label_encoder_backend.pkl, label_encoder_database.pkl — Label encoders that map model outputs to human-friendly names.
- requirements.txt — Python dependencies (if present in the repo).

Note: The repository may include a local virtual environment (.venv/) — this is not required in the repository and can be removed before sharing.

Usage
-----
1. Clone or open the project and change into the project root.

2. (Recommended) Create and activate a virtual environment:

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate     # Windows

3. Install dependencies. If a requirements.txt file exists, prefer that:

   pip install -r requirements.txt

   Or install the common runtime packages manually:

   pip install flask pandas numpy scikit-learn xgboost joblib

4. Run the application locally:

   python app.py

   By default the server listens on 0.0.0.0:5005 (see app startup output). Open http://localhost:5005 in a browser to use the UI.

API
---
- GET / — Serves the web UI (templates/index.html).
- POST /predict — Returns a JSON response with recommended frontend, backend, and database choices plus associated confidence scores. Check [app.py] for the exact request/response schema.
- GET /health — Returns model metadata and available classes (useful for automated health checks).

Example /predict input (JSON)
-----------------------------
The server expects a JSON payload describing the project. Typical fields include (see app.py for exact keys):

- domain (string) — project domain (e.g., "e-commerce")
- functional_requirements (string) — required features, primary text used by TF-IDF
- non_functional_requirements (string) — optional text
- project_size (Small | Medium | Large)
- team_size (int)
- budget_level (Low | Medium | High)
- duration_months (int)
- deployment (Cloud | On-premise)
- primary_language (string)

The response includes recommended labels for frontend, backend and database along with optional confidence scores.

Model artifacts
---------------
The app loads pre-trained artifacts from the repository root. These are required for inference:
- best_model.pkl — model bundle (may be a dict with multiple models)
- tfidf_vectorizer.pkl — TF-IDF transformer
- label_encoder_*.pkl — label encoders for mapping numeric outputs to strings

If any artifact is missing or corrupted the server will raise an error on startup. To update the model, re-run the training pipeline (outside the inference server), then replace these .pkl files.

Development notes
-----------------
- The app is intentionally kept as a single-file prototype (app.py). For larger projects split the code into modules such as:
  - app/ — Flask blueprints and routes
  - models/ — training, evaluation, and artifact packaging
  - static/ and templates/ — frontend assets
- Add automated tests (pytest) under a tests/ directory to validate behavior.
- Use `pip freeze > requirements.txt` to capture dependency versions before sharing or deployment.

Security & deployment
---------------------
- The prototype may run with debug enabled in development. Disable debug when deploying to production.
- Use a production-ready WSGI server (gunicorn or uWSGI) behind a reverse proxy (nginx).
- Keep model artifacts (the .pkl files) accessible only to the server process and not directly served to clients.

Reproducing / Retraining the model
----------------------------------
This repository focuses on inference. Training code and datasets are not included here by default. To retrain:
- Prepare a labeled dataset mapping requirements + structured features to stack labels.
- Use scikit-learn / xgboost for preprocessing and model training.
- Save the trained model, TF-IDF vectorizer, and label encoders with joblib (e.g., joblib.dump).
- Replace the artifacts in the repo root and restart the server.

Project structure (example)
---------------------------
StackSense/
├─ app.py
├─ templates/
│  └─ index.html
├─ best_model.pkl
├─ tfidf_vectorizer.pkl
├─ label_encoder_frontend.pkl
├─ label_encoder_backend.pkl
├─ label_encoder_database.pkl
├─ requirements.txt
└─ README.md

Contributing
------------
- Open an issue describing the change or feature.
- Create feature branches and open pull requests for review.
- Ensure changes include tests where applicable.

License
-------
Add a LICENSE file to explicitly state licensing terms. The MIT license is a commonly used permissive license.

Contact / Notes
---------------
For questions about how inference works, inspect [app.py] and the frontend in [templates/](/Users/kanukany/Documents/SE8101 - Research /StackSense/templates).

Acknowledgements
-----------------
Developed as part of an SE8101 research project. Thanks to all contributors and reviewers.


---

(Updated README)