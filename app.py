"""
Tech Stack Recommender — Flask Web Application
==============================================
Run locally:
    pip install flask pandas scikit-learn xgboost joblib
    python app.py
    Open: http://localhost:5000

Run on Google Colab (with ngrok):
    See colab_run.py for Colab-specific instructions.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Load all model artifacts ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading model artifacts...")
MODEL    = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
TFIDF    = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
LE_FE    = joblib.load(os.path.join(BASE_DIR, "label_encoder_frontend.pkl"))
LE_BE    = joblib.load(os.path.join(BASE_DIR, "label_encoder_backend.pkl"))
LE_DB    = joblib.load(os.path.join(BASE_DIR, "label_encoder_database.pkl"))

MODEL_FE   = MODEL["frontend"]
MODEL_BE   = MODEL["backend"]
MODEL_DB   = MODEL["database"]
FEAT_NAMES = MODEL["feat_names"]
DOMAIN_COLS = MODEL.get("domain_cols", [c for c in FEAT_NAMES if c.startswith("dom_")])
LANG_COLS   = MODEL.get("lang_cols",   [c for c in FEAT_NAMES if c.startswith("lang_")])
MODEL_NAME  = MODEL.get("model_name", "XGBoost")

print(f"✓ Model loaded: {MODEL_NAME}")
print(f"  Features: {len(FEAT_NAMES)} | "
      f"FE classes: {len(LE_FE.classes_)} | "
      f"BE classes: {len(LE_BE.classes_)} | "
      f"DB classes: {len(LE_DB.classes_)}")

# ── Dropdown options ──────────────────────────────────────────────────────────
DOMAINS = sorted([c.replace("dom_","") for c in DOMAIN_COLS])
LANGUAGES = sorted([c.replace("lang_","") for c in LANG_COLS
                    if c.replace("lang_","") not in
                    {"Unknown","Batchfile","Makefile","Dockerfile","YAML",
                     "Shell","Procfile","Markdown","MDX","EJS","HCL","Limbo",
                     "Mermaid","Gherkin","Jinja","Liquid","Twig","Handlebars",
                     "Blade","CMake","OpenSCAD","Astro","Ballerina","Nim",
                     "Monkey C","Move","Nix","SAS","TeX","Vim script","Elm",
                     "Erlang","OCaml","Smalltalk","GDScript","Groovy","Clojure",
                     "CoffeeScript","Elixir","F#","Haskell","SCSS","CSS","HTML",
                     "PLSQL","PLpgSQL","SQL","Jupyter Notebook","MATLAB","Matlab",
                     "Pascal","Perl","Procfile","Assembly"}])
SIZES      = ["Small", "Medium", "Large"]
BUDGETS    = ["Low", "Medium", "High"]
DEPLOYMENTS = ["Cloud", "On-premise"]

# ── Encoding maps ─────────────────────────────────────────────────────────────
SIZE_MAP   = {"Small":0, "Medium":1, "Large":2}
BUDGET_MAP = {"Low":0,   "Medium":1, "High":2}
DEPLOY_MAP = {"On-premise":0, "Cloud":1}

# ── Tech stack descriptions (for result page) ─────────────────────────────────
TECH_DESCRIPTIONS = {
    # Frontend
    "React":        "A JavaScript library for building component-based UIs. Most popular frontend framework worldwide.",
    "Angular":      "A TypeScript-based framework by Google. Best for large enterprise applications with complex structure.",
    "Vue.js":       "Progressive JavaScript framework. Gentle learning curve, great for medium-sized projects.",
    "Next.js":      "React framework with server-side rendering and static site generation. Excellent for SEO.",
    "React Native": "Build native mobile apps with React. Write once, deploy to iOS and Android.",
    "Flutter":      "Google's UI toolkit for natively compiled apps from a single Dart codebase.",
    "Bootstrap":    "CSS framework for rapid, responsive web development. Ideal for quick prototypes.",
    "HTML/CSS":     "Standard web technologies. Best for static sites and simple content-driven pages.",
    "jQuery":       "Lightweight JavaScript library. Good for adding interactivity to existing sites.",
    "Other":        "Less common framework. Check project-specific requirements for the best fit.",
    # Backend
    "Node.js":      "JavaScript runtime for server-side development. Event-driven, non-blocking. Ideal for real-time apps.",
    "Django":       "High-level Python web framework. Batteries-included, rapid development, strong security.",
    "Spring Boot":  "Java framework for enterprise applications. Robust, scalable, excellent for microservices.",
    "Laravel":      "Elegant PHP framework. Expressive syntax, great ORM, perfect for web applications.",
    "Rails":        "Ruby on Rails. Convention over configuration. Fast development, great for startups.",
    "Go":           "Google's compiled language. Extremely fast, efficient concurrency. Great for high-performance APIs.",
    "ASP.NET":      "Microsoft's framework for C#. Enterprise-grade, Azure integration, Windows environments.",
    "Flask":        "Lightweight Python micro-framework. Flexible, minimal — good for small APIs and microservices.",
    "Flutter":      "Also used as a backend with Dart. Emerging full-stack option.",
    "Other":        "Less common backend. Review project-specific requirements.",
    # Database
    "MongoDB":      "Document-oriented NoSQL database. Flexible schema, excellent for unstructured/semi-structured data.",
    "PostgreSQL":   "Advanced open-source relational database. ACID compliant, excellent for complex queries.",
    "MySQL":        "World's most popular open-source RDBMS. Fast, reliable, widely supported.",
    "Firebase":     "Google's real-time NoSQL cloud database. Ideal for mobile apps and real-time sync.",
    "Redis":        "In-memory key-value store. Ultra-fast caching, session management, pub/sub messaging.",
    "SQLite":       "Serverless, file-based database. Perfect for small apps, prototypes, and embedded systems.",
    "SQL Server":   "Microsoft's enterprise RDBMS. Deep Windows/Azure integration, strong business intelligence.",
    "Other":        "Less common database. Check project-specific requirements.",
}

# ── Use case recommendations ──────────────────────────────────────────────────
STACK_REASONS = {
    "React":       "Popular for interactive UIs, large ecosystem, component reusability",
    "Angular":     "Enterprise-grade, TypeScript support, strong structure and dependency injection",
    "Vue.js":      "Progressive adoption, gentle learning curve, excellent performance",
    "Next.js":     "Server-side rendering, SEO benefits, full-stack React capabilities",
    "Node.js":     "Non-blocking I/O, same language as frontend (JavaScript), real-time capabilities",
    "Django":      "Rapid development, built-in admin, ORM, strong security features",
    "Spring Boot": "Enterprise reliability, Java ecosystem, microservices support",
    "Laravel":     "Elegant syntax, Eloquent ORM, blade templating, strong PHP ecosystem",
    "Go":          "High throughput, low memory usage, excellent for concurrent workloads",
    "ASP.NET":     "Microsoft ecosystem, C# language, Azure cloud integration",
    "MongoDB":     "Flexible schema, horizontal scaling, JSON-like documents, high write throughput",
    "PostgreSQL":  "ACID compliance, complex queries, advanced data types, strong consistency",
    "MySQL":       "Proven reliability, wide hosting support, excellent for structured data",
    "Firebase":    "Real-time sync, serverless, Google ecosystem, mobile-first",
    "Redis":       "Sub-millisecond latency, caching, session storage, pub/sub",
    "SQLite":      "Zero configuration, serverless, perfect for development and small apps",
}


def build_input_vector(domain, fr, nfr, size, team, budget, duration, deploy, language):
    """Convert user form inputs into the base feature vector from FEAT_NAMES."""

    # Start with all zeros
    row = {col: 0.0 for col in FEAT_NAMES}

    # Numeric features (exact column names from preprocessing)
    row["Team_Size"]          = float(team)
    row["Duration_Months"]    = float(duration)
    row["Project_Size_enc"]   = float(SIZE_MAP.get(size, 1))
    row["Budget_Level_enc"]   = float(BUDGET_MAP.get(budget, 1))
    row["Deployment_enc"]     = float(DEPLOY_MAP.get(deploy, 1))

    # Domain one-hot
    dom_col = f"dom_{domain}"
    if dom_col in row:
        row[dom_col] = 1.0

    # Language one-hot
    lang_col = f"lang_{language}"
    if lang_col in row:
        row[lang_col] = 1.0

    # TF-IDF on combined text
    combined_text = str(fr) + " " + str(nfr)
    tfidf_vec     = TFIDF.transform([combined_text]).toarray()[0]
    for j, word in enumerate(TFIDF.get_feature_names_out()):
        key = f"tfidf_{word}"
        if key in row:
            row[key] = float(tfidf_vec[j])

    return pd.DataFrame([row])[FEAT_NAMES]


def align_input_for_model(X_input, model):
    """Pad/trim base features to satisfy each model's expected feature count."""
    expected = int(getattr(model, "n_features_in_", X_input.shape[1]))
    arr = X_input.to_numpy(dtype=float)

    if arr.shape[1] < expected:
        # Some saved models were trained with extra columns that are not in FEAT_NAMES.
        # Use zero padding for unknown columns to keep inference robust.
        pad = np.zeros((arr.shape[0], expected - arr.shape[1]), dtype=float)
        arr = np.hstack([arr, pad])
    elif arr.shape[1] > expected:
        arr = arr[:, :expected]

    return arr


def get_recommendation(domain, fr, nfr, size, team, budget, duration, deploy, language):
    """Run prediction and return structured results with confidence scores."""
    X_input = build_input_vector(
        domain, fr, nfr, size, team, budget, duration, deploy, language
    )

    X_fe = align_input_for_model(X_input, MODEL_FE)
    X_be = align_input_for_model(X_input, MODEL_BE)
    X_db = align_input_for_model(X_input, MODEL_DB)

    # Predictions
    fe_pred = LE_FE.inverse_transform(MODEL_FE.predict(X_fe))[0]
    be_pred = LE_BE.inverse_transform(MODEL_BE.predict(X_be))[0]
    db_pred = LE_DB.inverse_transform(MODEL_DB.predict(X_db))[0]

    # Confidence probabilities
    fe_proba = MODEL_FE.predict_proba(X_fe)[0]
    be_proba = MODEL_BE.predict_proba(X_be)[0]
    db_proba = MODEL_DB.predict_proba(X_db)[0]

    # Top-3 alternatives for each target
    fe_top3 = sorted(zip(LE_FE.classes_, fe_proba), key=lambda x: -x[1])[:3]
    be_top3 = sorted(zip(LE_BE.classes_, be_proba), key=lambda x: -x[1])[:3]
    db_top3 = sorted(zip(LE_DB.classes_, db_proba), key=lambda x: -x[1])[:3]

    return {
        "frontend": {
            "name":        fe_pred,
            "confidence":  round(float(max(fe_proba)) * 100, 1),
            "description": TECH_DESCRIPTIONS.get(fe_pred, ""),
            "reason":      STACK_REASONS.get(fe_pred, ""),
            "top3": [
                {"name": c, "confidence": round(float(p)*100, 1)}
                for c, p in fe_top3
            ],
        },
        "backend": {
            "name":        be_pred,
            "confidence":  round(float(max(be_proba)) * 100, 1),
            "description": TECH_DESCRIPTIONS.get(be_pred, ""),
            "reason":      STACK_REASONS.get(be_pred, ""),
            "top3": [
                {"name": c, "confidence": round(float(p)*100, 1)}
                for c, p in be_top3
            ],
        },
        "database": {
            "name":        db_pred,
            "confidence":  round(float(max(db_proba)) * 100, 1),
            "description": TECH_DESCRIPTIONS.get(db_pred, ""),
            "reason":      STACK_REASONS.get(db_pred, ""),
            "top3": [
                {"name": c, "confidence": round(float(p)*100, 1)}
                for c, p in db_top3
            ],
        },
        "overall_confidence": round(
            (float(max(fe_proba)) + float(max(be_proba)) + float(max(db_proba))) / 3 * 100, 1
        ),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
        domains=DOMAINS,
        languages=LANGUAGES,
        sizes=SIZES,
        budgets=BUDGETS,
        deployments=DEPLOYMENTS,
        model_name=MODEL_NAME,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        domain   = data.get("domain", "")
        fr       = data.get("functional_requirements", "")
        nfr      = data.get("non_functional_requirements", "")
        size     = data.get("project_size", "Medium")
        team     = int(data.get("team_size", 5))
        budget   = data.get("budget_level", "Medium")
        duration = int(data.get("duration_months", 6))
        deploy   = data.get("deployment", "Cloud")
        language = data.get("primary_language", "JavaScript")

        # Validate
        if not domain or not fr:
            return jsonify({"error": "Domain and Functional Requirements are required."}), 400

        result = get_recommendation(
            domain, fr, nfr, size, team, budget, duration, deploy, language
        )
        result["input"] = {
            "domain": domain, "size": size, "team": team,
            "budget": budget, "duration": duration,
            "deployment": deploy, "language": language,
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "features": len(FEAT_NAMES),
        "features_expected": {
            "frontend": int(getattr(MODEL_FE, "n_features_in_", len(FEAT_NAMES))),
            "backend": int(getattr(MODEL_BE, "n_features_in_", len(FEAT_NAMES))),
            "database": int(getattr(MODEL_DB, "n_features_in_", len(FEAT_NAMES))),
        },
        "frontend_classes": list(LE_FE.classes_),
        "backend_classes":  list(LE_BE.classes_),
        "database_classes": list(LE_DB.classes_),
    })


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Tech Stack Recommender — Starting server")
    print("  Open: http://localhost:5005")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5005)
