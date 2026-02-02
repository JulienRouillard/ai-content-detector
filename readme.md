# AI-Generated Content Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)

> Binary text classification system to detect AI-generated content using 43 linguistic features and XGBoost

---

## 🎯 Problem Statement

With the rise of Large Language Models like GPT and Claude, distinguishing between human-written and AI-generated text has become increasingly important. This project builds a classifier that analyzes linguistic patterns to identify AI-generated content.

**Use Cases:**
- Content moderation platforms
- Academic integrity verification  
- Review platform fraud detection
- Social media authenticity checks

---

## 🚀 Key Results

### Model Performance (Test Set - 2,000 texts)

| Metric | Score |
|--------|-------|
| **Recall** | **97.1%** 🎯 |
| Accuracy | 85.2% |
| Precision | 78.4% |
| F1-Score | 86.7% |

**Why 97.1% Recall Matters:** The model successfully detects 97 out of 100 AI-generated texts, ensuring reliable detection of AI-generated content in production environments.

---

## 💡 Solution Overview

### Architecture

```
Text Input → BERT Tokenization → Feature Extraction (43 features) → XGBoost Classifier → Prediction
```

### Key Design Choices

**Feature-Based Approach:**
- 43 engineered linguistic features capturing writing style patterns
- Interpretable features (stopwords ratio, sentence structure, POS tags)

**Why This Works:**
- Human and AI writing exhibit distinct linguistic patterns
- Grammatical structure differs significantly between sources
- Statistical features provide robust discrimination

---

## 🔬 Technical Approach

### 1. Data Pipeline

**Dataset:**
- Training: 16,193 texts (63% AI / 37% Human) from multiple Kaggle datasets
- Test: 2,000 texts (50% AI / 50% Human) - completely separate dataset
- Content types: essays, reviews, articles...

**Preprocessing:**
- BERT tokenizer (`bert-base-uncased`) for consistent text splitting
- SpaCy & NLTK for linguistic analysis
- Text normalization while preserving punctuation

### 2. Feature Engineering

**43 Features Across 4 Categories:**

**Text Structure (7 features)**
- Character count, word count, token count
- Sentence count and average sentence length
- Character-per-word and token-per-word ratios

**Writing Style (11 features)**
- Uppercase letter frequency
- Stopwords ratio
- Punctuation patterns: `.` `,` `!` `?` `;` `:` `"` `'` `...`

**Logical Connectors (9 features)**
- Ratios for addition, contrast, cause, consequence, concession, example, purpose, time, summary markers
- Examples: *and, however, because, therefore, although, for example, in order to, first, in conclusion*

**Part-of-Speech Tags (13 features)**
- Proportions of grammatical categories: DET, NOUN, VERB, ADJ, ADV, ADP, PRON, AUX, PART, CCONJ, SCONJ, PROPN, NUM

**Key Observations from EDA:**
- AI generates longer sentences on average
- Humans use more stopwords (51% vs 41%)
- Distinct POS patterns: humans rely more on pronouns, auxiliaries, and subordinating conjunctions

### 3. Model Training

**Algorithm Selection:**
- Tested: Logistic Regression, Random Forest, XGBoost, ensemble methods
- **Selected: XGBoost** for best performance

**Configuration:**
```python
XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.3
)
```

**Training Strategy:**
- GridSearchCV for hyperparameter optimization
- Cross-validation on training set
- Final evaluation on separate test dataset

### 4. MLOps & Deployment

**Experiment Tracking:**
- MLflow server on HuggingFace Spaces
- Model artifacts on AWS S3
- Metrics stored in Neon PostgreSQL

**Application Stack:**
- **FastAPI**: REST API with two endpoints
  - `/predict` - Single text classification
  - `/predict_batch` - Batch processing (JSON/CSV/Excel)
- **Streamlit**: Interactive web interface
  - Real-time text prediction
  - File upload with batch processing
  - Results visualization and export


---

## 📊 Results

### Performance

| Accuracy | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| **85.2%** | 78.4% | **97.1%** | 86.7% |

The model shows strong generalization, with excellent recall ensuring most AI-generated content is detected while maintaining acceptable precision.

---

## 📁 Project Structure

```
ai-review-detector/
├── datasets_source/       # Raw and processed datasets
├── eda/                   # Exploratory analysis notebook
├── preprocessing/         # Feature extraction script
├── ml_features_extracted/ # Model training notebooks
├── mlflow/                # MLflow tracking configuration
├── api/                   # FastAPI application
├── streamlit/             # Streamlit demo interface
├── requirements/          # Dependency management files
└── README.md
```

---

## 🎥 Demo

### API Screenshots

#### API Presentation
![API](https://raw.githubusercontent.com/JulienRouillard/ai-review-detector/main/images/api_1.png)

#### API Endpoints
![API](https://raw.githubusercontent.com/JulienRouillard/ai-review-detector/main/images/api_2.png)

### Video Demonstration

[![Watch Demo](https://img.shields.io/badge/▶️_Watch_Streamlit_Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/MJafzeLzvXA)

---

## 👤 Contact

**Julien Rouillard**  
[GitHub](https://github.com/JulienRouillard) | [LinkedIn](https://www.linkedin.com/in/julien-rouillard-a738a33a9/)

---

## 🛠️ Technologies

Python • scikit-learn • XGBoost • SpaCy • NLTK • Transformers • MLflow • FastAPI • Streamlit • AWS S3 • PostgreSQL • HuggingFace Spaces