import os
import pandas as pd
import mlflow
import mlflow.sklearn
import re
from collections import Counter, defaultdict
import spacy
from transformers import BertTokenizer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────── CONFIGURATION ───────────
mlflow.set_tracking_uri("https://julienrouillard-getaround-prediction.hf.space")

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.eu-west-3.amazonaws.com"
ARTIFACT_STORE_URI = os.getenv("ARTIFACT_STORE_URI")

#══════════════════════════════════════════════════════════════════════════════════════
#                             PARTIE MODIFIEE
#══════════════════════════════════════════════════════════════════════════════════════

class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp, tokenizer, stop_en, connectives):
        self.nlp = nlp
        self.tokenizer = tokenizer
        self.stop_en = stop_en
        self.connectives = connectives

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X : liste de textes bruts
        return : DataFrame de features numériques
        """
        df = pd.DataFrame({'text': X})
        df["bert_wp"] = df["text"].astype(str).apply(self.tokenizer.tokenize)
        df["bert_wp_len"] = df["bert_wp"].apply(len)
        df["bert_ids"] = df["text"].astype(str).apply(lambda t: self.tokenizer.encode(t, add_special_tokens=True))
        df["bert_ids_len"] = df["bert_ids"].apply(len)

        # Reconstruction des mots
        def reconstruct_words(wp_tokens):
            words = []
            current = ""
            for tok in wp_tokens:
                if tok.startswith("##"):
                    current += tok[2:]
                else:
                    if current:
                        words.append(current)
                    current = tok
            if current:
                words.append(current)
            return words

        df["bert_words"] = df["bert_wp"].apply(reconstruct_words)
        df["bert_words_len"] = df["bert_words"].apply(len)

        # Normalisation de texte
        df["text_norm"] = df["text"].astype(str).apply(lambda t: re.sub(r"\s+", " ", t.lower()).strip())

        # Longueurs
        len_chars = []
        len_tokens = []
        len_words = []
        n_sentences = []
        avg_sentence_len = []

        for doc in self.nlp.pipe(df["text"].astype(str).tolist(), batch_size=256):
            tokens = list(doc)
            len_chars.append(len(doc.text))
            len_tokens.append(len(tokens))
            words = [tok for tok in tokens if tok.is_alpha or tok.like_num or tok.like_url]
            len_words.append(len(words))
            sents = list(doc.sents)
            n_sentences.append(len(sents))
            avg_len = sum(len([t for t in sent if not t.is_space]) for sent in sents) / max(len(sents), 1)
            avg_sentence_len.append(avg_len)

        df["len_chars"] = len_chars
        df["len_tokens_all"] = len_tokens
        df["len_words"] = pd.Series(len_words).clip(lower=1)
        df["n_sentences"] = n_sentences
        df["average_sentences_length"] = avg_sentence_len
        df["len_chars_per_word"] = df["len_chars"] / df["len_words"]
        df["len_tokens_per_word"] = df["len_tokens_all"] / df["len_words"]

        # Ratio majuscules
        df["freq_uppercase"] = df["text"].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))

        # Ponctuation
        PUNCT_LIST = ['!', '?', ',', '.', ';', ':', '"', "'", '(']
        ELLIPSIS_TOKEN = '...'

        def punctuation_features(text, len_words):
            text = text or ""
            res = {}
            ellipses = text.count(ELLIPSIS_TOKEN)
            text_wo_ell = text.replace(ELLIPSIS_TOKEN, "")
            res["punct_ellipsis_ratio"] = ellipses / len_words
            for p in PUNCT_LIST:
                name = re.sub(r'\W', '_', p)
                key = f"punct_{name}_ratio"
                res[key] = text_wo_ell.count(p) / len_words
            return res

        punct_features = [punctuation_features(t, lw) for t, lw in zip(df["text"], df["len_words"])]
        punct_df = pd.DataFrame(punct_features)
        df = pd.concat([df.reset_index(drop=True), punct_df.reset_index(drop=True)], axis=1)

        # Stopwords ratio
        stopword_ratios = []
        for doc in self.nlp.pipe(df["text"].astype(str).tolist(), batch_size=256):
            valid_words = [tok.text.lower() for tok in doc if tok.is_alpha or tok.like_num or tok.like_url]
            count_stop = sum(1 for w in valid_words if w in self.stop_en)
            stopword_ratios.append(count_stop / max(len(valid_words), 1))
        df["stopwords_ratio"] = stopword_ratios

        # Connecteurs logiques
        def detect_connectives(words):
            detected = defaultdict(list)
            text = ' '.join(words).lower()
            for cat, phrases in self.connectives.items():
                for phrase in phrases:
                    if phrase in text:
                        detected[cat].append(phrase)
            return Counter({cat: len(lst) for cat, lst in detected.items()})

        connective_counts = [detect_connectives([tok.text for tok in doc if tok.is_alpha])
                             for doc in self.nlp.pipe(df["text"], batch_size=256)]
        conn_df = pd.DataFrame(connective_counts).fillna(0)

        # Créer toutes les colonnes dans un ordre fixe.
        all_categories = sorted(self.connectives.keys())
        for cat in all_categories:
            if cat not in conn_df.columns:
                conn_df[cat] = 0

        # Réorganiser les colonnes dans l'ordre alphabétique.
        conn_df = conn_df[all_categories]

        conn_df = conn_df.div(df["len_words"], axis=0)
        conn_df.columns = [f"connective_{c}_ratio" for c in conn_df.columns]
        df = pd.concat([df, conn_df], axis=1)

        # POS ratios
        POS_TAGS = [
            "DET", "VERB", "SCONJ", "AUX", "PART", "CCONJ",
            "ADV", "ADJ", "ADP", "PROPN", "PRON", "NOUN", "NUM"
        ]
        pos_counts = []
        for doc in self.nlp.pipe(df["text"].astype(str).tolist(), batch_size=256):
            cnt = Counter(tok.pos_ for tok in doc if not tok.is_space)
            pos_counts.append({pos: cnt.get(pos, 0) for pos in POS_TAGS})
        pos_df = pd.DataFrame(pos_counts)
        pos_df = pos_df.div(df["len_words"], axis=0)
        pos_df.columns = [f"pos_{col}_ratio" for col in pos_df.columns]
        df = pd.concat([df.reset_index(drop=True), pos_df.reset_index(drop=True)], axis=1)

        # Sélectionner uniquement les colonnes numériques
        cols_to_exclude = ['text', 'text_norm', 'bert_wp', 'bert_words', 'bert_ids']
        numeric_df = df.drop(columns=[col for col in df.columns if col in cols_to_exclude or df[col].dtype == 'object'])
        return numeric_df


# ─────────── CHARGEMENT DES DONNÉES ───────────
# Chargement des données brutes
df = pd.read_csv("../datasets_source/Compilation_from_Kaggle_IA_Human.csv")
for col in ["classe", "SYM"]:
    if col in df.columns:
        df = df.drop(columns=[col])

X = df["text"]
Y = df["label"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Initialisation des ressources NLP
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
stop_en = spacy.lang.en.stop_words.STOP_WORDS

# Définition des connecteurs logiques (en dehors de la classe)
connectives = {
    'addition': {'and', 'also', 'furthermore', 'moreover', 'in addition', 'besides', 'as well', 'what is more',
                 'not only... but also', 'similarly', 'likewise'},
    'contrast': {'but', 'however', 'on the other hand', 'nevertheless', 'nonetheless', 'yet', 'still',
                 'even so', 'although', 'though', 'whereas', 'while', 'in contrast', 'conversely'},
    'cause': {'because', 'since', 'as', 'due to', 'owing to', 'thanks to', 'considering that', 'for the reason that'},
    'consequence': {'so', 'therefore', 'thus', 'hence', 'as a result', 'consequently', 'accordingly', 'for this reason'},
    'concession': {'although', 'even though', 'though', 'while', 'granted that', 'admittedly', 'it is true that', 'nonetheless'},
    'example': {'for example', 'for instance', 'such as', 'like', 'to illustrate', 'namely', 'including', 'in particular'},
    'purpose': {'so that', 'in order to', 'in order that', 'so as to', 'to', 'for the purpose of', 'with the aim of'},
    'time': {'first', 'then', 'next', 'after that', 'afterwards', 'before', 'finally', 'meanwhile',
             'eventually', 'at the same time', 'subsequently'},
    'summary': {'in conclusion', 'to conclude', 'in summary', 'to sum up', 'overall', 'in short', 'all in all', 'ultimately'}
}

# ─────────── DÉFINITION DU XGBoost ───────────
xgb_best = XGBClassifier(
    gamma=0,
    learning_rate=0.3,
    max_depth=5,
    min_child_weight=1,
    n_estimators=300,
    subsample=1.0,
    random_state=42,
    eval_metric='logloss'
)

# ─────────── CONSTRUCTION DU PIPELINE ───────────
pipeline = Pipeline([ ("preprocessing", PreprocessingTransformer(
        nlp=nlp, tokenizer=tokenizer, stop_en=stop_en, connectives=connectives
    )),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("xgb_classifier", xgb_best)
])

# ─────────── ENTRAÎNEMENT ───────────
print("🔄 Début de l'entraînement...")
pipeline.fit(X_train, Y_train)
print("✅ Entraînement terminé !")

# ─────────── TEST DE SÉRIALISATION ───────────
import pickle

print("🧪 Test de sérialisation du pipeline...")
try:
    serialized = pickle.dumps(pipeline)
    print(f"✅ Sérialisation OK ({len(serialized)} bytes)")
    
    # Test de désérialisation
    pipeline_reloaded = pickle.loads(serialized)
    test_pred = pipeline_reloaded.predict(X_test[:5])
    print(f"✅ Désérialisation OK - Prédictions test: {test_pred}")
except Exception as e:
    print(f"❌ Erreur de sérialisation: {e}")
    raise

# ─────────── PRÉDICTIONS ───────────
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# ─────────── MÉTRIQUES ───────────
train_acc = accuracy_score(Y_train, y_train_pred)
test_acc = accuracy_score(Y_test, y_test_pred)
test_prec = precision_score(Y_test, y_test_pred, zero_division=0)
test_rec = recall_score(Y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(Y_test, y_test_pred, zero_division=0)

# ─────────── LOG DANS MLFLOW ───────────
mlflow.set_experiment("AI")

with mlflow.start_run(run_name="XGB_Classifier_with_Preprocessing"):

    # Log métriques
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_prec)
    mlflow.log_metric("test_recall", test_rec)
    mlflow.log_metric("test_f1_score", test_f1)
    
    # Log pipeline complet (preprocessing + modèle)
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="xgboost_pipeline",
        input_example=X_train[:5].tolist()
    )
    
    print("✅ Pipeline complet (preprocessor + modèle) enregistré sur MLflow !")
    print(f"📊 Test Accuracy: {test_acc:.3f}")
    print(f"📊 Test F1-Score: {test_f1:.3f}")
