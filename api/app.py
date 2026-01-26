#══════════════════════════════════════════════════════════════════════════════════════
#                              IMPORT DES LIBRAIRIES
#══════════════════════════════════════════════════════════════════════════════════════

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union
import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import mlflow
import spacy
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import os



#══════════════════════════════════════════════════════════════════════════════════════
#                           DESCRIPTION COMPLETE DE L'API
#══════════════════════════════════════════════════════════════════════════════════════

description = """
## AI Review Detector API

L'**AI Review Detector API** est une solution professionnelle de détection automatique des avis générés par intelligence artificielle.

### 🎯 Objectif

Cette API permet d'identifier si un avis client a été rédigé par un humain ou généré par une intelligence artificielle, 
aidant ainsi les entreprises à maintenir l'authenticité de leurs plateformes d'évaluation.

### 🔬 Modèle de détection

Notre système repose sur un modèle **XGBoost** entraîné sur un vaste corpus de textes authentiques et générés par IA.

**Performances du modèle sur de nouvelles données :**
- **Accuracy** : 85.2%
- **Precision** : 78.4%
- **Recall** : 97.1%
- **F1-Score** : 86.7%

**Dataset d'entraînement :**
- 16 193 avis étiquetés (37% humains / 63% IA)
- Source : Combinaison de plusieurs datasets issus de Kaggle
- Langue : Anglais
- Taille variable : De courts commentaires à des essais détaillés

### 💡 Cas d'usage

- **E-commerce** : Validation de l'authenticité des avis produits
- **Hôtellerie** : Détection de faux avis sur les plateformes de réservation  
- **Modération** : Filtrage automatique des contenus générés automatiquement par IA
- **Analyse qualitative** : Audit de bases de données d'avis existantes

### 🚀 Fonctionnalités

Cette API propose deux modes d'utilisation :

1. **Analyse unitaire** : Détection pour un seul texte
2. **Traitement par lot** : Analyse de jusqu'à 10,000 textes simultanément pour un traitement efficace

### 📊 Format des données

**Entrée :** Texte brut (sans limite de longueur)  
**Sortie :** Classification binaire (1 = IA ou 0 = Humain)

### ⚡ Performance

Temps de réponse moyen : < XXXms (Reste à calculer)
"""

app = FastAPI(
    title="AI Review Detector API",
    description=description,
    version="1.0.0",
    contact={
        "name": "AI Review Detector Team",
        "email": "contact@aireviewdetector.com",
    }
)



#══════════════════════════════════════════════════════════════════════════════════════
#                                MODELES DE DONNEES
#══════════════════════════════════════════════════════════════════════════════════════

#Modèle de données pour une requête unitaire
class TextInput(BaseModel):
    text: str = Field(
        ...,
        description="Texte de l'avis à analyser",
        example="This product exceeded my expectations! The quality is outstanding and delivery was super fast"
    )

#Modèle de données pour une requête batch
class BatchTextInput(BaseModel):
    texts: list[Union[str, dict]] = Field(
        ...,
        description="Liste de textes à analyser (maximum 10,000)",
        max_length=10000,
        example=[
            "This product exceeded my expectations! The quality is outstanding.",
            {"text": "I really enjoyed my stay at this hotel. The staff was very friendly."},
            "The food was delicious and the service was impeccable."
        ]
    )

#Modèle de données pour une réponse unitaire
class PredictionResponse(BaseModel):
    is_ai_generated: int = Field(..., description="1 si généré par l'IA, 0 si écrit par un humain")
    message: str = Field(..., description="Message explicatif du résultat")

#Modèle de données pour une réponse batch
class BatchPredictionResponse(BaseModel):
    predictions: list[int] = Field(..., description="Liste des prédictions (1 = IA, 0 = Humain)")


#══════════════════════════════════════════════════════════════════════════════════════
#                               CONFIGURATION GLOBALE
#══════════════════════════════════════════════════════════════════════════════════════

#Variables globales pour les ressources NLP
nlp = None
stop_en = None
tokenizer = None
model = None

#MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MODEL_URI = os.environ.get("MODEL_URI")

#Configuration
STOPWORDS_LANGUAGE = "english"
BERT_MODEL_NAME = "bert-base-uncased"

#Listes de ponctuations et connecteurs
PUNCT_LIST = ['!', '?', ',', '.', ';', ':', '"', "'", '(']
ELLIPSIS_TOKEN = '...'

ALL_POS_TAG = [
    "DET", "VERB", "SCONJ", "AUX", "PART", "CCONJ",
    "ADV", "ADJ", "ADP", "PROPN", "PRON", "NOUN", "NUM"
]

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



#══════════════════════════════════════════════════════════════════════════════════════
#                           CHARGEMENT DES RESSOURCES NLP
#══════════════════════════════════════════════════════════════════════════════════════

def load_nlp_ressources():
    """
    Charge toutes les ressources NLP nécessaires (Spacy, NLTK, BERT).
    À appeler UNE SEULE FOIS au démarrage de l'API.
    """

    global nlp, stop_en, tokenizer

    try:
        #Télécharger les stopwords si nécessaire
        try:
            stopwords.words(STOPWORDS_LANGUAGE)
        except LookupError:
            nltk.download('stopwords', quiet=True)

        #Charger les stopwords
        stop_en = set(stopwords.words(STOPWORDS_LANGUAGE))

        #Charger Spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

        #Charger le tokenizer BERT
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

        print("✅ Ressources NLP chargées avec succès")
        return True
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement des ressources NLP : {e}")
        raise RuntimeError(f"Impossible de charger les ressources NLP : {e}")



#══════════════════════════════════════════════════════════════════════════════════════
#                        FONCTION DE PREPROCESSING / EXTRACTION
#══════════════════════════════════════════════════════════════════════════════════════

def is_word_tok(tok):
    """
    Vérifie si un token est un 'vrai mort' (alpha, nombre, URL)
    """
    return tok.is_alpha or tok.like_num or tok.like_url


#══════════════════════════════════════════════════════════════════════════════════════
#                             PARTIE MODIFIEE
#══════════════════════════════════════════════════════════════════════════════════════
@app.post("/predict", response_model=PredictionResponse, tags=["Détection"])
async def predict_single_text(input_data: TextInput):
    """
    Analyse un seul texte pour déterminer s'il a été généré par une IA.
    """

    # Ici, on passe directement une liste contenant le texte au modèle (pipeline)
    prediction = int(model.predict([input_data.text])[0])

    if prediction == 1:
        message = "Cet avis semble avoir été généré par une intelligence artificielle."
    else:
        message = "Ce texte semble avoir été écrit par un humain."

    return PredictionResponse(
        is_ai_generated=prediction,
        message=message
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Détection"])
async def predict_batch_texts(input_data: BatchTextInput):
    """
    Analyse plusieurs textes simultanément pour déterminer s'ils ont été générés par une IA.
    """

    texts = []
    for item in input_data.texts:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            if "text" in item:
                texts.append(item["text"])
            elif "texte" in item:
                texts.append(item["texte"])
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Les dictionnaires doivent contenir une clé 'text' ou 'texte'."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporté: {type(item)}. Utilisez des strings ou des dictionnaires."
            )

    # Passage direct au modèle (pipeline)
    predictions = model.predict(texts)
    predictions = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

    return BatchPredictionResponse(predictions=predictions)

#══════════════════════════════════════════════════════════════════════════════════════
#                        CHARGEMENT DU MODELE DE PREDICTION
#══════════════════════════════════════════════════════════════════════════════════════

def load_ml_model():
    """
    Charge le modèle de ML depuis MLflow.
    À appeler UNE SEULE FOIS au démarrage de l'API.
    """

    global model

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.sklearn.load_model(MODEL_URI)

        print("✅ Modèle ML chargé avec succès")
        return True
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        raise RuntimeError(f"Impossible de charger le modèle : {e}")



#══════════════════════════════════════════════════════════════════════════════════════
#                                   ENDPOINTS
#══════════════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """
    Point d'entrée de l'API.
    
    Retourne les informations de base sur le service.
    """
    return "Retrouvez toute la documentation de l'API sur /docs"


@app.on_event("startup")
async def startup_event():
    """
    Chargement des ressources au démarrage de l'API.
    """
    print("🚀 Démarrage de l'API...")

    try:
        #Charger les ressources NLP
        load_nlp_ressources()

        #Charger le modèle
        load_ml_model()

        print("✅ API prête à recevoir des requêtes")

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE au démarrage : {e}")
        raise


@app.post("/predict", response_model=PredictionResponse, tags=["Détection"])
async def predict_single_text(input_data: TextInput):
    """
    Analyse un seul texte pour déterminé s'il a été généré par une IA.

    **Format accepté pour la requête :**
    json:
    `{
        "text": "Ici se trouve le texte."
    }`

    **Format accepté pour le champ "text" :**
    str `"Voici le texte."`

    **Retourne :**
    - `is_ai_generated` : 1 si le texte est généré par IA, 0 s'il est écrit par un humain

    - `message` : Message explicatif en langage naturel
    """

    #Preprocessing
    ### MODIF ### features_df = preprocessor(input_data.text)

    #Prédiction
    ### MODIF ### prediction = int(model.predict(features_df)[0])
    prediction = int(model.predict([input_data.text])[0])

    #Message explicatif
    if prediction == 1:
        message = "Cet avis semble avoir été généré par une intelligence artificielle."
    else:
        message = "Ce texte semble avoir été écrit par un humain."

    return PredictionResponse(
        is_ai_generated=prediction,
        message=message
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Détection"])
async def predict_batch_texts(input_data: BatchTextInput):
    """
    Analyse plusieurs textes simultanément pour déterminer s'ils ont été générés par une IA.

    **Limite :** Maximum 10,000 textes par requête

    **Format de la requête:**
    json:
    `{
        "texts": ["text1", "text2", ...]
    }`

    **Formats acceptés pour le champ "texts" :**
    - Liste de strings: `["text1", "text2", ...]`

    - Liste de dictionnaires: `[{"text": "text1}, {"texte": "text2}, ...]`

    **Retourne :**
    `predictions` : Liste des prédictions (1 = IA, 0 = Humain) dans le même ordre que les textes soumis
    """

    #Extraction des textes selon le format
    texts = []
    for item in input_data.texts:
        #Si c'est une liste de textes
        if isinstance(item, str):
            texts.append(item)
        #Sinon, si c'est une liste de dictionnaire
        elif isinstance(item, dict):
            #Et si ce dictionnaire contient une clé "text"
            if "text" in item:
                texts.append(item["text"])
            #Ou sinon si ce dictionnaire contient une clé "texte"
            elif "texte" in item:
                texts.append(item['texte'])
            #Sinon, on lève une erreur
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Les dictionnaires doivent contenir une clé 'texte."
                )
        #Si ce n'est ni une liste de texte ni une liste de dictionnaire, on lève une erreur
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporté: {type(item)}. Utilisez des strings ou des dictionnaires."
            )

    #Preprocessing
    ### MODIF ###  features_list = pd.concat([preprocessor(text) for text in texts], ignore_index=True)

    #Predictions
    ### MODIF ### predictions = model.predict(features_list)
    predictions = model.predict(texts)

    #Conversion en liste
    predictions = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

    return BatchPredictionResponse(predictions=predictions)