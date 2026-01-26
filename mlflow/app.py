import streamlit as st
import time
import random

# Configuration de la page
st.set_page_config(
    page_title="AI Review Detector",
    page_icon="🤖",
    layout="centered"
)

# Titre
st.title("🤖 AI Review Detector")
st.markdown("""
Cette application vous aide à détecter si un **avis client** a été rédigé par un **humain** ou généré par une **intelligence artificielle**.
""")

# Champ de saisie
text_input = st.text_area("📝 Entrez un avis à analyser :", height=200)

# Fonction de prédiction simulée
def fake_predict(text: str):
    """
    Simule une prédiction de détection IA :
    - Si le texte contient certains mots-clés : IA
    - Sinon : Humain
    """
    ai_keywords = ["ChatGPT", "AI", "artificial intelligence", "language model", "generated", "OpenAI"]
    lower_text = text.lower()

    if any(word.lower() in lower_text for word in ai_keywords) or len(text) < 50:
        return 1, "Cet avis semble avoir été généré par une intelligence artificielle."
    else:
        return 0, "Ce texte semble avoir été écrit par un humain."

# Bouton pour analyser
if st.button("🔍 Analyser"):
    if not text_input.strip():
        st.warning("⚠️ Veuillez entrer un texte avant d’analyser.")
    else:
        with st.spinner("Analyse en cours..."):
            time.sleep(1)  # Simule un délai de traitement

            prediction, message = fake_predict(text_input)

        # Affichage du résultat
        st.markdown("---")
        if prediction == 1:
            st.error("🟥 IA détectée")
        else:
            st.success("🟩 Texte humain détecté")

        st.markdown(f"**🧠 Interprétation :** {message}")
