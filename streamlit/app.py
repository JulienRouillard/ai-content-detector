import streamlit as st
import requests
import pandas as pd
import io

st.set_page_config(page_title="AI Review Detector", page_icon="🤖")

st.title("🤖 AI Review Detector")
st.markdown("**Détectez si un avis a été écrit par un humain ou généré par une IA**")
st.markdown("---")

st.info("💡 Copiez-collez le texte de l'avis dans la zone ci-dessous et cliquez sur Analyser.")

text_input = st.text_area("Texte à analyser :", height=200)

if st.button("🔍 Analyser", type="primary"):
    response = requests.post("http://127.0.0.1:8000/predict", json={"text": text_input})
    result = response.json()
    
    st.markdown("---")
    if result["is_ai_generated"] == 1:
        st.error(f"🤖 **Généré par IA**\n\n{result['message']}")
    else:
        st.success(f"✍️ **Écrit par un humain**\n\n{result['message']}")


# Section batch
st.markdown("---")
st.markdown("## 📁 Analyse par lot")
st.markdown("**Analysez plusieurs avis en uploadant un fichier**")

uploaded_file = st.file_uploader(
    "Choisissez un fichier (CSV, Excel ou JSON)",
    type=["csv", "xlsx", "json"],
    help="Le fichier doit contenir une colonne avec les textes à analyser"
)

if uploaded_file is not None:
    # Charger le fichier selon son type
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        
        st.success(f"✅ Fichier chargé : {uploaded_file.name} ({len(df)} lignes)")
        st.dataframe(df.head())
        
        # Sélection de la colonne contenant le texte
        text_column = st.selectbox(
            "Sélectionnez la colonne contenant les textes à analyser :",
            options=df.columns.tolist()
        )
        
        # Bouton d'analyse
        if st.button("🚀 Analyser le fichier", type="primary"):
            with st.spinner("Analyse en cours..."):
                # Préparer les données pour l'API
                texts = df[text_column].tolist()
                
                # Appel API
                response = requests.post(
                    "http://127.0.0.1:8000/predict-batch",
                    json={"texts": texts}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    predictions = result["predictions"]
                    
                    # Ajouter les prédictions au DataFrame
                    df["prediction"] = predictions
                    df["prediction"] = df["prediction"].map({1: "IA", 0: "Humain"})
                    
                    st.success("✅ Analyse terminée !")
                    
                    # Afficher les résultats
                    st.dataframe(df)
                    
                    # Statistiques
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🤖 Textes IA", (df["prediction"] == "IA").sum())
                    with col2:
                        st.metric("✍️ Textes Humains", (df["prediction"] == "Humain").sum())
                    
                    # Bouton de téléchargement
                    if uploaded_file.name.endswith('.csv'):
                        output = df.to_csv(index=False).encode('utf-8')
                        file_extension = 'csv'
                        mime_type = 'text/csv'
                    elif uploaded_file.name.endswith('.xlsx'):
                        output = io.BytesIO()
                        df.to_excel(output, index=False)
                        output.seek(0)
                        output = output.getvalue()
                        file_extension = 'xlsx'
                        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    else:  # json
                        output = df.to_json(orient='records').encode('utf-8')
                        file_extension = 'json'
                        mime_type = 'application/json'
                    
                    st.download_button(
                        label="📥 Télécharger les résultats",
                        data=output,
                        file_name=f"resultats_analyse.{file_extension}",
                        mime=mime_type
                    )
                else:
                    st.error(f"❌ Erreur API : {response.status_code}")
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier : {e}")