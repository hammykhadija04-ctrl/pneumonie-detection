import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io

SEUIL_HAUT = 0.85
SEUIL_BAS = 0.30


@st.cache_data
def image_en_base64(chemin):
    img = Image.open(chemin).convert("RGB")
    img.thumbnail((400, 400))
    tampon = io.BytesIO()
    img.save(tampon, format="JPEG", quality=80)
    return base64.b64encode(tampon.getvalue()).decode()

st.set_page_config(page_title="Détection de pneumonie")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #2a0a4a 0%, #0b0618 50%, #101a4a 100%);
}
.sous-titre {
    text-align: center;
    color: #d946ef;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0;
}
.titre {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 700;
}
.description {
    text-align: center;
    color: #d946ef;
    font-size: 0.95rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.stButton > button {
    border-radius: 999px;
    background: linear-gradient(90deg, #a21caf, #d946ef);
    color: white;
    border: none;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #d946ef, #f0abfc);
    color: #0b0618;
}

.defilement {
    overflow: hidden;
    width: 100%;
    margin: 1rem 0 1.5rem 0;
}
.piste {
    display: flex;
    width: max-content;
    animation: defiler 30s linear infinite;
}
.piste img {
    height: 180px;
    width: auto;
    border-radius: 12px;
    margin-right: 12px;
}
.defilement:hover .piste {
    animation-play-state: paused;
}
@keyframes defiler {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}




</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="titre">Détection de pneumonie</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Analyse de radiographies thoraciques par deep learning</p>',
            unsafe_allow_html=True)

st.warning("Projet pédagogique : ne remplace pas un diagnostic médical.")

if os.path.isdir("exemples"):
    noms = [f for f in sorted(os.listdir("exemples"))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and not f.startswith("._")][:6]

    balises = "".join(
        f'<img src="data:image/jpeg;base64,{image_en_base64(os.path.join("exemples", n))}">'
        for n in noms
    )

    st.markdown(f'<div class="defilement"><div class="piste">{balises}{balises}</div></div>',
                unsafe_allow_html=True)
if st.button("Comment interpréter le résultat ?"):
    st.info("Vert : aucun signe détecté. Orange : résultat incertain. "
            "Rouge : signes évocateurs de pneumonie. "
            "Ce résultat ne remplace jamais l'avis d'un médecin.")

@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model("pneumonia_model_v2.keras")

model = charger_modele()

fichier = st.file_uploader("Choisir une radio", type=["jpg", "jpeg", "png"])

if fichier is not None:
    try:
        img = Image.open(fichier).convert("RGB").resize((224, 224))
    except Exception:
        st.error("Ce fichier n'est pas une image valide. Choisis une radiographie JPG ou PNG.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Image envoyée")

    with col2:
        with st.spinner("Analyse en cours..."):
            x = np.expand_dims(np.array(img, dtype="float32"), axis=0)
            proba = float(model.predict(x)[0][0])

        st.metric("Probabilité de pneumonie", f"{proba:.1%}")
        st.progress(proba)

        if proba > SEUIL_HAUT:
            st.error("Signes évocateurs de PNEUMONIE")
        elif proba > SEUIL_BAS:
            st.warning("Résultat incertain : l'image est proche de la limite de décision. "
                       "Ne pas se fier à cette prédiction.")
        else:
            st.success("Aucun signe détecté")