import os
import numpy as np
import tensorflow as tf
from PIL import Image

SEUIL_HAUT = 0.85
SEUIL_BAS = 0.30
DOSSIER_TEST = "test_complet"

model = tf.keras.models.load_model("pneumonia_model_v2.keras")

def zone(p):
    if p > SEUIL_HAUT:
        return "rouge"
    elif p > SEUIL_BAS:
        return "orange"
    else:
        return "vert"

for classe in ["NORMAL", "PNEUMONIA"]:
    chemin = os.path.join(DOSSIER_TEST, classe)
    fichiers = [f for f in os.listdir(chemin)
                if not f.startswith("._")
                and f.lower().endswith((".jpg", ".jpeg", ".png"))]

    compte = {"vert": 0, "orange": 0, "rouge": 0}

    for nom in fichiers:
        img = Image.open(os.path.join(chemin, nom)).convert("RGB").resize((224, 224))
        x = np.expand_dims(np.array(img, dtype="float32"), axis=0)
        p = float(model.predict(x, verbose=0)[0][0])
        compte[zone(p)] += 1

    total = len(fichiers)
    print(f"\n{classe} ({total} images)")
    for couleur in ["vert", "orange", "rouge"]:
        n = compte[couleur]
        print(f"  {couleur:6s} : {n:4d}  ({n / total:.1%})")