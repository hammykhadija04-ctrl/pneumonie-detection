# Détection de pneumonie sur radiographies thoraciques

Application web qui analyse une radiographie thoracique par apprentissage profond et indique l'un de trois résultats : **aucun signe détecté**, **résultat incertain** ou **signes évocateurs de pneumonie**.

> **Projet pédagogique : cet outil ne remplace pas un diagnostic médical et n'est pas un dispositif médical.**

## Fonctionnement

- Modèle : MobileNetV2 (pré-entraîné sur ImageNet), transfert d'apprentissage puis réglage fin, enregistré dans `pneumonia_model_v2.keras`.
- L'image déposée (JPEG ou PNG) est redimensionnée en 224×224, puis le modèle calcule une probabilité de pneumonie.
- Deux seuils séparent les trois résultats :

| Probabilité p | Résultat |
|---|---|
| p ≤ 30 % | Aucun signe détecté |
| 30 % < p ≤ 85 % | Résultat incertain |
| p > 85 % | Signes évocateurs de pneumonie |

- Aucune image n'est enregistrée sur le disque.

## Résultats sur l'ensemble de test (624 images)

Modèle final (réglage fin), avec un seuil de décision de 0,5 (voir le carnet pour le détail) :

| Classe | Précision | Sensibilité | F1 |
|---|---|---|---|
| Normal | 0,96 | 0,62 | 0,75 |
| Pneumonie | 0,81 | 0,98 | 0,89 |

La sensibilité sur la classe pneumonie dépasse l'objectif de 80 % fixé dans le cahier des charges.

## Installation

Python 3.12 est recommandé.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

## Structure du projet

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit |
| `pneumonia_model_v2.keras` | Modèle entraîné |
| `test_lots.py` | Teste le modèle sur un dossier d'images (`test_complet/NORMAL` et `test_complet/PNEUMONIA`) |
| `.streamlit/config.toml` | Thème de l'interface |
| `docs/` | Rapport : cahier des charges et conception |
| `exemples/` | Six radiographies affichées dans le bandeau défilant |
| `notebooks/AI_Medical_Image_Classification.ipynb` | Carnet Google Colab : exploration, entraînement (transfert puis réglage fin), évaluation et seuils |

## Données

Jeu public de radiographies thoraciques (Kaggle), 5 856 images en deux classes (normal, pneumonie). Il n'est pas inclus dans ce dépôt. Pour utiliser `test_lots.py`, placez les images de test dans `test_complet/NORMAL` et `test_complet/PNEUMONIA`.

## Auteure

Khadija Hammi, 2026.
