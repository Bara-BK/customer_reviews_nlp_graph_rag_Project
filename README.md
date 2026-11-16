# 🍽️ Analyse et recommandation de produits à partir des avis clients

**NLP + graphes + GNN/GTN + Graph-RAG sur Amazon Fine Food Reviews**
Projet S4 — *Advanced Learning for Text and Graph Data* (Unit: IL) — ESPRIT, 5ERP-BI, 2025-2026

> Encadrant : Hajer GHODHBANI

## Objectif

Combiner le traitement du langage (NLP) et les graphes pour analyser des avis clients : détecter le sentiment, découvrir des thèmes et des communautés d'utilisateurs, prédire des notes avec des GNN / un Graph Transformer (GTN), recommander des produits et répondre à des questions sur le graphe (Graph-RAG). Les livrables sont rassemblés dans une application web Streamlit.

## Données

- **Dataset :** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (SNAP / Kaggle) : 568 454 avis, 256 059 utilisateurs, 74 258 produits (1999-2012). Seul le fichier `Reviews.csv` est utilisé.
- **Les données ne sont pas dans ce dépôt** (≈ 300 Mo) : téléchargez `archive.zip` depuis Kaggle (compte gratuit).
- **Nettoyage :** 392 482 avis conservés (69 %) après suppression des doublons, des votes incohérents et des doublons (utilisateur, produit).
- **Graphe exploité :** filtre *3-core* → **79 293 avis, 13 118 utilisateurs, 8 372 produits** ; découpage train/test 80/20 stratifié (même découpage pour toutes les tâches).

## Pipeline

```
Reviews.csv → nettoyage → 3-core → train/test
   ├─ NLP : TF-IDF → embeddings MiniLM → DistilBERT (sentiment) ; LDA + K-Means (thèmes)
   ├─ Graphe biparti utilisateurs–produits → communautés Louvain
   ├─ GNN / GTN : GraphSAGE, GAT, TransformerConv (prédiction de notes)
   ├─ Recommandation : popularité, contenu, Item-kNN, communauté, hybride
   └─ Graph-RAG : graphe de connaissances + retriever sémantique → mini Q&A
                                   ↓
                      Application Streamlit (5 pages)
```

## Résultats (jeu de test, exécution sur Colab T4)

| Tâche | Résultat | Référence |
|---|---|---|
| Sentiment (3 classes) | **DistilBERT : F1 macro 0,730** | TF-IDF 0,709 ; embeddings MiniLM 0,596 |
| Communautés (Louvain) | **34 communautés, modularité 0,562** | NMI avec clusters texte : 0,100 |
| Prédiction de notes | **GTN : RMSE 1,074** | baseline biais 1,078 ; GraphSAGE 1,102 ; GAT 1,101 |
| Recommandation top-10 | **Hybride : Hit@10 0,146, NDCG@10 0,066** | Popularité 0,077 / 0,025 ; Item-kNN 0,118 / 0,053 |
| Graph-RAG | **Pureté thématique 0,99** | baseline sans retrieval 0,05 |

Points d'honnêteté : le gain de DistilBERT sur TF-IDF est modeste (+0,021 de F1 macro) ; le gain du GTN sur la baseline de biais est très faible (−0,004 de RMSE, un seul découpage et une seule graine) et GraphSAGE et GAT font moins bien que cette baseline. Les tableaux complets sont dans [`results/`](results/).

## Structure du dépôt

```
avis-clients-nlp-graph/
├── README.md
├── .gitignore
├── requirements.txt              # dépendances de l'application Streamlit
├── requirements-notebook.txt     # dépendances pour rejouer le notebook en local
├── notebooks/
│   └── Projet_Analyse_Reco_Avis_Clients.ipynb   # notebook exécuté, avec les sorties
├── app/
│   ├── app.py                    # application Streamlit
│   └── app_data/                 # artefacts de l'application (voir ci-dessous)
│       ├── app_data.joblib       # graphe, embeddings, communautés (≈ 69 Mo)
│       └── sentiment_tfidf.joblib# modèle de sentiment TF-IDF (≈ 2 Mo)
├── results/                      # tableaux de résultats (CSV) et graphe (.gexf, Gephi)
└── reports/                      # rapport et présentation de soutenance
```

## Démarrage rapide

### 1. Rejouer le notebook (Google Colab, recommandé)
1. Téléchargez `archive.zip` depuis Kaggle et placez-le dans votre Google Drive (`Mon Drive/archive.zip`).
2. Ouvrez `notebooks/Projet_Analyse_Reco_Avis_Clients.ipynb` dans Colab, puis *Exécution → Modifier le type d'exécution → GPU T4*.
3. Exécutez les cellules dans l'ordre (≈ 25–35 min). Seule l'installation de `sentence-transformers` et `torch_geometric` est nécessaire (déjà dans la première cellule).
4. La section 10 génère `streamlit_app.zip` (application + `app_data/`).

### 2. Lancer l'application en local
```bash
pip install -r requirements.txt
# copier app_data.joblib et sentiment_tfidf.joblib (issus de streamlit_app.zip) dans app/app_data/
streamlit run app/app.py
```
Pages : Accueil · Sentiment · Recommandation · Questions/Réponses (Graph-RAG) · Communautés.
Si les versions de `scikit-learn`, `pandas` ou `numpy` diffèrent de celles de Colab, le chargement des `.joblib` peut échouer : alignez-les sur celles de votre session Colab (`!pip list`).

### 3. Déployer sur Streamlit Community Cloud
1. Poussez ce dépôt sur GitHub (avec `app/app_data/*.joblib`).
2. Sur [share.streamlit.io](https://share.streamlit.io) : *New app* → votre dépôt → fichier principal `app/app.py` → *Deploy*.
3. Ne poussez **pas** le dossier `distilbert_sentiment/` (≈ 268 Mo, au-dessus de la limite de 100 Mo de GitHub) : l'application bascule alors automatiquement sur le modèle TF-IDF. Pour garder DistilBERT, hébergez-le sur Hugging Face Hub.

## Limites

- Filtre *3-core* : les utilisateurs très peu actifs sont exclus, le démarrage à froid n'est pas traité.
- La classe « neutre » (3★) est ambiguë et reste la plus difficile à prédire (F1 ≤ 0,52).
- Le dataset ne contient pas de noms de produits : les produits sont décrits par des identifiants et des mots-clés.
- Les GNN sont évalués sur un seul découpage et une seule graine (pas de validation croisée).
- Le Graph-RAG est évalué par une pureté lexicale, non par un jugement humain ; la recherche sémantique peut rapprocher des produits voisins mais non conformes à la requête (ex. torréfaction légère pour une requête « dark roast »).

## Technologies

Python · pandas · scikit-learn · PyTorch · PyTorch Geometric (GraphSAGE, GAT, TransformerConv) · Hugging Face Transformers (DistilBERT) · sentence-transformers (MiniLM) · NetworkX (Louvain) · Streamlit

## Licence et données

Le code de ce dépôt est destiné à un usage pédagogique. Le dataset Amazon Fine Food Reviews appartient à ses auteurs (SNAP / Stanford) et n'est pas redistribué ici.
