# -*- coding: utf-8 -*-
# Application Streamlit : Analyse et recommandation de produits a partir des avis clients
# Lancement :  streamlit run app.py
import os, re, collections
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import streamlit as st
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).parent
DATA = HERE / 'app_data'
LABELS = ['négatif', 'neutre', 'positif']
METHODS = ['Hybride', 'Item-kNN', 'Content (embeddings)', 'Communauté (graphe)', 'Popularité']


# ------------------------------------------------------------------ moteur (logique sans interface)
def minmax(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-9)


class Engine:
    # Charge les artefacts du notebook et expose : sentiment, recommandation, Graph-RAG.
    def __init__(self, d, sent):
        self.core = d['core'].reset_index(drop=True)
        self.emb = d['emb'].astype('float32')
        self.prod_ids, self.user_ids = list(d['prod_ids']), list(d['user_ids'])
        self.pidx = {p: i for i, p in enumerate(self.prod_ids)}
        self.uidx = {u: i for i, u in enumerate(self.user_ids)}
        self.node_comm, self.topic_kw, self.prod_topic = d['node_comm'], d['topic_kw'], np.asarray(d['prod_topic'])
        self.comm_profile, self.results = d['comm_profile'], d['results']
        self.sent = sent
        c = self.core
        self.n_u, self.n_p = len(self.user_ids), len(self.prod_ids)

        # statistiques produits (note bayesienne, C = 10 avis fictifs)
        m = c.Score.mean()
        g = c.groupby('pid').agg(n=('Score', 'size'), mean=('Score', 'mean'), s=('Score', 'sum'))
        g['bayes'] = (10 * m + g.s) / (10 + g.n)
        self.pstats = g.reindex(range(self.n_p))

        # matrices utilisateur x produit
        pos = c[c.Score >= 4]
        ones = np.ones(len(pos), dtype='float32')
        self.R_pos = sp.csr_matrix((ones, (pos.uid.values, pos.pid.values)), shape=(self.n_u, self.n_p))
        self.R_seen = sp.csr_matrix((np.ones(len(c), dtype='float32'), (c.uid.values, c.pid.values)), shape=(self.n_u, self.n_p))
        self.pop = np.asarray(self.R_pos.sum(0)).ravel().astype(float)

        # content-based : profils utilisateur / produit = moyennes d'embeddings des avis positifs
        P = np.zeros((self.n_p, self.emb.shape[1]), dtype='float32'); np.add.at(P, pos.pid.values, self.emb[pos.index.values])
        U = np.zeros((self.n_u, self.emb.shape[1]), dtype='float32'); np.add.at(U, pos.uid.values, self.emb[pos.index.values])
        self.Pvec, self.Uvec = normalize(P), normalize(U)

        # item-kNN : similarite cosinus entre colonnes de R_pos
        Rn = normalize(self.R_pos, axis=0).tocsc().astype('float32')
        S = (Rn.T @ Rn).tocsr(); S.setdiag(0); S.eliminate_zeros(); self.S = S

        # communautes
        self.u_c = np.full(self.n_u, -1); self.p_c = np.full(self.n_p, -1)
        for node, ci in self.node_comm.items():
            if node.startswith('U:') and node[2:] in self.uidx: self.u_c[self.uidx[node[2:]]] = ci
            elif node.startswith('P:') and node[2:] in self.pidx: self.p_c[self.pidx[node[2:]]] = ci
        self.comm_pop = {ci: np.where(self.p_c == ci, self.pop, 0.0) for ci in set(self.p_c[self.p_c >= 0])}

        # voisinages du graphe (positifs) pour la recherche d'utilisateurs similaires
        self.user_liked = pos.groupby('uid').pid.apply(set).to_dict()
        self.prod_likers = pos.groupby('pid').uid.apply(list).to_dict()

        # retriever semantique (MiniLM) si disponible, sinon TF-IDF sur le texte
        self.retriever = 'tfidf'
        try:
            if self.emb.shape[1] == 384:
                from sentence_transformers import SentenceTransformer
                self.st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); self.st.max_seq_length = 128
                self.retriever = 'MiniLM'
        except Exception:
            pass
        if self.retriever == 'tfidf':
            self.tf = TfidfVectorizer(stop_words='english', min_df=3, sublinear_tf=True, max_features=60000)
            self.Xtf = self.tf.fit_transform(c.text)

    # ---------------------------------------------------------------- fiches produit
    def product_card(self, p, query_sims=None):
        rows = np.where(self.core.pid.values == p)[0]
        sc = self.core.Score.values[rows]
        if query_sims is not None: best = rows[np.argmax(query_sims[rows])]
        else: best = rows[np.argmax(sc)]
        return dict(produit=self.prod_ids[p], note_moyenne=round(float(sc.mean()), 2), note_bayesienne=round(float(self.pstats.bayes[p]), 2),
                    nb_avis=int(len(rows)), theme=self.topic_kw[int(self.prod_topic[p])].split(',')[0].strip(),
                    mots_cles_theme=self.topic_kw[int(self.prod_topic[p])], extrait=self.core.text.values[best][:180])

    # ---------------------------------------------------------------- 1) sentiment
    def predict_sentiment(self, text, use_bert=False):
        if use_bert:
            probs = self._bert_probs(text)
            return probs, []
        X = self.sent['vec'].transform([text]); probs = self.sent['clf'].predict_proba(X)[0]
        cls = int(probs.argmax()); contrib = X.multiply(self.sent['clf'].coef_[cls]).tocoo()
        names = self.sent['vec'].get_feature_names_out()
        top = sorted(zip(contrib.col, contrib.data), key=lambda t: -t[1])[:8]
        return probs, [(names[i], float(v)) for i, v in top if v > 0]

    def _bert_probs(self, text):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if not hasattr(self, '_bert'):
            path = str(DATA / 'distilbert_sentiment')
            self._bert = (AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path).eval())
        tok, mdl = self._bert
        with torch.no_grad(): logits = mdl(**tok([text], truncation=True, max_length=128, return_tensors='pt')).logits
        return torch.softmax(logits, -1)[0].numpy()

    # ---------------------------------------------------------------- 2) recommandation
    def _score(self, u, method):
        if method == 'Popularité': return self.pop.copy()
        if method == 'Content (embeddings)': return self.Pvec @ self.Uvec[u]
        if method == 'Item-kNN': return (self.R_pos[u] @ self.S).toarray().ravel()
        if method == 'Communauté (graphe)':
            ci = self.u_c[u]; return self.comm_pop[ci].copy() if ci in self.comm_pop else self.pop.copy()
        return 0.5 * minmax(self._score(u, 'Item-kNN')) + 0.3 * minmax(self._score(u, 'Content (embeddings)')) + 0.2 * minmax(self._score(u, 'Communauté (graphe)'))

    def liked_products(self, user_id, k=6):
        u = self.uidx[user_id]; ps = list(self.R_pos[u].indices)[:k]
        return [self.product_card(p) for p in ps]

    def recommend(self, user_id, method='Hybride', k=5):
        u = self.uidx[user_id]
        s = self._score(u, method).astype(float); s[self.R_seen[u].indices] = -1e9
        top = np.argsort(-s)[:k]
        return [dict(rang=i + 1, score=round(float(s[p]), 3), **self.product_card(p)) for i, p in enumerate(top)]

    # ---------------------------------------------------------------- 3) Graph-RAG
    def _sims(self, query):
        if self.retriever == 'MiniLM': return self.emb @ self.st.encode([query], normalize_embeddings=True)[0].astype('float32')
        return (self.Xtf @ self.tf.transform([query]).T).toarray().ravel()

    def best_products(self, query, k=5, k_reviews=300, min_reviews=5):
        sims = self._sims(query); top = np.argpartition(-sims, k_reviews)[:k_reviews]
        hits = collections.Counter(self.core.pid.values[top])
        cand = [(p, h) for p, h in hits.items() if self.pstats.n[p] >= min_reviews and h >= 2]
        cand.sort(key=lambda t: (-self.pstats.bayes[t[0]], -t[1]))
        out = []
        for p, h in cand[:k]:
            card = self.product_card(p, sims); card['avis_pertinents'] = int(h); out.append(card)
        return out

    def similar_users(self, user_id, k=5):
        u = self.uidx[user_id]; liked = self.user_liked.get(u, set())
        votes = collections.Counter(v for p in liked for v in self.prod_likers.get(p, []) if v != u)
        scored = sorted(((v, c / (len(liked) + len(self.user_liked[v]) - c)) for v, c in votes.items()), key=lambda t: -t[1])[:k]
        return [dict(utilisateur=self.user_ids[v], jaccard=round(j, 3), produits_communs=len(liked & self.user_liked[v]),
                     meme_communaute=bool(self.u_c[v] == self.u_c[u] and self.u_c[u] >= 0)) for v, j in scored], len(liked)

    def answer(self, query, k=3):
        res = self.best_products(query, k)
        if not res: return f"Aucun produit fiable trouvé pour « {query} »."
        lines = [f"Pour « {query} », voici les produits les mieux notés (note bayésienne) :"]
        for r in res: lines.append(f"• {r['produit']} : {r['note_bayesienne']}/5 sur {r['nb_avis']} avis (thème : {r['theme']}). Extrait : « {r['extrait']}… »")
        return '\n'.join(lines)


# ------------------------------------------------------------------ chargement (mis en cache par Streamlit)
@st.cache_resource(show_spinner='Chargement du graphe, des embeddings et des modèles…')
def load_engine():
    d = joblib.load(DATA / 'app_data.joblib'); sent = joblib.load(DATA / 'sentiment_tfidf.joblib')
    return Engine(d, sent)


def show_cards(cards, cols=('produit', 'note_moyenne', 'note_bayesienne', 'nb_avis', 'theme', 'extrait')):
    st.dataframe(pd.DataFrame(cards)[[c for c in cols if c in cards[0]]], width='stretch', hide_index=True)


# ------------------------------------------------------------------ interface
def main():
    st.set_page_config(page_title='Avis clients : NLP + graphes', page_icon='🍽️', layout='wide')
    st.title('🍽️ Analyse et recommandation de produits à partir des avis clients')
    st.caption('Amazon Fine Food Reviews · NLP + graphes + GNN/GTN + Graph-RAG · Advanced Learning for Text and Graph Data')
    try:
        eng = load_engine()
    except FileNotFoundError:
        st.error("Artefacts introuvables : exécutez d'abord la section 10 du notebook (dossier `streamlit_app/app_data`)."); return

    page = st.sidebar.radio('Navigation', ['Accueil', 'Sentiment', 'Recommandation', 'Questions/Réponses (Graph-RAG)', 'Communautés'])
    st.sidebar.caption(f'Retriever : {eng.retriever}')

    # ---------------- accueil
    if page == 'Accueil':
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Avis', f'{len(eng.core):,}'); c2.metric('Utilisateurs', f'{eng.n_u:,}'); c3.metric('Produits', f'{eng.n_p:,}')
        c4.metric('Communautés', len(set(eng.node_comm.values())))
        st.subheader('Distribution des notes')
        st.bar_chart(eng.core.Score.value_counts().sort_index())
        for name, table in eng.results.items():
            st.subheader(name); st.dataframe(table, width='stretch')
        st.info('Utilisez le menu à gauche : classer un avis, recommander des produits, poser une question au graphe, explorer les communautés.')

    # ---------------- sentiment
    elif page == 'Sentiment':
        st.header('Classer le sentiment d\'un avis')
        txt = st.text_area('Saisissez un avis (en anglais)', value='Great taste but way too sweet, I would not buy it again.', height=120)
        has_bert = (DATA / 'distilbert_sentiment').exists()
        use_bert = st.checkbox('Utiliser DistilBERT fine-tuné', value=False) if has_bert else False
        if st.button('Analyser') and txt.strip():
            probs, words = eng.predict_sentiment(txt, use_bert)
            st.success(f'Sentiment prédit : **{LABELS[int(np.argmax(probs))]}** ({probs.max():.0%})')
            st.bar_chart(pd.Series(probs, index=LABELS))
            if words: st.write('Mots qui ont le plus pesé :', ', '.join(f'{w}' for w, _ in words))

    # ---------------- recommandation
    elif page == 'Recommandation':
        st.header('Recommander des produits à un utilisateur')
        active = eng.core.UserId.value_counts().index[:50].tolist()
        uid = st.selectbox('Utilisateur (les 50 plus actifs)', active)
        custom = st.text_input('… ou saisissez un UserId', value='')
        if custom.strip(): uid = custom.strip()
        method = st.selectbox('Méthode', METHODS)
        k = st.slider('Nombre de recommandations', 3, 10, 5)
        if uid not in eng.uidx: st.warning("Utilisateur absent du graphe (filtre 3-core).")
        else:
            st.subheader('Produits déjà appréciés'); show_cards(eng.liked_products(uid))
            if st.button('Recommander'):
                st.subheader(f'Top-{k} ({method})'); show_cards(eng.recommend(uid, method, k), ('rang', 'produit', 'note_moyenne', 'nb_avis', 'theme', 'extrait'))

    # ---------------- Graph-RAG
    elif page == 'Questions/Réponses (Graph-RAG)':
        st.header('Mini Q&A sur le graphe')
        mode = st.radio('Type de question', ['Quels produits sont les mieux notés pour… ?', 'Quels utilisateurs ont des goûts similaires ?'])
        if mode.startswith('Quels produits'):
            q = st.text_input('Type d\'aliment (en anglais)', value='strong dark roast coffee')
            if st.button('Répondre') and q.strip():
                st.text(eng.answer(q, 3)); show_cards(eng.best_products(q, 5), ('produit', 'note_bayesienne', 'nb_avis', 'avis_pertinents', 'theme', 'extrait'))
        else:
            active = eng.core.UserId.value_counts().index[:50].tolist()
            uid = st.selectbox('Utilisateur', active)
            if st.button('Trouver des goûts similaires'):
                res, n_liked = eng.similar_users(uid, 5)
                st.write(f'{n_liked} produits aimés par cet utilisateur.'); st.dataframe(pd.DataFrame(res), width='stretch', hide_index=True)

    # ---------------- communautes
    else:
        st.header('Communautés détectées (Louvain)')
        st.dataframe(eng.comm_profile, width='stretch', hide_index=True)
        if 'utilisateurs' in eng.comm_profile:
            st.bar_chart(eng.comm_profile.set_index('communaute')[['utilisateurs', 'produits']])
        ci = st.selectbox('Détail d\'une communauté', eng.comm_profile.communaute.tolist())
        ps = np.where(eng.p_c == ci)[0]
        top = ps[np.argsort(-eng.pstats.n.values[ps])][:10]
        st.write(f'{len(ps)} produits dans la communauté {ci}. Les plus commentés :')
        show_cards([eng.product_card(p) for p in top])


if __name__ == '__main__':
    main()
