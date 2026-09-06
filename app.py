"""
TP - Système RAG Local (Clone de NotebookLM)

Dépendances :
    pip install streamlit langchain langchain-community numpy \
                sentence-transformers pymupdf langchain-ollama langchain-huggingface

Lancement :
    streamlit run app.py

Il faut qu'Ollama tourne en fond avec un modèle chargé (ex : ollama run mistral).

A propos de la base vectorielle : j'utilise une classe maison (SimpleVectorStore,
numpy + similarité cosinus) à la place de ChromaDB. ChromaDB tire grpc en
dépendance pour sa télémétrie, et ça a posé des problèmes d'installation chez moi
sous Windows (DLL bloquée / pas de compilateur pour les vieilles versions). Le
résultat est le même : stockage local des vecteurs et recherche par similarité,
juste sans cette dépendance qui pose souci.
"""

import os
import pickle
import tempfile

import numpy as np
import streamlit as st

# Chargement de documents
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

# Découpage de texte
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# LLM local (Ollama) + prompt
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =====================================================================
# CONFIG
# =====================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistral"
VECTOR_STORE_PATH = "./vector_store.pkl"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NB_CHUNKS_RETRIEVED = 4


# =====================================================================
# BASE VECTORIELLE MAISON (remplace ChromaDB)
# =====================================================================

class SimpleVectorStore:
    """
    Base vectorielle minimaliste basée sur numpy : une matrice de vecteurs
    + la liste des Document LangChain correspondants. La recherche se fait
    par similarité cosinus (produit scalaire normalisé), ce qui suffit
    largement pour quelques milliers de chunks en local.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vectors = None   # matrice (n_chunks, dim_embedding)
        self.documents = []

    @classmethod
    def from_documents(cls, documents, embedding_model):
        """Construit la base à partir des chunks déjà découpés."""
        store = cls(embedding_model)
        texts = [doc.page_content for doc in documents]

        # tout vectoriser en un seul appel, plus efficace que chunk par chunk
        raw_vectors = embedding_model.embed_documents(texts)

        store.vectors = np.array(raw_vectors, dtype=np.float32)
        store.documents = documents
        return store

    def similarity_search(self, query, k=4):
        """Renvoie les k chunks les plus proches de la requête (cosinus)."""
        if self.vectors is None or len(self.documents) == 0:
            return []

        query_vector = np.array(self.embedding_model.embed_query(query), dtype=np.float32)

        # cosinus = produit scalaire / (norme des docs * norme de la requête)
        doc_norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        denominator = doc_norms * query_norm + 1e-10  # anti division par zéro

        similarities = (self.vectors @ query_vector) / denominator

        k = min(k, len(self.documents))
        top_indices = np.argsort(-similarities)[:k]

        return [self.documents[i] for i in top_indices]

    def save(self, path):
        """Sauvegarde vecteurs + documents dans un fichier pickle local."""
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors, "documents": self.documents}, f)

    @classmethod
    def load(cls, path, embedding_model):
        """Recharge une base sauvegardée précédemment."""
        store = cls(embedding_model)
        with open(path, "rb") as f:
            data = pickle.load(f)
        store.vectors = data["vectors"]
        store.documents = data["documents"]
        return store


st.set_page_config(page_title="RAG Local - Clone NotebookLM", layout="wide")

# Streamlit ne propose pas de mettre la sidebar à droite par défaut, donc
# petit hack CSS : on inverse l'ordre flexbox du conteneur principal.
st.markdown(
    """
    <style>
    div[data-testid="stAppViewContainer"] > section[data-testid="stSidebar"] {
        order: 2;
    }
    div[data-testid="stAppViewContainer"] > div.main {
        order: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# RESSOURCES EN CACHE (une seule instanciation pour toute la session)
# =====================================================================

@st.cache_resource(show_spinner="Chargement du modèle d'embeddings...")
def get_embedding_model():
    """Modèle d'embeddings HuggingFace, gardé en cache pour ne pas le recharger."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Connexion au LLM local (Ollama)...")
def get_llm():
    """Client vers le LLM Ollama, mis en cache pour la même raison."""
    return ChatOllama(model=LLM_MODEL_NAME, temperature=0.1)


# =====================================================================
# ÉTAT DE SESSION
# =====================================================================

# historique du chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# base vectorielle : si un vector_store.pkl existe déjà d'une session
# précédente, autant le recharger plutôt que forcer une réindexation
if "vector_store" not in st.session_state:
    if os.path.exists(VECTOR_STORE_PATH):
        st.session_state.vector_store = SimpleVectorStore.load(
            VECTOR_STORE_PATH, get_embedding_model()
        )
        st.session_state.indexed = True
    else:
        st.session_state.vector_store = None
        st.session_state.indexed = False


# =====================================================================
# INGESTION / INDEXATION
# =====================================================================

def load_and_split_documents(uploaded_files):
    """
    Pipeline d'ingestion (étape 2 du TP). Pour chaque fichier uploadé :
    écriture temporaire sur disque (les loaders LangChain veulent un
    chemin, pas juste des bytes), choix du loader selon l'extension,
    extraction du texte, puis découpage en chunks avec overlap.

    Renvoie la liste de tous les chunks (objets Document), prêts à
    être passés au modèle d'embeddings.
    """
    all_chunks = []

    # séparateurs dans l'ordre de préférence : on essaie de couper aux
    # frontières naturelles (paragraphe, ligne, phrase) avant de forcer
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if suffix == ".pdf":
                loader = PyMuPDFLoader(tmp_path)
            elif suffix in (".txt", ".md"):
                loader = TextLoader(tmp_path, encoding="utf-8")
            else:
                st.warning(f"Type de fichier non supporté, ignoré : {uploaded_file.name}")
                continue

            raw_documents = loader.load()

            # le loader garde le chemin temporaire comme "source" par défaut,
            # on le remplace par le vrai nom du fichier pour l'affichage
            for doc in raw_documents:
                doc.metadata["source"] = uploaded_file.name

            chunks = text_splitter.split_documents(raw_documents)
            all_chunks.extend(chunks)

        finally:
            os.remove(tmp_path)

    return all_chunks


def build_vector_store(chunks, embedding_model):
    """Vectorise les chunks et sauvegarde la base sur disque (étape 2, suite)."""
    vector_store = SimpleVectorStore.from_documents(
        documents=chunks,
        embedding_model=embedding_model,
    )
    vector_store.save(VECTOR_STORE_PATH)
    return vector_store


def get_indexed_sources(vector_store):
    """Renvoie la liste triée des noms de fichiers déjà présents dans la base."""
    if vector_store is None:
        return []
    sources = {doc.metadata.get("source", "source inconnue") for doc in vector_store.documents}
    return sorted(sources)


# =====================================================================
# RECHERCHE / GÉNÉRATION
# =====================================================================

def semantic_search(vector_store, query, k=NB_CHUNKS_RETRIEVED):
    """Mode "recherche sémantique pure" (étape 3) : pas de LLM, juste le retrieval."""
    results = vector_store.similarity_search(query, k=k)
    return results


# Prompt qui force le LLM à rester dans le contexte fourni, et à dire
# clairement qu'il ne sait pas plutôt que d'halluciner.
RAG_PROMPT_TEMPLATE = """Tu es un assistant qui répond aux questions en te basant UNIQUEMENT sur le contexte fourni ci-dessous, extrait des documents de l'utilisateur.

Règles strictes :
- Réponds exclusivement à partir des informations présentes dans le contexte.
- Si la réponse ne se trouve pas dans le contexte, réponds explicitement que tu ne disposes pas de cette information dans les documents fournis.
- N'invente aucune information et ne fais pas appel à tes connaissances générales.
- Réponds dans la même langue que la question, de manière claire et concise.

Contexte :
{context}

Question :
{question}

Réponse :"""


def format_context(documents):
    """Assemble les chunks récupérés en un bloc de texte, avec la source de chacun."""
    parts = []
    for doc in documents:
        source = doc.metadata.get("source", "source inconnue")
        parts.append(f"[Source : {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_rag_answer(vector_store, llm, query, k=NB_CHUNKS_RETRIEVED):
    """
    Mode "RAG complet" (étape 4) : récupère les chunks pertinents, les
    injecte dans le prompt via PromptTemplate, envoie le tout au LLM
    (chaîne LCEL prompt -> llm -> parseur), et renvoie la réponse plus
    les documents utilisés comme contexte (pour l'affichage des sources).
    """
    retrieved_docs = vector_store.similarity_search(query, k=k)
    context_text = format_context(retrieved_docs)

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"context": context_text, "question": query})

    return answer, retrieved_docs


# =====================================================================
# INTERFACE : SIDEBAR
# =====================================================================

with st.sidebar:
    st.header("Importation des documents")

    uploaded_files = st.file_uploader(
        "Charger vos documents (PDF, TXT ou Markdown)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    index_button = st.button(" Importer", use_container_width=True)

    if index_button:
        if not uploaded_files:
            st.warning("Veuillez charger au moins un fichier avant d'importer.")
        else:
            with st.spinner("Extraction, découpage et vectorisation en cours..."):
                embedding_model = get_embedding_model()
                chunks = load_and_split_documents(uploaded_files)

                if len(chunks) == 0:
                    st.error("Aucun contenu exploitable n'a été extrait des fichiers.")
                else:
                    vector_store = build_vector_store(chunks, embedding_model)
                    st.session_state.vector_store = vector_store
                    st.session_state.indexed = True
                    st.success(f"Importation terminée : {len(chunks)} chunks créés à partir de {len(uploaded_files)} fichier(s).")

    # liste des documents actuellement dans la base (utile pour savoir ce
    # qui est déjà indexé sans avoir à tout ré-importer)
    indexed_sources = get_indexed_sources(st.session_state.vector_store)
    if indexed_sources:
        with st.expander(f"📑 Documents indexés ({len(indexed_sources)})"):
            for source_name in indexed_sources:
                st.markdown(f"- `{source_name}`")

    st.divider()

    st.header(" Mode de fonctionnement")
    llm_enabled = st.toggle(
        "Activer le LLM (mode RAG complet)",
        value=False,
        help="Désactivé : recherche sémantique brute uniquement. Activé : réponse générée par le LLM local.",
    )

    if llm_enabled:
        st.info("Mode actif : **Assistant RAG complet** ")
    else:
        st.info("Mode actif : **Recherche Sémantique pure** ")

    st.divider()
    st.caption(f"Modèle d'embeddings : `{EMBEDDING_MODEL_NAME}`")
    st.caption(f"Modèle LLM (Ollama) : `{LLM_MODEL_NAME}`")


# =====================================================================
# INTERFACE : CHAT
# =====================================================================

st.title(" RAG Local — Clone de NotebookLM")
st.caption("Interrogez vos propres documents, 100% en local, sans aucun appel à une API externe.")

# on réaffiche tout l'historique à chaque rerun, Streamlit ne le fait pas tout seul
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander(" Sources utilisées"):
                for i, doc in enumerate(message["sources"], start=1):
                    source_name = doc.metadata.get("source", "source inconnue")
                    st.markdown(f"**Extrait {i} — source : `{source_name}`**")
                    st.markdown(f"> {doc.page_content}")

user_query = st.chat_input("Posez une question sur vos documents...")

if user_query:
    if st.session_state.vector_store is None:
        with st.chat_message("assistant"):
            st.error("Aucun document indexé. Veuillez charger et indexer des documents via la barre latérale avant de poser une question.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if not llm_enabled:
                # -------- Recherche sémantique pure --------
                with st.spinner("Recherche des extraits les plus pertinents..."):
                    results = semantic_search(st.session_state.vector_store, user_query)

                if not results:
                    response_text = "Aucun extrait pertinent n'a été trouvé dans les documents indexés."
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    response_text = f"Voici les {len(results)} extraits les plus pertinents trouvés dans vos documents :"
                    st.markdown(response_text)
                    for i, doc in enumerate(results, start=1):
                        source_name = doc.metadata.get("source", "source inconnue")
                        st.markdown(f"**Extrait {i} — source : `{source_name}`**")
                        st.markdown(f"> {doc.page_content}")
                    # on garde les sources dans le message pour le réaffichage après rerun
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": results,
                    })

            else:
                # -------- Assistant RAG complet --------
                with st.spinner("Génération de la réponse par le LLM local..."):
                    llm = get_llm()
                    try:
                        answer, sources = generate_rag_answer(
                            st.session_state.vector_store, llm, user_query
                        )
                    except Exception as e:
                        answer = (
                            "⚠️ Impossible de contacter le LLM local via Ollama. "
                            "Vérifiez qu'Ollama est bien lancé et que le modèle "
                            f"`{LLM_MODEL_NAME}` est disponible.\n\nDétail : {e}"
                        )
                        sources = []

                st.markdown(answer)

                # transparence : on montre les extraits qui ont servi de contexte
                if sources:
                    with st.expander("📚 Sources utilisées pour cette réponse"):
                        for i, doc in enumerate(sources, start=1):
                            source_name = doc.metadata.get("source", "source inconnue")
                            st.markdown(f"**Extrait {i} — source : `{source_name}`**")
                            st.markdown(f"> {doc.page_content}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
