# main.py

import glob
import json
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from llama_cpp import Llama
import streamlit as st

from session_logger import log_session

# ------------------------------
# 1. UČITAVANJE I CHUNKANJE DOKUMENATA
# ------------------------------
docs = []
for filename in glob.glob('./knowledge_base/*'):
    if filename.endswith('.pdf'):
        loader = PyPDFLoader(filename)
    elif filename.endswith('.txt'):
        loader = TextLoader(filename, encoding="utf-8")
    else:
        continue
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
split_docs = splitter.split_documents(docs)

# ------------------------------
# 2. GENERIRANJE EMBEDDINGA I VEKTORSKA BAZA
# ------------------------------
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")

# ------------------------------
# 3. POKRETANJE LOKALNOG LLM-a
# ------------------------------
llm = Llama(
    model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=512,
    n_threads=8,
    verbose=False,
    )

def chat(query, k=1):
    results = db.similarity_search(query, k=k)
    context = "\n".join([r.page_content for r in results])
    prompt = f"Na temelju donjeg konteksta odgovori jasno i jednostavno na korisnikovo pitanje:\n\nKontekst:\n{context}\n\nPitanje: {query}\nOdgovor (završi cijelu rečenicu samo na hrvatskom jeziku):"
    output = llm(prompt=prompt, max_tokens=256, stop=["Kontekst:", "Pitanje:"])
    answer = output["choices"][0]["text"].strip()
    
    log_session(query, context, answer)
    
    return answer

# ------------------------------
# 4. STREAMLIT CHAT SUČELJE
# ------------------------------ 
def load_logs(log_path="session_logs.jsonl"):
    logs = []
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                logs.append(entry)
    except FileNotFoundError:
        pass
    return logs

st.title("AI edukacija o ulaganju – Chatbot")

tab1, tab2 = st.tabs(["Chat", "Pregled sesija"])

with tab1:
    user_query = st.text_input("Postavi pitanje iz područja ulaganja:")
    if user_query:
        answer = chat(user_query)
        st.write(answer)

with tab2:
    logs = load_logs()
    if logs:
        search = st.text_input("Pretraži po pitanju ili odgovoru:")
        filtered = [e for e in logs if search.lower() in e["question"].lower() or search.lower() in e["answer"].lower()] if search else logs
        st.write(f"Prikazano {len(filtered)} od {len(logs)} logova.")
        for entry in reversed(filtered):
            with st.expander(f"{entry['datetime']} | Pitanje: {entry['question'][:50]}..."):
                st.write("**Pitanje:**", entry["question"])
                st.write("**Kontekst:**", entry["retrieved_context"])
                st.write("**Odgovor:**", entry["answer"])
    else:
        st.info("Nema logiranih sesija još.")