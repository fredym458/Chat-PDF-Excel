import streamlit as st
import pandas as pd
import PyPDF2
import numpy as np
from openai import OpenAI

# --- CONFIGURATION ---
st.set_page_config(page_title="Data Chat Pro", page_icon="📊", layout="wide")
st.title("📊 Chat with PDF & Excel")

# SECURE KEY HANDLING

if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

else:
    st.error("⚠️ OpenAI API Key missing in .streamlit/secrets.toml")
    st.stop()

# --- HELPER FUNCTIONS ---

def get_embedding(text):
    """Generates vector embedding for a text chunk."""
    try:
        # Clean newlines to reduce noise
        text = text.replace("\n", " ")
        return client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
    except Exception as e:
        print(f"Error embedding text: {e}")
        return []

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_pdf(file):
    """Extracts text from PDF."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def process_excel(file):
    """Converts Excel rows into descriptive text sentences."""
    df = pd.read_excel(file)
    # Handle empty cells
    df = df.fillna("")
    
    text_data = []
    # We turn each row into a context string: "Column A is Value, Column B is Value..."
    # This preserves the relationship between cells.
    for index, row in df.iterrows():
        row_string = ", ".join([f"{col}: {val}" for col, val in row.items()])
        text_data.append(row_string)
    
    return "\n".join(text_data)

def smart_chunking(text, chunk_size=1000, overlap=200):
    """
    Industry Standard: Sliding Window Chunking.
    We don't just cut text; we overlap it so context isn't lost at the cut.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward, but step back by the overlap amount
        start += chunk_size - overlap
    
    return chunks

# --- SIDEBAR: FILE UPLOAD ---
with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload PDF or Excel", type=["pdf", "xlsx", "xls"])
    
    if uploaded_file and st.button("Process Data"):
        with st.spinner("Parsing & Indexing..."):
            raw_text = ""
            
            # 1. Detect File Type & Extract Text
            if uploaded_file.name.endswith('.pdf'):
                raw_text = process_pdf(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                raw_text = process_excel(uploaded_file)
            
            # 2. Smart Chunking
            text_chunks = smart_chunking(raw_text)
            
            # 3. Generate Embeddings
            knowledge_base = []
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(text_chunks):
                vector = get_embedding(chunk)
                if len(vector) > 0:
                    knowledge_base.append({"text": chunk, "vector": vector})
                # Update progress bar
                progress_bar.progress((i + 1) / len(text_chunks))
            
            st.session_state["vector_db"] = knowledge_base
            st.success(f"Successfully indexed {len(knowledge_base)} chunks!")

# --- CHAT INTERFACE ---

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful analyst. Answer based ONLY on the provided documents."}
    ]

# Display Chat History
for msg in st.session_state["messages"]:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User Input
if user_input := st.chat_input("Ask a question about your document..."):
    
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # 2. Retrieval (RAG)
    context_text = ""
    if "vector_db" in st.session_state:
        query_vec = get_embedding(user_input)
        
        # Calculate scores for ALL chunks
        scores = []
        for item in st.session_state["vector_db"]:
            score = cosine_similarity(query_vec, item["vector"])
            scores.append((score, item["text"]))
        
        # Sort by score (High to Low) and take Top 3
        # Taking top 3 ensures we get more context than just one lucky guess
        scores.sort(key=lambda x: x[0], reverse=True)
        top_matches = scores[:3]
        
        context_text = "\n\n---\n".join([match[1] for match in top_matches])
        
        # Debug option in sidebar to see what was retrieved
        with st.sidebar:
            with st.expander("🔍 Debug: Retrieval"):
                st.write(f"Top Score: {top_matches[0][0]:.4f}")
                st.text(top_matches[0][1])

    # 3. Generate Response
    full_prompt = f"Context from documents:\n{context_text}\n\nUser Question: {user_input}"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful analyst. Use the provided context to answer. If the answer is not in the context, say 'I cannot find that in the document'."},
            {"role": "user", "content": full_prompt}
        ]
    )
    
    ai_reply = response.choices[0].message.content
    
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
    st.session_state["messages"].append({"role": "assistant", "content": ai_reply})