import streamlit as st
import pandas as pd
import PyPDF2
import numpy as np
import json
from openai import OpenAI
from duckduckgo_search import DDGS

# --- CONFIGURATION ---
st.set_page_config(page_title="Data Chat Pro", page_icon="📊", layout="wide")
st.title("📊 Chat with PDF & Excel")

# --- SECURE KEY HANDLING ---
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("⚠️ OpenAI API Key missing! Please add it to .streamlit/secrets.toml")
    st.stop()

# --- HELPER FUNCTIONS ---

def get_embedding(text):
    """Generates vector embedding for a text chunk."""
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding
    except Exception as e:
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

def process_data(file):
    """Loads CSV or Excel, cleans headers, saves to Session State, and returns text."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
            
        # --- FIX: SANITIZE COLUMN NAMES (Removes colons to fix Charting Error) ---
        df.columns = [str(col).replace(":", " -") for col in df.columns]
        # ------------------------------------------------------------------------
        
        df = df.fillna("")
        
        # Save for Analyst Tool & Plotter
        st.session_state["dataframe"] = df

        # Generate Text Description for RAG (Limit rows for token safety)
        text_data = []
        for index, row in df.head(100).iterrows():
            row_string = ", ".join([f"{col}: {val}" for col, val in row.items()])
            text_data.append(row_string)
        
        return "\n".join(text_data)
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return ""

def smart_chunking(text, chunk_size=1000, overlap=200):
    """Sliding Window Chunking."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# --- TOOL DEFINITIONS ---

def web_search(query):
    """Searches the internet."""
    try:
        results = DDGS().text(query, max_results=3)
        return "\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Error searching web: {e}"

def python_analytics(code):
    """Executes Python code on the dataframe."""
    if "dataframe" not in st.session_state:
        return "Error: No Excel/CSV file uploaded."
    
    df = st.session_state["dataframe"]
    # Pass 'local_env' as globals so the function sees 'df'
    local_env = {"pd": pd, "np": np, "df": df}
    
    try:
        if any(bad in code for bad in ["import", "os", "sys", "open", "exec", "eval"]):
            return "Error: Unsafe code detected."
            
        wrapped_code = f"def solver():\n    return {code}"
        exec(wrapped_code, local_env) 
        return str(local_env["solver"]())
    except Exception as e:
        return f"Error executing code: {e}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for facts or current events.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python_analytics",
            "description": "Run Python pandas code. 'df' is the dataframe. Useful for aggregation, sorting, or filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The python line to run, e.g. df['Salary'].mean()"}
                },
                "required": ["code"]
            }
        }
    }
]

# --- SIDEBAR: FILE UPLOAD & VISUALIZATION ---
with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload PDF, Excel, or CSV", type=["pdf", "xlsx", "xls", "csv"])
    
    if uploaded_file and st.button("Process Data"):
        with st.spinner("Parsing & Indexing..."):
            raw_text = ""
            
            # 1. Detect File Type
            if uploaded_file.name.endswith('.pdf'):
                raw_text = process_pdf(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls', '.csv')):
                raw_text = process_data(uploaded_file)
            
            # 2. Chunking & Embedding
            text_chunks = smart_chunking(raw_text)
            knowledge_base = []
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(text_chunks):
                vector = get_embedding(chunk)
                if len(vector) > 0:
                    knowledge_base.append({"text": chunk, "vector": vector})
                progress_bar.progress((i + 1) / len(text_chunks))
            
            st.session_state["vector_db"] = knowledge_base
            st.success(f"Indexed {len(knowledge_base)} chunks!")

    # --- INTERACTIVE VISUALIZER ---
    if "dataframe" in st.session_state:
        st.divider()
        st.header("📊 Visualizer")
        df = st.session_state["dataframe"]
        
        # 1. Get Columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 2. User Controls
        if numeric_cols:
            chart_type = st.radio("Chart Type", ["Bar", "Line", "Area"], horizontal=True)
            
            # Smart defaults
            default_x = text_cols[0] if text_cols else numeric_cols[0]
            default_y = numeric_cols[0]
            
            x_axis = st.selectbox("X-Axis (Category)", df.columns, index=df.columns.get_loc(default_x))
            y_axis = st.selectbox("Y-Axis (Value)", numeric_cols, index=numeric_cols.index(default_y))
            
            # 3. Plot
            st.caption(f"Plotting **{y_axis}** by **{x_axis}**")
            chart_data = df.head(50).set_index(x_axis)[y_axis]
            
            if chart_type == "Bar":
                st.bar_chart(chart_data)
            elif chart_type == "Line":
                st.line_chart(chart_data)
            elif chart_type == "Area":
                st.area_chart(chart_data)
        else:
            st.warning("No numeric data found to plot.")

# --- CHAT INTERFACE ---

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful analyst. Answer based ONLY on the provided documents."}
    ]

# Display Chat History
for msg in st.session_state["messages"]:
    if msg["role"] != "system" and msg["role"] != "tool":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User Input
if user_input := st.chat_input("Ask a question about your document..."):
    
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # 2. Retrieval (RAG)
    context_text = ""
    references = [] 
    
    if "vector_db" in st.session_state:
        query_vec = get_embedding(user_input)
        scores = []
        for item in st.session_state["vector_db"]:
            score = cosine_similarity(query_vec, item["vector"])
            scores.append((score, item["text"]))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        top_matches = scores[:3]
        
        for i, match in enumerate(top_matches):
            score, text = match
            context_text += text + "\n\n"
            references.append(f"**Match {i+1} (Score: {score:.2f}):**\n{text[:150]}...")

    # 3. Dynamic System Prompt
    math_keywords = ["average", "mean", "sum", "count", "top", "bottom", "rank", "sort", "chart", "graph"]
    
    if any(word in user_input.lower() for word in math_keywords):
        if "dataframe" in st.session_state:
            columns = st.session_state["dataframe"].columns.tolist()
            col_info = f"Available Columns: {columns}"
        else:
            col_info = "No dataframe loaded."

        system_instruction = f"""
        You are a Python Data Analyst. 
        DATA SCHEMA: {col_info}
        RULES:
        1. You MUST use the 'python_analytics' tool for calculations/sorting.
        2. DO NOT estimate from text.
        3. 'Rank' column is text. Sort by 'Total' or numeric columns.
        """
        full_prompt = f"User Question: {user_input}" 
    else:
        system_instruction = "You are a helpful analyst. Use the provided text context to answer."
        full_prompt = f"Context from uploaded documents:\n{context_text}\n\nUser Question: {user_input}"

    # First Call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": full_prompt}
        ],
        tools=tools,
        tool_choice="auto"
    )

    response_msg = response.choices[0].message
    tool_calls = response_msg.tool_calls

    if tool_calls:
        with st.chat_message("assistant"):
            st.write("⚙️ *AI is using tools...*")
        
        tool_outputs = []
        for tool_call in tool_calls:
            if tool_call.function.name == "web_search":
                args = json.loads(tool_call.function.arguments)
                result = web_search(args["query"])
                tool_outputs.append({
                    "role": "tool", "tool_call_id": tool_call.id,
                    "name": "web_search", "content": result
                })
                with st.sidebar:
                    with st.expander("🌍 Web Search Result"):
                        st.write(result)
            
            elif tool_call.function.name == "python_analytics":
                args = json.loads(tool_call.function.arguments)
                code = args["code"]
                
                with st.chat_message("assistant"):
                    st.code(code, language="python") 
                
                result = python_analytics(code)
                tool_outputs.append({
                    "role": "tool", "tool_call_id": tool_call.id,
                    "name": "python_analytics", "content": result
                })
                with st.sidebar:
                    with st.expander("🐍 Python Result"):
                        st.write(result)

        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": full_prompt},
                response_msg
            ] + tool_outputs
        )
        ai_reply = final_response.choices[0].message.content

    else:
        ai_reply = response_msg.content

    # Display Final Answer
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
        
        if references and not tool_calls:
            with st.expander("📚 View Source Documents"):
                for ref in references:
                    st.markdown(ref)
                    st.divider()

    st.session_state["messages"].append({"role": "assistant", "content": ai_reply})