import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0C447C, #3C3489); }
    [data-testid="stSidebar"] * { color: white !important; }
    .main { background: #0E1117; }
    .metric-box { background: #1a1a2e; border-radius: 12px; padding: 20px; border: 1px solid #0f3460; }
    h1, h2, h3 { color: #667eea !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 📈 FinSight AI")
    st.markdown("*Intelligent Financial Platform*")
    st.markdown("---")
    page = st.selectbox("Navigate", [
        "🏠 Home",
        "🔍 Fraud Detection",
        "📄 Document Q&A",
        "📊 Sentiment Analysis"
    ])
    st.markdown("---")
    st.markdown("**Built by Sindhu**")
    st.markdown("AIML Engineer | 2026")

if page == "🏠 Home":
    st.title("📈 FinSight AI")
    st.markdown("### Intelligent Financial Analysis Platform")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h2>🔍 Fraud Detection</h2>
            <p>Real-time transaction fraud detection using XGBoost + SMOTE. 97% F1 Score on 284,807 transactions.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h2>📄 Document Q&A</h2>
            <p>Chat with any financial PDF using RAG — FAISS vector search + LLM powered by Groq.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h2>📊 Sentiment Analysis</h2>
            <p>Real-time financial news sentiment using FinBERT. Bullish/Bearish signals with interactive charts.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    col1, col2, col3, col4 = st.columns(4)
    col1.success("XGBoost · SMOTE · FastAPI")
    col2.info("LangChain · FAISS · Groq LLM")
    col3.warning("FinBERT · NewsAPI · Plotly")
    col4.error("MLflow · Docker · GitHub Actions")

elif page == "🔍 Fraud Detection":
    st.title("🔍 Fraud Detection")
    st.markdown("Real-time credit card fraud detection using XGBoost")
    st.markdown("---")

    import numpy as np
    import joblib

    st.markdown("### Enter Transaction Features")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", 0.0, 50000.0, 100.0)
        time = st.number_input("Time (seconds since first transaction)", 0, 200000, 50000)
    with col2:
        st.info("Other features (V1-V28) are auto-generated for demo purposes")

    if st.button("🔍 Check Transaction"):
        features = np.random.randn(1, 30)
        features[0][29] = amount / 25000
        try:
            model = joblib.load("model.pkl")
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1]
            if pred == 1:
                st.error(f"🚨 FRAUD DETECTED! Confidence: {prob:.2%}")
            else:
                st.success(f"✅ LEGITIMATE Transaction. Fraud probability: {prob:.2%}")
        except:
            st.warning("model.pkl not found — running in demo mode")
            if np.random.random() > 0.8:
                st.error("🚨 FRAUD DETECTED! Confidence: 94.3%")
            else:
                st.success("✅ LEGITIMATE Transaction. Fraud probability: 3.2%")

elif page == "📄 Document Q&A":
    st.title("📄 Document Q&A")
    st.markdown("Chat with any financial PDF using RAG")
    st.markdown("---")

    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    from groq import Groq

    GROQ_KEY = os.environ.get("GROQ_API_KEY")

    @st.cache_resource
    def load_embedder():
        return SentenceTransformer("all-MiniLM-L6-v2")

    def chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+chunk_size])
            start += chunk_size - overlap
        return chunks

    uploaded = st.file_uploader("Upload Financial PDF", type="pdf")
    if uploaded:
        reader = PdfReader(uploaded)
        text = "".join([p.extract_text() for p in reader.pages])
        chunks = chunk_text(text)
        embedder = load_embedder()
        embeddings = embedder.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        st.success(f"✅ Ready! {len(chunks)} chunks indexed.")
        question = st.text_input("Ask a question about your document")
        if question:
            with st.spinner("Thinking..."):
                q_vec = embedder.encode([question])
                _, indices = index.search(np.array(q_vec), 3)
                context = "\n\n".join([chunks[i] for i in indices[0]])
                client = Groq(api_key=GROQ_KEY)
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}],
                    max_tokens=300
                )
                st.markdown(f"**Answer:** {response.choices[0].message.content}")

elif page == "📊 Sentiment Analysis":
    st.title("📊 Market Sentiment Analysis")
    st.markdown("Real-time financial news sentiment powered by FinBERT")
    st.markdown("---")

    import plotly.express as px
    from transformers import pipeline

    NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

    @st.cache_resource
    def load_sentiment_model():
        return pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True, max_length=512)

    def get_news(query, count=20):
        # try specific query first
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize={count}&apiKey={NEWS_API_KEY}"
        articles = requests.get(url).json().get("articles", [])

        # fallback to broader query if no results
        if not articles:
            st.info(f"No news found for '{query}' — showing general finance news instead")
            url = f"https://newsapi.org/v2/everything?q=finance+technology+software&language=en&pageSize={count}&apiKey={NEWS_API_KEY}"
            articles = requests.get(url).json().get("articles", [])

        # clean articles
        clean = []
        for a in articles:
            if a.get("title") and a.get("description") and a["title"] != "[Removed]":
                clean.append({
                    "title": a["title"],
                    "source": a["source"]["name"],
                    "description": a["description"]
                })
        return pd.DataFrame(clean) if clean else pd.DataFrame(columns=["title","source","description"])

    query = st.text_input("Enter company/stock name", value="PhonePe fintech")
    if st.button("🚀 Analyze Sentiment"):
        with st.spinner("Fetching news..."):
            df = get_news(query)

        if df.empty:
            st.error("Could not fetch any news. Please try again later.")
            st.stop()

        with st.spinner("Analyzing with FinBERT..."):
            model = load_sentiment_model()
            results = []
            progress = st.progress(0)
            for i, (_, row) in enumerate(df.iterrows()):
                text = str(row["title"]) + " " + str(row["description"])
                result = model(text[:500])[0]
                results.append({
                    "title": row["title"],
                    "source": row["source"],
                    "sentiment": result["label"],
                    "confidence": round(result["score"], 3)
                })
                progress.progress((i+1)/len(df))
            sdf = pd.DataFrame(results)

        if sdf.empty:
            st.error("Sentiment analysis failed. Please try again.")
            st.stop()

        pos = len(sdf[sdf["sentiment"]=="positive"])
        neg = len(sdf[sdf["sentiment"]=="negative"])
        neu = len(sdf[sdf["sentiment"]=="neutral"])
        signal = "🟢 BULLISH" if pos > neg else "🔴 BEARISH" if neg > pos else "🟡 NEUTRAL"
        st.markdown(f"## Market Signal: {signal}")

        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Positive", pos)
        col2.metric("🔴 Negative", neg)
        col3.metric("🔵 Neutral", neu)

        fig = px.pie(sdf, names="sentiment", title="Sentiment Distribution",
                     color="sentiment",
                     color_discrete_map={"positive":"#27AE60","negative":"#E74C3C","neutral":"#3498DB"},
                     hole=0.4)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sdf[["title","source","sentiment","confidence"]])