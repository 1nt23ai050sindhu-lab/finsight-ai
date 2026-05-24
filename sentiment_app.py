import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline

NEWS_API_KEY = "ca666ef5b94a4764868663054ad088e0"

# page config
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom CSS
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .positive { border-color: #27AE60; }
    .negative { border-color: #E74C3C; }
    .neutral { border-color: #3498DB; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        font-size: 16px;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; }
    h1 { color: #667eea !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True, max_length=512)

def get_news(query, count=20):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize={count}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return pd.DataFrame([{
        "title": a["title"],
        "description": a["description"],
        "published": a["publishedAt"],
        "source": a["source"]["name"]
    } for a in articles if a["title"] and a["description"]])

# sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-market.png", width=80)
    st.title("FinSight AI")
    st.markdown("---")
    st.markdown("### 🔍 Search Settings")
    query = st.text_input("Company / Stock", value="PhonePe fintech")
    count = st.slider("Number of articles", 5, 30, 20)
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("Powered by FinBERT — a BERT model fine-tuned on financial text for accurate sentiment analysis.")
    analyze_btn = st.button("🚀 Analyze Now")

# main area
st.title("📈 FinSight AI — Market Sentiment Dashboard")
st.markdown("*Real-time financial news sentiment analysis powered by FinBERT*")
st.markdown("---")

if analyze_btn:
    with st.spinner("📰 Fetching latest news..."):
        df = get_news(query, count)

    if df.empty:
        st.error("No articles found. Try a different search term.")
    else:
        with st.spinner("🤖 Analyzing sentiment with FinBERT..."):
            model = load_model()
            results = []
            progress = st.progress(0)
            for i, (_, row) in enumerate(df.iterrows()):
                text = str(row["title"]) + " " + str(row["description"])
                result = model(text[:500])[0]
                results.append({
                    "title": row["title"],
                    "source": row["source"],
                    "published": row["published"],
                    "sentiment": result["label"],
                    "confidence": round(result["score"], 3)
                })
                progress.progress((i+1)/len(df))
            sentiment_df = pd.DataFrame(results)

        # metrics row
        pos = len(sentiment_df[sentiment_df["sentiment"]=="positive"])
        neg = len(sentiment_df[sentiment_df["sentiment"]=="negative"])
        neu = len(sentiment_df[sentiment_df["sentiment"]=="neutral"])
        total = len(sentiment_df)
        overall = "🟢 BULLISH" if pos > neg else "🔴 BEARISH" if neg > pos else "🟡 NEUTRAL"

        st.markdown(f"### Market Signal: **{overall}**")
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📰 Total Articles", total)
        col2.metric("🟢 Positive", pos, f"{round(pos/total*100)}%")
        col3.metric("🔴 Negative", neg, f"{round(neg/total*100)}%")
        col4.metric("🔵 Neutral", neu, f"{round(neu/total*100)}%")

        st.markdown("---")

        # charts row
        col_left, col_right = st.columns(2)

        with col_left:
            fig1 = px.pie(
                sentiment_df, names="sentiment",
                title="Sentiment Distribution",
                color="sentiment",
                color_discrete_map={"positive":"#27AE60","negative":"#E74C3C","neutral":"#3498DB"},
                hole=0.4
            )
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_right:
            source_counts = sentiment_df["source"].value_counts().head(8).reset_index()
            source_counts.columns = ["source", "count"]
            fig2 = px.bar(
                source_counts, x="count", y="source",
                orientation="h", title="Top News Sources",
                color="count", color_continuous_scale="Viridis"
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # confidence chart
        fig3 = px.bar(
            sentiment_df.head(15), x="confidence", y="title",
            orientation="h", color="sentiment",
            title="Sentiment Confidence per Article",
            color_discrete_map={"positive":"#27AE60","negative":"#E74C3C","neutral":"#3498DB"}
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=500
        )
        st.plotly_chart(fig3, use_container_width=True)

        # news table
        st.markdown("### 📰 Latest News")
        for _, row in sentiment_df.iterrows():
            color = "#27AE60" if row["sentiment"]=="positive" else "#E74C3C" if row["sentiment"]=="negative" else "#3498DB"
            emoji = "🟢" if row["sentiment"]=="positive" else "🔴" if row["sentiment"]=="negative" else "🔵"
            st.markdown(f"""
            <div style="border-left: 3px solid {color}; padding: 8px 12px; margin: 6px 0; background: #1a1a2e; border-radius: 4px;">
                <b>{emoji} {row['title']}</b><br>
                <small style="color:#888">{row['source']} | Confidence: {row['confidence']}</small>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px; color: #666;">
        <h2>👈 Enter a company name and click Analyze</h2>
        <p>Try: PhonePe, JPMorgan, Tesla, Bitcoin, Infosys</p>
    </div>
    """, unsafe_allow_html=True)