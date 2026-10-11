"""Streamlit user interface. Run with: streamlit run streamlit_app.py"""

import pandas as pd
import streamlit as st

from news_model import WEIGHTS, NewsEncoder, rank_articles, score_article


st.set_page_config(page_title="News Popularity Intelligence", page_icon="📰", layout="wide")


@st.cache_resource(show_spinner="Loading pretrained DistilBERT…")
def get_encoder():
    return NewsEncoder()


page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "News Intelligence", "Model Reasoning"],
)

if page == "Home":
    st.title("📰 News Popularity Intelligence")
    st.subheader("A simple transformer-based news analysis demo")
    st.warning(
        "There are no click, share, or impression labels. This app shows an "
        "unvalidated attention-potential proxy, not actual popularity."
    )
    st.markdown(
        """
        ### What the system does

        1. Reads a news title and description.
        2. Uses pretrained DistilBERT to create semantic text vectors.
        3. Combines transformer signals with simple, visible text signals.
        4. Shows a 0–100 score and explanations.
        5. Ranks articles relative to a user-uploaded CSV batch.

        ### Why the score is only a proxy

        Actual popularity also depends on timing, audience, placement, and
        distribution. Those things are not present in the unlabeled text file,
        so the app does not invent popularity labels or train on them.
        """
    )

elif page == "News Intelligence":
    st.title("🔎 News Intelligence")
    st.caption("Scores are content-based proxies, not measured popularity.")
    single_tab, batch_tab = st.tabs(["Score one article", "Compare a CSV batch"])

    with single_tab:
        with st.form("article_form"):
            title = st.text_input("News title")
            description = st.text_area("News description", height=140)
            submit = st.form_submit_button("Analyze article", type="primary")

        if submit:
            if not title.strip() or not description.strip():
                st.error("Enter both a title and a description.")
            else:
                try:
                    with st.spinner("Analyzing the article…"):
                        result = score_article(title.strip(), description.strip(), get_encoder())
                    st.metric("Attention-potential proxy", f"{result['score']}/100")
                    st.progress(int(result["score"]))
                    st.subheader("Why this score?")
                    for highlight in result["highlights"]:
                        st.write("• " + highlight)
                    labels = {
                        "urgency": "Urgency wording",
                        "emotion": "Emotional language",
                        "title_length": "Title length",
                        "clarity": "Word clarity proxy",
                        "word_variety": "Lexical variety",
                        "text_length": "Text length fit",
                        "transformer_match": "Transformer title-description match",
                        "batch_novelty": "Batch novelty (neutral for one article)",
                    }
                    signals = pd.DataFrame([
                        {"Signal": labels[name], "Value": value, "Weight": WEIGHTS[name]}
                        for name, value in result["signals"].items()
                    ])
                    st.dataframe(signals, hide_index=True, use_container_width=True)
                except Exception as error:
                    st.error(f"Model could not run: {error}")
                    st.info("The first use downloads DistilBERT; check internet access and try again.")

    with batch_tab:
        st.write("Upload a CSV with `title` and `description` columns. Only those columns are used.")
        upload = st.file_uploader("Choose a CSV file", type=["csv"])
        if upload is not None:
            try:
                articles = pd.read_csv(upload)
                if len(articles) > 100:
                    st.warning("For CPU-friendly use, only the first 100 rows will be scored.")
                    articles = articles.head(100)
                if st.button("Rank articles", type="primary"):
                    with st.spinner("Creating transformer embeddings and ranking…"):
                        ranked = rank_articles(articles, get_encoder())
                    st.dataframe(
                        ranked[["rank", "score", "title", "highlights"]],
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download ranked CSV",
                        data=ranked.to_csv(index=False).encode("utf-8"),
                        file_name="news_proxy_scores.csv",
                        mime="text/csv",
                    )
                    st.caption("Novelty is relative only to this uploaded batch.")
            except Exception as error:
                st.error(f"Could not read or rank this CSV: {error}")

else:
    st.title("🧠 Model Reasoning")
    st.warning("The score is not calibrated to a probability and has not been validated against popularity labels.")
    st.write("The pretrained transformer measures semantic alignment between the title and description. Other values are simple text proxies.")
    explanations = {
        "Urgency wording": "Counts terms such as breaking, urgent, alert, and warning.",
        "Emotional language": "Counts a short list of high-intensity words.",
        "Title length": "Prefers a title near ten words, a design assumption.",
        "Word clarity proxy": "Uses average word length; it is not a grammar checker.",
        "Lexical variety": "Unique words divided by all words.",
        "Text length fit": "Prefers about 45 words across title and description.",
        "Transformer title-description match": "Cosine similarity from DistilBERT embeddings.",
        "Batch novelty": "Relative novelty compared with other rows in the same CSV.",
    }
    table = pd.DataFrame([
        {"Signal": label, "Weight": WEIGHTS[key], "How it is estimated": explanations[label]}
        for key, label in [
            ("urgency", "Urgency wording"),
            ("emotion", "Emotional language"),
            ("title_length", "Title length"),
            ("clarity", "Word clarity proxy"),
            ("word_variety", "Lexical variety"),
            ("text_length", "Text length fit"),
            ("transformer_match", "Transformer title-description match"),
            ("batch_novelty", "Batch novelty"),
        ]
    ])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.markdown(
        "The final score is a weighted sum of these 0–1 signals, multiplied by 100. "
        "No ground-truth clicks or shares are available, so real-world accuracy cannot be measured."
    )
