import streamlit as st
import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from transformers import pipeline
from reportlab.pdfgen import canvas
from datetime import datetime
import tempfile

st.set_page_config(page_title="Reddit Sentiment Analyzer", layout="wide")

client_id = st.secrets["reddit_client_id"]
client_secret = st.secrets["reddit_client_secret"]
user_agent = "Reddit Sentiment Analyzer"

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
analyzer = SentimentIntensityAnalyzer()
summarizer = pipeline("summarization")

def classify_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

st.title("Reddit Sentiment Tracker")

subreddit_name = st.text_input("Enter Subreddit Name", "technology")
post_limit = st.slider("Number of Posts to Analyze", 5, 50, 10)

if st.button("Analyze"):
    posts = []
    subreddit = reddit.subreddit(subreddit_name)

    for post in subreddit.hot(limit=post_limit):
        sentiment = classify_sentiment(post.title)
        posts.append({"Title": post.title, "Sentiment": sentiment})

    df = pd.DataFrame(posts)
    st.dataframe(df)

    sentiment_count = df["Sentiment"].value_counts().reset_index()
    sentiment_count.columns = ["Sentiment", "Count"]
    fig = px.pie(sentiment_count, names="Sentiment", values="Count", title="Sentiment Distribution")
    st.plotly_chart(fig)

    full_text = " ".join(df["Title"].tolist())
    if len(full_text) > 100:
        summary = summarizer(full_text[:1000], max_length=60, min_length=30, do_sample=False)[0]["summary_text"]
        st.subheader("Summary")
        st.write(summary)

        if st.button("Download Summary as PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                c = canvas.Canvas(tmp_file.name)
                c.drawString(100, 800, "Reddit Sentiment Summary")
                c.drawString(100, 780, f"Subreddit: {subreddit_name}")
                c.drawString(100, 760, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
                text_object = c.beginText(100, 730)
                for line in summary.split(". "):
                    text_object.textLine(line.strip())
                c.drawText(text_object)
                c.save()
                with open(tmp_file.name, "rb") as f:
                    st.download_button("Download PDF", f, file_name="reddit_summary.pdf")
