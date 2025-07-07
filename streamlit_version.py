import streamlit as st
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from textwrap import wrap
import os
import plotly.io as pio
import kaleido

reddit = praw.Reddit(
    client_id = "DGmiAFFAAk7gMiCV-d2l3Q",
    client_secret = "95yf3LugFCbZR_g9Uy0mth6qWXlIvg",
    user_agent = "SentimentTracker by u/Responsible-Solid113"
)

st.title("Reddit Sentiment Tracker with GenAI")
subreddit_name = st.text_input("Enter Subreddit Name (e.g. tech, psychology)", value="technology")

if st.button("Analyze"):
    st.info(f"Fetching top posts from r/{subreddit_name}...")

    subreddit = reddit.subreddit(subreddit_name)
    top_posts = list(subreddit.hot(limit=10))

    analyzer = SentimentIntensityAnalyzer()

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    classified_posts = []

    for post in top_posts:
        title = post.title
        sentiment = analyzer.polarity_scores(title)
        compound = sentiment["compound"]
        if compound >= 0.05:
            sentiment_type = "Positive"
            positive_count += 1
        elif compound <= -0.05:
            sentiment_type = "Negative"
            negative_count += 1
        else:
            sentiment_type = "Neutral"
            neutral_count += 1
        classified_posts.append((title, sentiment_type))

    st.subheader("Sentiment Analysis on Titles")
    for title, sentiment_type in classified_posts:
        st.write(f"**{sentiment_type}** - {title}")

    labels = ["Positive", "Neutral", "Negative"]
    values = [positive_count, neutral_count, negative_count]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.03)])
    fig.update_layout(title_text="Sentiment Distribution")
    st.plotly_chart(fig)

    pio.write_image(fig, "sentiment_chart.png", format="png", width=600, height=400)

    st.subheader("GenAI Summary of Top Comments")

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    top_post = top_posts[0]
    top_post.comments.replace_more(limit=0)
    top_comments = [comment.body for comment in top_post.comments[:10]]
    comment_text = "\n".join(top_comments)

    summary_output = summarizer(comment_text, max_length=100, min_length=30, do_sample=False)
    summary = summary_output[0]['summary_text']
    st.success(summary)

    pdf_filename = f"Reddit_Report_{subreddit_name}.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, f"Reddit Sentiment Report: r/{subreddit_name}")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Positive Posts: {positive_count}")
    c.drawString(50, height - 110, f"Neutral Posts: {neutral_count}")
    c.drawString(50, height - 130, f"Negative Posts: {negative_count}")

    if os.path.exists("sentiment_chart.png"):
        c.drawImage("sentiment_chart.png", 50, height - 400, width=400, height=250)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 420, "Summary of Top Comments:")
    c.setFont("Helvetica", 11)
    wrapped_summary = wrap(summary, width=100)
    y = height - 440
    for line in wrapped_summary:
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)
        c.drawString(50, y, line)
        y -= 15

    c.save()
    st.success(f" PDF Report saved as: `{pdf_filename}`")

    with open(pdf_filename, "rb") as file:
        st.download_button(label=" Download PDF Report", data=file, file_name=pdf_filename, mime="application/pdf")
