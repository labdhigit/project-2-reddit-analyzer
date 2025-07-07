import praw

reddit = praw.Reddit(
    client_id = "DGmiAFFAAk7gMiCV-d2l3Q",
    client_secret = "95yf3LugFCbZR_g9Uy0mth6qWXlIvg",
    user_agent = "SentimentTracker by u/Responsible-Solid113"
)

subreddit_name = input("Enter the name of a subreddit (e.g. tech): ")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

subreddit = reddit.subreddit(subreddit_name)
top_posts = list(subreddit.hot(limit=10))
for post in top_posts:
    print(f"Post Title: {post.title}")
    print(f"Post Upvotes: {post.score}")
    print(f"Comments: {post.num_comments}")
    print("-------------------")


positive_count = 0
negative_count = 0
neutral_count = 0

for post in top_posts:
  title = post.title
  sentiment = analyzer.polarity_scores(title)

  def classify_sentiment(score):
    if score >=0.05:
      return "Positive"
    elif score <= -0.05:
      return "Negative"
    else:
      return "Neutral"

  sentiment_type = classify_sentiment(sentiment["compound"])

for post in top_posts:
  title = post.title
  sentiment = analyzer.polarity_scores(title)
  sentiment_type = classify_sentiment(sentiment["compound"])

  if sentiment_type == "Positive":
      positive_count += 1
  elif sentiment_type == "Negative":
      negative_count += 1
  else:
      neutral_count += 1

  print(f"Title : {title}")
  print(f"Sentiment : {sentiment_type}")
  print("----------------------")

import plotly.graph_objects as go

labels = ["Postive","Neutral","Negative"]
values = [positive_count,neutral_count,negative_count]

fig = go.Figure(data=[go.Pie(labels=labels,values=values, hole=0.03)])
fig.update_layout(title_text="Sentiment Analyzer Distribution")
fig.show()

pip install transformers

from transformers import pipeline

summarizer = pipeline("summarization",model ="facebook/bart-large-cnn")
top_post = top_posts[0]
top_post.comments.replace_more(limit=0)
top_comments = [comment.body for comment in top_post.comments[:10]]
comment_text = "\n".join(top_comments)

summary_output = summarizer(comment_text, max_length=100, min_length=30, do_sample=False)
summary = summary_output[0]['summary_text']

print("\nSummary of Reddit Comments:")
print(summary)

pip install reportlab -U kaleido==0.2.1

import plotly.io as pio
pio.write_image(fig, "sentiment_chart.png", format="png", width=600, height=400)

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from textwrap import wrap
import os

pdf_filename = f"Reddit_Report_{subreddit_name}.pdf"
c = canvas.Canvas(pdf_filename, pagesize=A4)
width, height = A4

c.setFont("Helvetica-Bold", 18)
c.drawString(50, height - 50, f"Reddit Sentiment Report: r/{subreddit_name}")

c.setFont("Helvetica", 12)
c.drawString(50, height - 90, f"Positive Posts: {positive_count}")
c.drawString(50, height - 110, f"Neutral Posts: {neutral_count}")
c.drawString(50, height - 130, f"Negative Posts: {negative_count}")

image_path = "sentiment_chart.png"
if os.path.exists(image_path):
    c.drawImage(image_path, 50, height - 400, width=400, height=250)
else:
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, height - 400, "⚠️ Chart image not found!")

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
print(f"PDF report saved as: {pdf_filename}")
