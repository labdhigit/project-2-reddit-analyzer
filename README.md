**Reddit Sentiment Tracker with GenAI Summarization & PDF Report**

A Python-based application that performs sentiment analysis on the top Reddit posts from any subreddit, generates a visual sentiment distribution chart, summarizes top comments using a pre-trained GenAI model, and compiles a professional PDF report of the analysis.

---


**Features**

- Fetches top 10 posts from any user-specified subreddit using Reddit API (`praw`)
- Analyzes sentiment (Positive, Neutral, Negative) of post titles using VADER
- Visualizes sentiment distribution using interactive Plotly pie chart
- Uses GenAI (`facebook/bart-large-cnn`) to summarize top comments
- Exports a clean PDF report with sentiment stats, pie chart, and comment summary
- Optional Streamlit UI for user input and simplified interaction

---

**Tech Stack**

| Tool/Library      | Purpose                                 |
|------------------|-----------------------------------------|
| `praw`            | Reddit API wrapper                      |
| `vaderSentiment`  | Rule-based sentiment analysis           |
| `plotly`          | Interactive charting                    |
| `transformers`    | GenAI summarization (`BART`)            |
| `reportlab`       | PDF generation                          |
| `streamlit`       | Optional UI (can be toggled)            |
| `kaleido`         | Static image export for charts          |

---

**Installation**

pip install praw vaderSentiment plotly pdfkit transformers reportlab kaleido==0.2.1

---
