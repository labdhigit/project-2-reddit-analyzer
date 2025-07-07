# Reddit Sentiment Analyzer with GenAI Summarization

Analyze the sentiment of posts from any subreddit and generate a summary using Hugging Face Transformers.

## Features
- Fetch top Reddit posts using PRAW
- Classify post sentiment using VADER
- Generate visual pie chart using Plotly
- Summarize titles using transformers
- Download summary as PDF using ReportLab

## How to Run

1. Clone the repo  
2. Add `reddit_client_id` and `reddit_client_secret` in Streamlit Secrets  
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
