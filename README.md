🦅 Garuda Rakshak - News & Twitter Emotion Insight

An intelligent, integrated platform built with *Streamlit* that analyzes **emotions in city-wise news headlines** and **Twitter activity** using the power of **transformer-based NLP models** and **web scraping techniques**.

🔍 Overview

This tool leverages:
- 📰 Real-time **city-specific news** from trusted sources like *Times of India* and *Times Now*
- 🐦 **Tweets** from specific users, hashtags, or keywords using the **Nitter API**
- 🤖 **HuggingFace Transformers** emotion classification model: `bhadresh-savani/distilbert-base-uncased-emotion`

Use Cases
- Gauge public sentiment during elections, crises, or major events
- Detect emotional trends in Twitter posts or news coverage
- Aid researchers in **sentiment analysis**, **threat detection**, or **societal behavior mapping**

---

🚀 Features

🗞️ News Headlines Analysis
- Choose from top Indian cities and news sources
- Scrape latest headlines and analyze emotions
- Visualize results using **bar** and **line charts**

🐤 Twitter Analysis
- Scrape tweets from users, hashtags, or keywords
- Detect emotional tone in real-time Twitter discussions
- Trend analysis via **seasonal decomposition of likes over time**

---

🛠️ Tech Stack

| Component        | Tool/Library                                |
|------------------|---------------------------------------------|
| Frontend UI      | [Streamlit](https://streamlit.io/)          |
| Emotion Model    | `distilbert-base-uncased-emotion` (HuggingFace Transformers) |
| Twitter Scraping | [`ntscraper`](https://pypi.org/project/ntscraper/) |
| News Scraping    | `requests`, `BeautifulSoup`                 |
| Visualization    | `matplotlib`, `pandas`, `Streamlit charts` |
| Trend Analysis   | `statsmodels`                               |

---

📦 Installation

1. Clone this repo
   git clone https://github.com/yourusername/tak-analyzer.git
   cd tak-analyzer
   
2. Install dependencies
   pip install -r requirements.txt
   
3. Run the Streamlit app
   streamlit run app.py
