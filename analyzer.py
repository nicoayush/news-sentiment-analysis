import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from ntscraper import Nitter
from transformers import pipeline
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
import streamlit.components.v1 as components



# Load pre-trained emotion detection model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
# Function to scrape tweets and save them to a CSV file
def create_tweets_dataset(query, no_of_tweets, search_mode="user"):
    scraper = Nitter()
    
    try:
        # Scraping tweets
        if search_mode == "user":
            tweets = scraper.get_tweets(query, mode="user", number=no_of_tweets)
        elif search_mode == "hashtag":
            tweets = scraper.get_tweets(query, mode="hashtag", number=no_of_tweets)
        else:
            tweets = scraper.get_tweets(query, mode="term", number=no_of_tweets)

        # Check if tweets were retrieved
        if not tweets or 'tweets' not in tweets or not tweets['tweets']:
            st.error("No tweets found or invalid response received.")
            return None

        # Prepare data structure
        data = {
            'link': [],
            'text': [],
            'user': [],
            'likes': [],
            'quotes': [],
            'retweets': [],
            'comments': [],
            'timestamp': []
        }

        # Parse tweets
        for tweet in tweets['tweets']:
            data['link'].append(tweet.get('link', 'N/A'))
            data['text'].append(tweet.get('text', 'N/A'))
            data['user'].append(tweet.get('user', {}).get('name', 'N/A'))
            data['likes'].append(tweet.get('stats', {}).get('likes', 0))
            data['quotes'].append(tweet.get('stats', {}).get('quotes', 0))
            data['retweets'].append(tweet.get('stats', {}).get('retweets', 0))
            data['comments'].append(tweet.get('stats', {}).get('comments', 0))
            data['timestamp'].append(tweet.get('timestamp', None))

        # Save data to CSV
        filename = f"{query.replace('/', '_')}_tweets_data.csv"
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

        return filename

    except IndexError as e:
        st.error("Error: Invalid tweet data format received.")
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
        print(f"Exception details: {e}")

    return None


# Function to analyze detailed emotions
def analyze_emotions(tweets):
    emotions_count = {"joy": 0, "anger": 0, "sadness": 0, "fear": 0, "surprise": 0, "disgust": 0}

    valid_tweets = [tweet for tweet in tweets if isinstance(tweet, str) and tweet.strip()]

    if not valid_tweets:
        return {emotion: 0 for emotion in emotions_count}
    
    truncated_texts = [emotion_classifier.decode(emotion_classifier(text, max_length=512, truncation=True)['input_ids']) for text in valid_tweets]

    results = emotion_classifier(valid_tweets)
    
    for result in results:
        emotion = result['label'].lower()  # Convert to lowercase for consistency
        if emotion in emotions_count:
            emotions_count[emotion] += 1

    total = len(valid_tweets)
    emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotions_count.items()}
    
    return emotion_percentages

# Read tweets from CSV file
def read_tweets_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        tweets = df["text"].tolist()
        timestamps = pd.to_datetime(df["timestamp"], errors='coerce')
        return tweets, timestamps
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' does not exist.")
        return [], []
    except KeyError:
        st.error(f"Error: The required columns are not found in the file.")
        return [], []

# Function to detect trends and anomalies
def detect_trends_and_anomalies(timestamps, likes):
    if timestamps.empty or likes.empty:
        return None  # Not enough data to analyze

    df = pd.DataFrame({'timestamp': timestamps, 'likes': likes})
    
    # Remove rows with NaT in the timestamp column
    df = df.dropna(subset=['timestamp'])
    
    if df.empty:
        return None  # No valid data to analyze
    
    df.set_index('timestamp', inplace=True)
    
    # Ensure the 'timestamp' column is in datetime format
    df.index = pd.to_datetime(df.index, errors='coerce')
    
    # Drop rows with invalid timestamps after coercion
    df = df.dropna()
    
    if df.empty:
        return None  # No valid data to analyze
    
    df = df.resample('D').sum()  # Resample by day and sum likes
    
    decomposition = seasonal_decompose(df['likes'].fillna(0), model='additive', period=30)
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()
    residual = decomposition.resid.dropna()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
    df['likes'].plot(ax=axes[0], title='Likes Over Time')
    axes[0].set_ylabel('Likes')
    trend.plot(ax=axes[1], title='Trend Component')
    axes[1].set_ylabel('Likes')
    residual.plot(ax=axes[2], title='Residual Component')
    axes[2].set_ylabel('Likes')
    
    plt.tight_layout()
    return fig


# URL mapping based on selected source
url_map = {
    "Times Now": {
        'Kolkata': 'https://www.timesnownews.com/kolkata',
        'Delhi': 'https://www.timesnownews.com/delhi',
        'Mumbai': 'https://www.timesnownews.com/mumbai',
        'Chennai': 'https://www.timesnownews.com/chennai',
        'Bengaluru': 'https://www.timesnownews.com/bengaluru',
        'Hyderabad': 'https://www.timesnownews.com/hyderabad',
        'Ahmedabad': 'https://www.timesnownews.com/city/ahmedabad'
    },
    "Times of India": {
        'Kolkata': 'https://timesofindia.indiatimes.com/city/kolkata',
        'Delhi': 'https://timesofindia.indiatimes.com/city/delhi',
        'Mumbai': 'https://timesofindia.indiatimes.com/city/mumbai',
        'Chennai': 'https://timesofindia.indiatimes.com/city/chennai',
        'Bengaluru': 'https://timesofindia.indiatimes.com/city/bangalore',
        'Hyderabad': 'https://timesofindia.indiatimes.com/city/hyderabad',
        'Ahmedabad': 'https://timesofindia.indiatimes.com/city/ahmedabad'
    }
}

# Function to process the selected URL and extract headlines
def process_url(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        htmlcontent = r.content
        soup = BeautifulSoup(htmlcontent, 'html.parser')
        anchors = soup.find_all('a')
        all_links = set()
        for link in anchors:
            href = link.get('href')
            if href and href != '#':
                link_text = link.get_text().strip()
                all_links.add(link_text)
        return all_links
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while processing {url}: {e}")
        return set()

# Function to analyze emotions in headlines
def analyze_emotions(headlines):
    emotions_count = {"joy": 0, "anger": 0, "sadness": 0, "fear": 0, "surprise": 0}

    valid_headlines = [headline for headline in headlines if isinstance(headline, str) and headline.strip()]

    if not valid_headlines:
        return {emotion: 0 for emotion in emotions_count}

    results = emotion_classifier(valid_headlines)
    
    for result in results:
        emotion = result['label'].lower()
        if emotion in emotions_count:
            emotions_count[emotion] += 1

    total = len(valid_headlines)
    emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotions_count.items()}
    
    return emotion_percentages

# Streamlit app setup
st.set_page_config(layout="centered", page_title="Integrated Platform")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a page", ["News Headlines", "Twitter Analysis"])

if app_mode == "News Headlines":
    st.subheader("News Headlines")

    # Sidebar for options
    with st.sidebar:
        st.subheader("Choose your city")
        city = st.selectbox("Select your city", ["Kolkata", "Delhi", "Mumbai", "Chennai", "Bengaluru", "Hyderabad", "Ahmedabad"], key="city_select")

        source = st.selectbox("Select the news source", ["Times Now", "Times of India"], key="source_select")

        # Initialize placeholder for emotion analysis
        emotion_placeholder = st.empty()

    # Main content area
    st.subheader(f"Headlines for {city} from {source}")

    # Button to start the news fetching and analysis process
    if st.button("Check it Out!", key="check_button"):
        selected_url = url_map[source][city]
        headlines = process_url(selected_url)
        emotion_percentages = analyze_emotions(headlines)
        st.line_chart(pd.Series(emotion_percentages))
        
        if headlines:
            # Display headlines
            for headline in headlines:
                if city.lower() in headline.lower():
                    st.write(f"- {headline}")



            # Analyze emotions for all headlines collectively
            emotion_percentages = analyze_emotions(headlines)
            
            # Update emotion analysis in the sidebar
            with emotion_placeholder.container():
                st.subheader(f"Overall Sentiment Analysis for {city}")
           
                st.write("The general sentiment in the news headlines:")
                for emotion, percentage in emotion_percentages.items():
                    st.write(f"{emotion.capitalize()}: {percentage:.2f}%")
            # st.write(emotion_percentages)
            st.bar_chart(pd.Series(emotion_percentages))
            st.write("Made by Team Garuda Rakshak")

        else:
            st.write("No headlines found.")
                            
              

elif app_mode == "Twitter Analysis":
    st.subheader("Twitter Analysis")

    # User input for Twitter analysis
    query = st.text_input("Enter the username or hashtag", value="elonmusk")
    #no_of_tweets = st.slider("Number of tweets", min_value=10, max_value=200, value=50)
    search_mode = st.selectbox("Search mode", ["user", "hashtag", "term"], index=0)
    
    if st.button("Fetch Tweets"):
        filename = create_tweets_dataset(query, 100 , search_mode)
        st.write(f"Tweets saved to {filename}")

        tweets, timestamps = read_tweets_from_csv(filename)
        
        if tweets:
            st.write(f"Number of tweets: {len(tweets)}")

            # Display a sample of tweets
            emotion_percentages = analyze_emotions(tweets)
            st.line_chart(pd.Series(emotion_percentages))

            # Analyze emotions in tweets
            if (search_mode=="user"):
                st.write("Embed a Twitter profile timeline below:")
    
                twitter_profile_url = f"https://twitter.com/{query}?ref_src=twsrc%5Etfw"
        
        # Embed Twitter timeline using <a> tag and <script>
                #coded by AYUSH CHAURASIA**
                st.markdown(f"""
        <a class="twitter-timeline" href="{twitter_profile_url}">Tweets by {query}</a>
        <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
        """, unsafe_allow_html=True)
           
            st.subheader("Emotion Analysis")
            for emotion, percentage in emotion_percentages.items():
                st.write(f"{emotion.capitalize()}: {percentage:.2f}%")
                
            st.subheader("Sample Tweets:")
            for tweet in tweets[:5]:
                st.write(f"- {tweet}")
            st.write("Made by Ayush Chaurasia")
