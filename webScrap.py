import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Database Connection Setup
DB_USER = 'tharaka'
DB_PASSWORD = '456'
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'movie_reviews_db'

# Create a connection to the MySQL database
engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')


# Step 1: Web Scraping
def extract_reviews(movie_url):
    response = requests.get(movie_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Scraping review content
    reviews = soup.find_all('div', class_='text show-more__control')
    review_list = [review.get_text() for review in reviews]

    return review_list


# Example IMDb URL for Joker movie reviews
movie_url = "https://www.imdb.com/title/tt7286456/reviews"

# Extract reviews
reviews = extract_reviews(movie_url)


# Step 2: Data Transformation
def transform_reviews(reviews):
    # Transform the data into a DataFrame
    df = pd.DataFrame(reviews, columns=['review_text'])
    return df


# Transform extracted reviews into a DataFrame
df_reviews = transform_reviews(reviews)


# Step 3: Load Data into MySQL
def load_data_to_sql(df, table_name):
    # Load DataFrame into a MySQL table
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)


# Load the reviews DataFrame to SQL
load_data_to_sql(df_reviews, 'movie_reviews')


# Step 4: Sentiment Analysis
def analyze_sentiment(review):
    # Use TextBlob for sentiment polarity (-1 to 1)
    sentiment = TextBlob(review).sentiment.polarity
    return sentiment


# Apply sentiment analysis to the reviews
df_reviews['sentiment'] = df_reviews['review_text'].apply(analyze_sentiment)

# Classify sentiment
df_reviews['sentiment_label'] = df_reviews['sentiment'].apply(
    lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Load sentiment data back to SQL
load_data_to_sql(df_reviews, 'movie_reviews_with_sentiment')


# Step 5: Visualization
def visualize_sentiment(df):
    # Sentiment distribution bar chart
    sentiment_counts = df['sentiment_label'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title('Sentiment Distribution of Movie Reviews')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    # Word Cloud for Positive Reviews
    positive_reviews = df[df['sentiment_label'] == 'Positive']['review_text']
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(positive_reviews))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# Visualize sentiment analysis results
visualize_sentiment(df_reviews)
