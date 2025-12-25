import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline

# Load dataset
DATA_FILE = "chat_conversations_dataset.csv"
data = pd.read_csv(DATA_FILE)

def plot_sentiment_distribution():
    """
    Perform sentiment analysis on 'full_conversation' texts and plot sentiment label counts.
    """
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
    
    # Analyze sentiments for each full conversation
    sentiments = data['full_conversation'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
    sentiment_counts = sentiments.value_counts()
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis', ax=ax)
    ax.set_title("Sentiment Distribution of Chat Conversations")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def plot_customer_rating_distribution():
    """
    Plot distribution of customer ratings from data.
    """
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(data['customer_rating'], bins=10, kde=False, color='blue', ax=ax)
    ax.set_title("Customer Ratings Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def plot_common_issues_clusters(num_clusters=5):
    """
    Perform TF-IDF vectorization and KMeans clustering on full_conversation texts,
    then plot the frequency of each cluster as a bar chart.
    """
    issue_descriptions = data['full_conversation'].tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(issue_descriptions)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=[f"Cluster {i}" for i in cluster_counts.index], y=cluster_counts.values, palette='magma', ax=ax)
    ax.set_title("Frequency of Common Issue Clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Issues")
    fig.tight_layout()
    return fig

def plot_issue_type_distribution():
    """
    Plot the distribution of common vs uncommon issues based on customer rating threshold 3.
    """
    issue_types = data['customer_rating'].apply(lambda x: "Common Issue" if x >= 3 else "Uncommon Issue")
    issue_type_counts = issue_types.value_counts()
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie(issue_type_counts.values, labels=issue_type_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
    ax.set_title("Distribution of Common vs Uncommon Issues")
    fig.tight_layout()
    return fig

def main():
    plot_sentiment_distribution()
    plot_customer_rating_distribution()
    plot_common_issues_clusters()
    plot_issue_type_distribution()

if __name__ == "__main__":
    main()
