"""
Gemini-Powered Sentiment Analyzer for HFT

This service continuously monitors news sources, uses the Gemini API for
real-time sentiment analysis, and publishes scores to a Google Cloud Pub/Sub topic.
The trading agents can then subscribe to this topic to get a live sentiment feed.
"""

import os
import asyncio
import logging
import json
import time
from typing import Dict, List
from google.cloud import pubsub_v1
import google.generativeai as genai
# Using a placeholder for a news API client
# from newsapi import NewsApiClient

# --- Configuration ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "hft-aster-trader")
PUBSUB_TOPIC_ID = "hft-sentiment"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY") # Placeholder for a real news API
FETCH_INTERVAL_SECONDS = 60 * 5 # Fetch news every 5 minutes

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiSentimentAnalyzer:
    """
    Analyzes financial news sentiment using Gemini and publishes scores.
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        if not GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID environment variable not set.")

        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini Pro model initialized.")

        # Configure Pub/Sub
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TOPIC_ID)
        logger.info(f"Pub/Sub publisher initialized for topic: {self.topic_path}")

        # Placeholder for a real news client
        # self.news_client = NewsApiClient(api_key=NEWS_API_KEY)
        # logger.info("News API client initialized.")

    def get_sentiment_from_gemini(self, text: str) -> float:
        """
        Gets a sentiment score from Gemini for a given text.

        Args:
            text: The news headline or article summary.

        Returns:
            A sentiment score between -1.0 (very bearish) and 1.0 (very bullish).
        """
        try:
            prompt = (
                "Analyze the sentiment of the following financial news headline for the "
                "crypto market. Your response MUST be only a single floating-point number "
                "between -1.0 (extremely bearish) and 1.0 (extremely bullish). "
                "Do not include any other text or explanation.\n\n"
                f"Headline: '{text}'"
            )
            response = self.model.generate_content(prompt)

            # Extract the float from the response
            score_text = response.text.strip()
            score = float(score_text)

            # Clamp the score to the expected range
            score = max(-1.0, min(1.0, score))

            logger.info(f"Sentiment score for '{text}': {score:.2f}")
            return score
        except Exception as e:
            logger.error(f"Error getting sentiment from Gemini: {e}")
            return 0.0 # Return neutral sentiment on error

    def fetch_crypto_news(self) -> List[Dict]:
        """
        Fetches the latest crypto news headlines.
        This is a placeholder and should be replaced with a real news API implementation.
        """
        logger.info("Fetching latest crypto news (using placeholder data)...")
        # In a real implementation, you would use a library like 'newsapi-python'
        # response = self.news_client.get_everything(q='crypto OR bitcoin OR ethereum',
        #                                           language='en',
        #                                           sort_by='publishedAt',
        #                                           page_size=20)
        # return response.get('articles', [])

        # Placeholder data to simulate a real news feed
        return [
            {"title": "Bitcoin Surges Past $80,000 as Institutional Interest Peaks", "source": {"name": "Crypto Times"}},
            {"title": "Ethereum's New Upgrade Promises 100x Scalability, But Faces Delays", "source": {"name": "DeFi World"}},
            {"title": "Regulatory Crackdown Fears Intensify as SEC Chairman Issues Stern Warning", "source": {"name": "Financial News"}},
            {"title": "Aster DEX volume explodes after announcing 1001x leverage on new assets", "source": {"name": "DEX Reporter"}},
            {"title": "Market Cools Off After Record-Breaking Rally, Analysts Predict Correction", "source": {"name": "Coin Journal"}},
        ]

    def publish_sentiment_score(self, symbol: str, score: float, source: str):
        """
        Publishes a sentiment score to the Google Cloud Pub/Sub topic.

        Args:
            symbol: The crypto symbol (e.g., 'BTC/USD'). For now, we use a generic 'market' symbol.
            score: The sentiment score from Gemini.
            source: The source of the news (e.g., the publication name).
        """
        try:
            message_data = {
                "symbol": symbol,
                "sentiment_score": score,
                "source": source,
                "timestamp": time.time()
            }
            data = json.dumps(message_data).encode("utf-8")

            future = self.publisher.publish(self.topic_path, data)
            future.result()  # Wait for the publish to complete

            logger.info(f"Published sentiment score {score:.2f} for {symbol} to Pub/Sub.")
        except Exception as e:
            logger.error(f"Failed to publish to Pub/Sub: {e}")

    async def run(self):
        """
        The main loop that fetches news, analyzes sentiment, and publishes scores.
        """
        logger.info("Starting Gemini Sentiment Analyzer loop...")
        while True:
            try:
                articles = self.fetch_crypto_news()
                if not articles:
                    logger.info("No new articles found.")

                for article in articles:
                    headline = article.get("title")
                    source = article.get("source", {}).get("name", "Unknown")
                    if headline:
                        # Get sentiment score
                        score = self.get_sentiment_from_gemini(headline)

                        # Publish score (using a generic market symbol for now)
                        self.publish_sentiment_score("market-sentiment", score, source)

                        # Small delay to avoid hitting API rate limits
                        await asyncio.sleep(5)

                logger.info(f"Completed news cycle. Waiting for {FETCH_INTERVAL_SECONDS} seconds...")
                await asyncio.sleep(FETCH_INTERVAL_SECONDS)

            except Exception as e:
                logger.error(f"An error occurred in the main loop: {e}")
                await asyncio.sleep(60) # Wait longer on error

if __name__ == "__main__":
    analyzer = GeminiSentimentAnalyzer()
    asyncio.run(analyzer.run())

