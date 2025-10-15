"""
Sentiment and news data ingestion.
Supports X/Twitter, CryptoPanic, NewsAPI, and RSS feeds.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
import feedparser
import snscrape.modules.twitter as sntwitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

logger = logging.getLogger(__name__)


@dataclass
class SentimentData:
    """Sentiment data structure."""
    timestamp: datetime
    source: str  # 'twitter', 'cryptopanic', 'newsapi', 'rss'
    text: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0 to 1
    symbols: List[str]  # Related trading symbols
    metadata: Dict[str, Any]


@dataclass
class NewsConfig:
    """Configuration for news sources."""
    cryptopanic_api_key: Optional[str] = None
    newsapi_api_key: Optional[str] = None
    twitter_accounts: List[str] = None
    rss_feeds: List[str] = None
    keywords: List[str] = None
    symbols: List[str] = None


class SentimentAnalyzer:
    """Sentiment analysis using multiple methods."""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.sentiment_cache = {}
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using VADER."""
        # Use cached result if available
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
        
        # Analyze with VADER
        scores = self.vader.polarity_scores(text)
        
        # Cache result
        self.sentiment_cache[text] = scores
        
        return scores
    
    def get_sentiment_label(self, compound_score: float) -> str:
        """Convert compound score to label."""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_symbols(self, text: str, symbols: List[str]) -> List[str]:
        """Extract trading symbols from text."""
        found_symbols = []
        text_upper = text.upper()
        
        for symbol in symbols:
            if symbol.upper() in text_upper:
                found_symbols.append(symbol)
        
        return found_symbols


class TwitterIngestion:
    """Twitter/X data ingestion using snscrape."""
    
    def __init__(self, config: NewsConfig, sentiment_analyzer: SentimentAnalyzer):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.accounts = config.twitter_accounts or []
        self.keywords = config.keywords or []
        self.symbols = config.symbols or []
    
    async def fetch_tweets(self, 
                          accounts: List[str] = None, 
                          keywords: List[str] = None,
                          hours_back: int = 24) -> List[SentimentData]:
        """Fetch tweets from specified accounts and keywords."""
        accounts = accounts or self.accounts
        keywords = keywords or self.keywords
        
        sentiment_data = []
        
        # Fetch from accounts
        for account in accounts:
            try:
                tweets = self._fetch_account_tweets(account, hours_back)
                for tweet in tweets:
                    sentiment_data.append(self._process_tweet(tweet, 'account'))
            except Exception as e:
                logger.error(f"Error fetching tweets from {account}: {e}")
        
        # Fetch from keywords
        for keyword in keywords:
            try:
                tweets = self._fetch_keyword_tweets(keyword, hours_back)
                for tweet in tweets:
                    sentiment_data.append(self._process_tweet(tweet, 'keyword'))
            except Exception as e:
                logger.error(f"Error fetching tweets for {keyword}: {e}")
        
        return sentiment_data
    
    def _fetch_account_tweets(self, account: str, hours_back: int) -> List[Dict]:
        """Fetch tweets from a specific account."""
        tweets = []
        since_date = datetime.now() - timedelta(hours=hours_back)
        
        try:
            # Use snscrape to get tweets
            query = f"from:{account} since:{since_date.strftime('%Y-%m-%d')}"
            
            for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                tweets.append({
                    'id': tweet.id,
                    'text': tweet.content,
                    'timestamp': tweet.date,
                    'username': tweet.user.username,
                    'retweet_count': tweet.retweetCount,
                    'like_count': tweet.likeCount,
                    'reply_count': tweet.replyCount
                })
                
                # Limit to recent tweets
                if len(tweets) >= 100:
                    break
                    
        except Exception as e:
            logger.error(f"Error fetching tweets from {account}: {e}")
        
        return tweets
    
    def _fetch_keyword_tweets(self, keyword: str, hours_back: int) -> List[Dict]:
        """Fetch tweets containing specific keywords."""
        tweets = []
        since_date = datetime.now() - timedelta(hours=hours_back)
        
        try:
            query = f"{keyword} since:{since_date.strftime('%Y-%m-%d')} lang:en"
            
            for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                tweets.append({
                    'id': tweet.id,
                    'text': tweet.content,
                    'timestamp': tweet.date,
                    'username': tweet.user.username,
                    'retweet_count': tweet.retweetCount,
                    'like_count': tweet.likeCount,
                    'reply_count': tweet.replyCount
                })
                
                # Limit to recent tweets
                if len(tweets) >= 50:
                    break
                    
        except Exception as e:
            logger.error(f"Error fetching tweets for {keyword}: {e}")
        
        return tweets
    
    def _process_tweet(self, tweet: Dict, source_type: str) -> SentimentData:
        """Process a tweet into sentiment data."""
        text = tweet['text']
        
        # Analyze sentiment
        sentiment_scores = self.sentiment_analyzer.analyze_text(text)
        compound_score = sentiment_scores['compound']
        sentiment_label = self.sentiment_analyzer.get_sentiment_label(compound_score)
        
        # Calculate confidence based on compound score
        confidence = abs(compound_score)
        
        # Extract symbols
        symbols = self.sentiment_analyzer.extract_symbols(text, self.symbols)
        
        return SentimentData(
            timestamp=tweet['timestamp'],
            source='twitter',
            text=text,
            sentiment_score=compound_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            symbols=symbols,
            metadata={
                'tweet_id': tweet['id'],
                'username': tweet['username'],
                'retweet_count': tweet['retweet_count'],
                'like_count': tweet['like_count'],
                'reply_count': tweet['reply_count'],
                'source_type': source_type
            }
        )


class CryptoPanicIngestion:
    """CryptoPanic news ingestion."""
    
    def __init__(self, config: NewsConfig, sentiment_analyzer: SentimentAnalyzer):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.api_key = config.cryptopanic_api_key
        self.symbols = config.symbols or []
        self.base_url = "https://cryptopanic.com/api/v1"
    
    async def fetch_news(self, hours_back: int = 24) -> List[SentimentData]:
        """Fetch news from CryptoPanic."""
        if not self.api_key:
            logger.warning("CryptoPanic API key not provided")
            return []
        
        sentiment_data = []
        
        try:
            # Calculate timestamp for filtering
            since_timestamp = int((datetime.now() - timedelta(hours=hours_back)).timestamp())
            
            # Fetch news
            params = {
                'auth_token': self.api_key,
                'public': 'true',
                'filter': 'hot',
                'currencies': ','.join(self.symbols) if self.symbols else None
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/posts/", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for post in data.get('results', []):
                            # Filter by timestamp
                            post_timestamp = datetime.fromisoformat(post['created_at'].replace('Z', '+00:00'))
                            if post_timestamp.timestamp() >= since_timestamp:
                                sentiment_data.append(self._process_news_post(post))
                    else:
                        logger.error(f"CryptoPanic API error: {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching CryptoPanic news: {e}")
        
        return sentiment_data
    
    def _process_news_post(self, post: Dict) -> SentimentData:
        """Process a news post into sentiment data."""
        title = post.get('title', '')
        description = post.get('description', '')
        text = f"{title} {description}".strip()
        
        # Analyze sentiment
        sentiment_scores = self.sentiment_analyzer.analyze_text(text)
        compound_score = sentiment_scores['compound']
        sentiment_label = self.sentiment_analyzer.get_sentiment_label(compound_score)
        
        # Calculate confidence
        confidence = abs(compound_score)
        
        # Extract symbols
        symbols = self.sentiment_analyzer.extract_symbols(text, self.symbols)
        
        return SentimentData(
            timestamp=datetime.fromisoformat(post['created_at'].replace('Z', '+00:00')),
            source='cryptopanic',
            text=text,
            sentiment_score=compound_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            symbols=symbols,
            metadata={
                'post_id': post.get('id'),
                'url': post.get('url'),
                'domain': post.get('domain'),
                'votes': post.get('votes', {}),
                'source': post.get('source', {})
            }
        )


class NewsAPIIngestion:
    """NewsAPI ingestion."""
    
    def __init__(self, config: NewsConfig, sentiment_analyzer: SentimentAnalyzer):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.api_key = config.newsapi_api_key
        self.symbols = config.symbols or []
        self.base_url = "https://newsapi.org/v2"
    
    async def fetch_news(self, hours_back: int = 24) -> List[SentimentData]:
        """Fetch news from NewsAPI."""
        if not self.api_key:
            logger.warning("NewsAPI key not provided")
            return []
        
        sentiment_data = []
        
        try:
            # Calculate date for filtering
            since_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
            
            # Build query
            query_terms = []
            if self.symbols:
                query_terms.extend(self.symbols)
            if self.config.keywords:
                query_terms.extend(self.config.keywords)
            
            query = ' OR '.join(query_terms) if query_terms else 'cryptocurrency'
            
            params = {
                'apiKey': self.api_key,
                'q': query,
                'from': since_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 100
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/everything", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for article in data.get('articles', []):
                            sentiment_data.append(self._process_article(article))
                    else:
                        logger.error(f"NewsAPI error: {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
        
        return sentiment_data
    
    def _process_article(self, article: Dict) -> SentimentData:
        """Process a news article into sentiment data."""
        title = article.get('title', '')
        description = article.get('description', '')
        text = f"{title} {description}".strip()
        
        # Analyze sentiment
        sentiment_scores = self.sentiment_analyzer.analyze_text(text)
        compound_score = sentiment_scores['compound']
        sentiment_label = self.sentiment_analyzer.get_sentiment_label(compound_score)
        
        # Calculate confidence
        confidence = abs(compound_score)
        
        # Extract symbols
        symbols = self.sentiment_analyzer.extract_symbols(text, self.symbols)
        
        return SentimentData(
            timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
            source='newsapi',
            text=text,
            sentiment_score=compound_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            symbols=symbols,
            metadata={
                'url': article.get('url'),
                'source_name': article.get('source', {}).get('name'),
                'author': article.get('author'),
                'url_to_image': article.get('urlToImage')
            }
        )


class RSSIngestion:
    """RSS feed ingestion."""
    
    def __init__(self, config: NewsConfig, sentiment_analyzer: SentimentAnalyzer):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.feeds = config.rss_feeds or []
        self.symbols = config.symbols or []
    
    async def fetch_news(self, hours_back: int = 24) -> List[SentimentData]:
        """Fetch news from RSS feeds."""
        sentiment_data = []
        
        for feed_url in self.feeds:
            try:
                feed_data = self._fetch_rss_feed(feed_url, hours_back)
                sentiment_data.extend(feed_data)
            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_url}: {e}")
        
        return sentiment_data
    
    def _fetch_rss_feed(self, feed_url: str, hours_back: int) -> List[SentimentData]:
        """Fetch and parse RSS feed."""
        sentiment_data = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                # Parse timestamp
                entry_time = datetime(*entry.published_parsed[:6])
                
                # Check if within time window
                if entry_time >= datetime.now() - timedelta(hours=hours_back):
                    sentiment_data.append(self._process_rss_entry(entry, feed_url))
        
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {e}")
        
        return sentiment_data
    
    def _process_rss_entry(self, entry: Dict, feed_url: str) -> SentimentData:
        """Process RSS entry into sentiment data."""
        title = entry.get('title', '')
        description = entry.get('description', '')
        text = f"{title} {description}".strip()
        
        # Analyze sentiment
        sentiment_scores = self.sentiment_analyzer.analyze_text(text)
        compound_score = sentiment_scores['compound']
        sentiment_label = self.sentiment_analyzer.get_sentiment_label(compound_score)
        
        # Calculate confidence
        confidence = abs(compound_score)
        
        # Extract symbols
        symbols = self.sentiment_analyzer.extract_symbols(text, self.symbols)
        
        return SentimentData(
            timestamp=datetime(*entry.published_parsed[:6]),
            source='rss',
            text=text,
            sentiment_score=compound_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            symbols=symbols,
            metadata={
                'url': entry.get('link'),
                'feed_url': feed_url,
                'author': entry.get('author')
            }
        )


class SentimentIngestionManager:
    """Main manager for sentiment data ingestion."""
    
    def __init__(self, config: NewsConfig):
        self.config = config
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize ingestion sources
        self.twitter = TwitterIngestion(config, self.sentiment_analyzer)
        self.cryptopanic = CryptoPanicIngestion(config, self.sentiment_analyzer)
        self.newsapi = NewsAPIIngestion(config, self.sentiment_analyzer)
        self.rss = RSSIngestion(config, self.sentiment_analyzer)
    
    async def fetch_all_sentiment_data(self, hours_back: int = 24) -> List[SentimentData]:
        """Fetch sentiment data from all sources."""
        all_data = []
        
        # Fetch from all sources concurrently
        tasks = [
            self.twitter.fetch_tweets(hours_back=hours_back),
            self.cryptopanic.fetch_news(hours_back=hours_back),
            self.newsapi.fetch_news(hours_back=hours_back),
            self.rss.fetch_news(hours_back=hours_back)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in sentiment ingestion: {result}")
        
        return all_data
    
    def aggregate_sentiment_by_symbol(self, sentiment_data: List[SentimentData]) -> Dict[str, Dict]:
        """Aggregate sentiment data by trading symbol."""
        symbol_sentiment = {}
        
        for data in sentiment_data:
            for symbol in data.symbols:
                if symbol not in symbol_sentiment:
                    symbol_sentiment[symbol] = {
                        'total_score': 0.0,
                        'count': 0,
                        'positive_count': 0,
                        'negative_count': 0,
                        'neutral_count': 0,
                        'avg_confidence': 0.0,
                        'sources': set()
                    }
                
                symbol_data = symbol_sentiment[symbol]
                symbol_data['total_score'] += data.sentiment_score
                symbol_data['count'] += 1
                symbol_data['sources'].add(data.source)
                
                if data.sentiment_label == 'positive':
                    symbol_data['positive_count'] += 1
                elif data.sentiment_label == 'negative':
                    symbol_data['negative_count'] += 1
                else:
                    symbol_data['neutral_count'] += 1
                
                symbol_data['avg_confidence'] = (
                    symbol_data['avg_confidence'] * (symbol_data['count'] - 1) + data.confidence
                ) / symbol_data['count']
        
        # Calculate final metrics
        for symbol, data in symbol_sentiment.items():
            data['avg_sentiment'] = data['total_score'] / data['count'] if data['count'] > 0 else 0
            data['sentiment_ratio'] = data['positive_count'] / data['count'] if data['count'] > 0 else 0.5
            data['sources'] = list(data['sources'])
        
        return symbol_sentiment

