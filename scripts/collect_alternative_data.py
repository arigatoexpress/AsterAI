#!/usr/bin/env python3
"""
Alternative Data Sources Collection
Collects sentiment data, Google Trends, Fear & Greed Index, and whale movements.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import aiohttp
import json
import logging
from dataclasses import dataclass
from pytrends.request import TrendReq
import praw  # Reddit API
from textblob import TextBlob

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.data.api_manager import APIKeyManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AlternativeDataSource:
    """Alternative data source configuration"""
    name: str
    data_type: str  # 'sentiment', 'search_volume', 'on_chain', 'market_sentiment'
    update_frequency: str  # 'hourly', 'daily', 'real-time'
    requires_api_key: bool


class AlternativeDataCollector:
    """
    Collects alternative data including:
    - Social media sentiment (Twitter, Reddit)
    - Google Trends search volume
    - Fear & Greed Index
    - Whale movements and on-chain data
    - News sentiment
    """
    
    def __init__(self, output_dir: str = "data/historical/alternative_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API manager
        self.api_manager = APIKeyManager()
        self.api_manager.load_credentials()
        
        # Initialize Google Trends
        self.pytrends = TrendReq(hl='en-US', tz=360)
        
        # Reddit API (if credentials available)
        self.reddit = None
        self._init_reddit()
        
        # Collection settings
        self.start_date = datetime.now() - timedelta(days=90)  # 3 months for alternative data
        self.end_date = datetime.now()
        
        # Keywords to track
        self.crypto_keywords = [
            "bitcoin", "ethereum", "crypto", "cryptocurrency", "defi",
            "altcoin", "blockchain", "btc", "eth", "bull market", "bear market"
        ]
        
        self.market_keywords = [
            "stock market", "recession", "inflation", "federal reserve",
            "interest rates", "market crash", "bull run", "sp500"
        ]
        
        self.session = None
    
    def _init_reddit(self):
        """Initialize Reddit API if credentials available"""
        # Reddit requires app credentials (not included in default setup)
        # Users can add their own Reddit app credentials
        try:
            reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
            reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
            reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'AsterAI/1.0')
            
            if reddit_client_id and reddit_secret:
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_secret,
                    user_agent=reddit_user_agent
                )
                logger.info("Reddit API initialized")
        except Exception as e:
            logger.warning(f"Reddit API not initialized: {e}")
    
    async def initialize(self):
        """Initialize collectors"""
        logger.info("Initializing alternative data collectors...")
        self.session = aiohttp.ClientSession()
        logger.info("✅ Alternative data collectors initialized")
    
    async def collect_all_data(self) -> Dict[str, Any]:
        """Collect all alternative data"""
        logger.info("Starting alternative data collection...")
        
        results = {
            'sentiment': {},
            'search_trends': {},
            'market_sentiment': {},
            'on_chain': {},
            'news': {}
        }
        
        # Collect Google Trends data
        try:
            logger.info("Collecting Google Trends data...")
            trends_data = await self._collect_google_trends()
            results['search_trends'] = trends_data
        except Exception as e:
            logger.error(f"Google Trends error: {e}")
        
        # Collect Fear & Greed Index
        try:
            logger.info("Collecting Fear & Greed Index...")
            fng_data = await self._collect_fear_greed_index()
            results['market_sentiment']['fear_greed'] = fng_data
        except Exception as e:
            logger.error(f"Fear & Greed error: {e}")
        
        # Collect Reddit sentiment
        if self.reddit:
            try:
                logger.info("Collecting Reddit sentiment...")
                reddit_data = await self._collect_reddit_sentiment()
                results['sentiment']['reddit'] = reddit_data
            except Exception as e:
                logger.error(f"Reddit error: {e}")
        
        # Collect news sentiment (if NewsAPI key available)
        if self.api_manager.credentials.newsapi_key:
            try:
                logger.info("Collecting news sentiment...")
                news_data = await self._collect_news_sentiment()
                results['news'] = news_data
            except Exception as e:
                logger.error(f"News API error: {e}")
        
        # Collect on-chain data
        try:
            logger.info("Collecting on-chain metrics...")
            onchain_data = await self._collect_onchain_data()
            results['on_chain'] = onchain_data
        except Exception as e:
            logger.error(f"On-chain data error: {e}")
        
        # Save all results
        self._save_results(results)
        
        return results
    
    async def _collect_google_trends(self) -> Dict[str, pd.DataFrame]:
        """Collect Google Trends data for keywords"""
        trends_data = {}
        
        # Crypto trends
        try:
            # Process in batches (max 5 keywords per request)
            for i in range(0, len(self.crypto_keywords), 5):
                batch = self.crypto_keywords[i:i+5]
                
                self.pytrends.build_payload(
                    batch,
                    timeframe=f'{self.start_date.strftime("%Y-%m-%d")} {self.end_date.strftime("%Y-%m-%d")}'
                )
                
                # Get interest over time
                interest_df = self.pytrends.interest_over_time()
                
                if not interest_df.empty:
                    # Remove 'isPartial' column if present
                    if 'isPartial' in interest_df.columns:
                        interest_df = interest_df.drop('isPartial', axis=1)
                    
                    for keyword in batch:
                        if keyword in interest_df.columns:
                            trends_data[f'crypto_{keyword}'] = interest_df[keyword]
                
                # Brief pause to avoid rate limiting
                await asyncio.sleep(2)
            
            # Market trends
            for i in range(0, len(self.market_keywords), 5):
                batch = self.market_keywords[i:i+5]
                
                self.pytrends.build_payload(
                    batch,
                    timeframe=f'{self.start_date.strftime("%Y-%m-%d")} {self.end_date.strftime("%Y-%m-%d")}'
                )
                
                interest_df = self.pytrends.interest_over_time()
                
                if not interest_df.empty:
                    if 'isPartial' in interest_df.columns:
                        interest_df = interest_df.drop('isPartial', axis=1)
                    
                    for keyword in batch:
                        if keyword in interest_df.columns:
                            trends_data[f'market_{keyword}'] = interest_df[keyword]
                
                await asyncio.sleep(2)
            
            # Create combined DataFrame
            if trends_data:
                combined_df = pd.DataFrame(trends_data)
                combined_df.index.name = 'date'
                
                # Save to file
                output_file = self.output_dir / "google_trends.parquet"
                combined_df.to_parquet(output_file)
                logger.info(f"✅ Saved Google Trends data: {len(combined_df)} records")
                
                return {'combined': combined_df, 'keywords': list(trends_data.keys())}
            
        except Exception as e:
            logger.error(f"Google Trends error: {e}")
            
        return {}
    
    async def _collect_fear_greed_index(self) -> pd.DataFrame:
        """Collect Crypto Fear & Greed Index"""
        url = "https://api.alternative.me/fng/"
        params = {
            'limit': 0,  # Get all available data
            'format': 'json'
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                if 'data' in data:
                    # Convert to DataFrame
                    df = pd.DataFrame(data['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                    df['value'] = df['value'].astype(float)
                    df = df.set_index('timestamp')
                    df = df.sort_index()
                    
                    # Map value classifications
                    df['classification'] = df['value_classification']
                    
                    # Create additional features
                    df['extreme_fear'] = (df['value'] < 25).astype(int)
                    df['fear'] = ((df['value'] >= 25) & (df['value'] < 45)).astype(int)
                    df['neutral'] = ((df['value'] >= 45) & (df['value'] < 55)).astype(int)
                    df['greed'] = ((df['value'] >= 55) & (df['value'] < 75)).astype(int)
                    df['extreme_greed'] = (df['value'] >= 75).astype(int)
                    
                    # Save to file
                    output_file = self.output_dir / "fear_greed_index.parquet"
                    df.to_parquet(output_file)
                    logger.info(f"✅ Saved Fear & Greed Index: {len(df)} records")
                    
                    return df
        
        return pd.DataFrame()
    
    async def _collect_reddit_sentiment(self) -> Dict[str, Any]:
        """Collect sentiment from Reddit cryptocurrency subreddits"""
        if not self.reddit:
            return {}
        
        subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'altcoin', 'defi']
        sentiment_data = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts
                for submission in subreddit.hot(limit=100):
                    # Analyze sentiment
                    title_sentiment = TextBlob(submission.title).sentiment
                    
                    if submission.selftext:
                        body_sentiment = TextBlob(submission.selftext).sentiment
                        avg_polarity = (title_sentiment.polarity + body_sentiment.polarity) / 2
                        avg_subjectivity = (title_sentiment.subjectivity + body_sentiment.subjectivity) / 2
                    else:
                        avg_polarity = title_sentiment.polarity
                        avg_subjectivity = title_sentiment.subjectivity
                    
                    sentiment_data.append({
                        'timestamp': datetime.fromtimestamp(submission.created_utc),
                        'subreddit': subreddit_name,
                        'title': submission.title,
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'upvote_ratio': submission.upvote_ratio,
                        'polarity': avg_polarity,
                        'subjectivity': avg_subjectivity,
                        'sentiment': 'positive' if avg_polarity > 0.1 else 'negative' if avg_polarity < -0.1 else 'neutral'
                    })
                
                logger.info(f"✅ Collected sentiment from r/{subreddit_name}")
                
            except Exception as e:
                logger.error(f"Reddit error for r/{subreddit_name}: {e}")
        
        if sentiment_data:
            # Convert to DataFrame
            df = pd.DataFrame(sentiment_data)
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            # Save to file
            output_file = self.output_dir / "reddit_sentiment.parquet"
            df.to_parquet(output_file)
            logger.info(f"✅ Saved Reddit sentiment: {len(df)} posts analyzed")
            
            # Calculate aggregated metrics
            daily_sentiment = df.resample('D').agg({
                'polarity': 'mean',
                'score': 'sum',
                'num_comments': 'sum'
            })
            
            return {
                'raw_data': df,
                'daily_aggregated': daily_sentiment,
                'subreddits_analyzed': subreddits
            }
        
        return {}
    
    async def _collect_news_sentiment(self) -> Dict[str, Any]:
        """Collect news sentiment using NewsAPI"""
        news_data = []
        
        # Define queries
        queries = [
            'cryptocurrency OR bitcoin OR ethereum',
            'stock market OR S&P 500',
            'federal reserve OR interest rates',
            'inflation OR recession'
        ]
        
        base_url = "https://newsapi.org/v2/everything"
        
        for query in queries:
            params = {
                'q': query,
                'from': self.start_date.strftime('%Y-%m-%d'),
                'to': self.end_date.strftime('%Y-%m-%d'),
                'sortBy': 'popularity',
                'pageSize': 100,
                'apiKey': self.api_manager.credentials.newsapi_key
            }
            
            try:
                async with self.session.get(base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for article in data.get('articles', []):
                            # Analyze sentiment
                            title = article.get('title', '')
                            description = article.get('description', '')
                            
                            if title:
                                title_sentiment = TextBlob(title).sentiment
                                
                                if description:
                                    desc_sentiment = TextBlob(description).sentiment
                                    avg_polarity = (title_sentiment.polarity + desc_sentiment.polarity) / 2
                                else:
                                    avg_polarity = title_sentiment.polarity
                                
                                news_data.append({
                                    'timestamp': pd.to_datetime(article['publishedAt']),
                                    'source': article.get('source', {}).get('name', 'Unknown'),
                                    'title': title,
                                    'url': article.get('url', ''),
                                    'query': query,
                                    'polarity': avg_polarity,
                                    'sentiment': 'positive' if avg_polarity > 0.1 else 'negative' if avg_polarity < -0.1 else 'neutral'
                                })
                
                logger.info(f"✅ Collected news for query: {query}")
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"NewsAPI error for query '{query}': {e}")
        
        if news_data:
            # Convert to DataFrame
            df = pd.DataFrame(news_data)
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            # Save to file
            output_file = self.output_dir / "news_sentiment.parquet"
            df.to_parquet(output_file)
            logger.info(f"✅ Saved news sentiment: {len(df)} articles analyzed")
            
            # Calculate aggregated metrics
            daily_sentiment = df.resample('D').agg({
                'polarity': 'mean'
            })
            
            sentiment_by_source = df.groupby('source')['polarity'].mean().sort_values()
            
            return {
                'raw_data': df,
                'daily_aggregated': daily_sentiment,
                'by_source': sentiment_by_source,
                'total_articles': len(df)
            }
        
        return {}
    
    async def _collect_onchain_data(self) -> Dict[str, pd.DataFrame]:
        """Collect on-chain metrics (free sources)"""
        onchain_data = {}
        
        # Glassnode free API endpoints (limited)
        # Alternative: blockchain.info for basic Bitcoin metrics
        
        try:
            # Bitcoin network stats
            btc_stats_url = "https://api.blockchain.info/charts/n-transactions"
            params = {
                'timespan': '90days',
                'format': 'json'
            }
            
            async with self.session.get(btc_stats_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'values' in data:
                        df = pd.DataFrame(data['values'])
                        df['timestamp'] = pd.to_datetime(df['x'], unit='s')
                        df = df.set_index('timestamp')
                        df = df.rename(columns={'y': 'btc_transactions'})
                        onchain_data['btc_transactions'] = df
            
            # Hash rate
            hashrate_url = "https://api.blockchain.info/charts/hash-rate"
            async with self.session.get(hashrate_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'values' in data:
                        df = pd.DataFrame(data['values'])
                        df['timestamp'] = pd.to_datetime(df['x'], unit='s')
                        df = df.set_index('timestamp')
                        df = df.rename(columns={'y': 'btc_hashrate'})
                        onchain_data['btc_hashrate'] = df
            
            # Combine all on-chain metrics
            if onchain_data:
                combined_df = pd.concat(onchain_data.values(), axis=1)
                
                # Save to file
                output_file = self.output_dir / "onchain_metrics.parquet"
                combined_df.to_parquet(output_file)
                logger.info(f"✅ Saved on-chain metrics: {len(combined_df)} records")
                
                return {'combined': combined_df, 'metrics': list(onchain_data.keys())}
            
        except Exception as e:
            logger.error(f"On-chain data error: {e}")
        
        return {}
    
    def _save_results(self, results: Dict[str, Any]):
        """Save collection summary"""
        summary = {
            'collection_time': datetime.now().isoformat(),
            'data_collected': {
                'google_trends': bool(results.get('search_trends')),
                'fear_greed_index': bool(results.get('market_sentiment', {}).get('fear_greed')),
                'reddit_sentiment': bool(results.get('sentiment', {}).get('reddit')),
                'news_sentiment': bool(results.get('news')),
                'onchain_metrics': bool(results.get('on_chain'))
            },
            'date_range': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat()
            }
        }
        
        # Add specific metrics
        if results.get('search_trends', {}).get('keywords'):
            summary['google_trends_keywords'] = results['search_trends']['keywords']
        
        if results.get('news', {}).get('total_articles'):
            summary['news_articles_analyzed'] = results['news']['total_articles']
        
        if results.get('sentiment', {}).get('reddit', {}).get('subreddits_analyzed'):
            summary['reddit_subreddits'] = results['sentiment']['reddit']['subreddits_analyzed']
        
        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Alternative data collection summary saved to {summary_file}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()


async def main():
    """Main execution"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║             Alternative Data Sources Collection                ║
║         Sentiment, Trends, Fear & Greed, On-Chain             ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    collector = AlternativeDataCollector()
    
    try:
        await collector.initialize()
        results = await collector.collect_all_data()
        
        print(f"\n✅ Collection complete!")
        print(f"   Data saved to: {collector.output_dir}")
        
    finally:
        await collector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

