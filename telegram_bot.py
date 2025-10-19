"""
Telegram Bot Integration for Aster Trading Agent
Provides real-time notifications and control via Telegram
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import aiohttp
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TelegramConfig:
    """Configuration for Telegram bot"""
    bot_token: str
    chat_id: str
    enabled: bool = True
    notification_level: str = "all"  # all, trades_only, errors_only

class TelegramNotifier:
    """Handles Telegram notifications for trading bot"""
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{config.bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to Telegram"""
        if not self.config.enabled or not self.session:
            return False
            
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.config.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    logger.info("Telegram message sent successfully")
                    return True
                else:
                    logger.error(f"Failed to send Telegram message: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """Send trade execution notification"""
        if self.config.notification_level not in ["all", "trades_only"]:
            return False
            
        symbol = trade_data.get("symbol", "Unknown")
        side = trade_data.get("side", "Unknown")
        qty = trade_data.get("qty", 0)
        price = trade_data.get("price", 0)
        pnl = trade_data.get("pnl", 0)
        timestamp = trade_data.get("ts", datetime.now().isoformat())
        
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        pnl_emoji = "💰" if pnl > 0 else "📉" if pnl < 0 else "➖"
        
        message = f"""
{emoji} *TRADE EXECUTED* {emoji}

📊 *Symbol:* {symbol}
📈 *Side:* {side}
💵 *Quantity:* {qty}
💲 *Price:* ${price:,.4f}
{pnl_emoji} *PnL:* ${pnl:,.2f}
⏰ *Time:* {timestamp}

🤖 *Aster Trading Bot*
        """.strip()
        
        return await self.send_message(message)
    
    async def send_error_notification(self, error: str, context: str = "") -> bool:
        """Send error notification"""
        if self.config.notification_level not in ["all", "errors_only"]:
            return False
            
        message = f"""
🚨 *BOT ERROR* 🚨

❌ *Error:* {error}
📍 *Context:* {context}
⏰ *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🤖 *Aster Trading Bot*
        """.strip()
        
        return await self.send_message(message)
    
    async def send_status_update(self, status_data: Dict[str, Any]) -> bool:
        """Send bot status update"""
        if self.config.notification_level not in ["all"]:
            return False
            
        running = status_data.get("running", False)
        status_emoji = "🟢" if running else "🔴"
        
        message = f"""
{status_emoji} *BOT STATUS UPDATE* {status_emoji}

🤖 *Status:* {"Running" if running else "Stopped"}
⏰ *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 *Additional Info:*
{json.dumps(status_data, indent=2)}
        """.strip()
        
        return await self.send_message(message)
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send daily trading summary"""
        total_trades = summary_data.get("total_trades", 0)
        total_pnl = summary_data.get("total_pnl", 0)
        win_rate = summary_data.get("win_rate", 0)
        
        pnl_emoji = "💰" if total_pnl > 0 else "📉" if total_pnl < 0 else "➖"
        
        message = f"""
📊 *DAILY TRADING SUMMARY* 📊

📈 *Total Trades:* {total_trades}
{pnl_emoji} *Total PnL:* ${total_pnl:,.2f}
🎯 *Win Rate:* {win_rate:.1f}%
📅 *Date:* {datetime.now().strftime('%Y-%m-%d')}

🤖 *Aster Trading Bot*
        """.strip()
        
        return await self.send_message(message)

class TelegramCommandHandler:
    """Handles incoming Telegram commands"""
    
    def __init__(self, trading_agent_url: str):
        self.trading_agent_url = trading_agent_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def handle_command(self, command: str, chat_id: str) -> str:
        """Handle incoming Telegram commands"""
        if not self.session:
            return "❌ Bot not initialized"
            
        try:
            if command == "/start":
                return await self._handle_start()
            elif command == "/status":
                return await self._handle_status()
            elif command == "/start_trading":
                return await self._handle_start_trading()
            elif command == "/stop_trading":
                return await self._handle_stop_trading()
            elif command == "/help":
                return await self._handle_help()
            else:
                return "❌ Unknown command. Use /help for available commands."
                
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
            return f"❌ Error processing command: {str(e)}"
    
    async def _handle_start(self) -> str:
        return """
🤖 *Welcome to Aster Trading Bot!*

I can help you monitor and control your trading bot.

Available commands:
/status - Check bot status
/start_trading - Start the trading bot
/stop_trading - Stop the trading bot
/help - Show this help message

Use /help anytime for assistance!
        """.strip()
    
    async def _handle_status(self) -> str:
        try:
            async with self.session.get(f"{self.trading_agent_url}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    running = data.get("running", False)
                    status_emoji = "🟢" if running else "🔴"
                    
                    return f"""
{status_emoji} *Bot Status: {"Running" if running else "Stopped"}*

📊 *Details:*
{json.dumps(data, indent=2)}
                    """.strip()
                else:
                    return "❌ Failed to get bot status"
        except Exception as e:
            return f"❌ Error getting status: {str(e)}"
    
    async def _handle_start_trading(self) -> str:
        try:
            async with self.session.post(f"{self.trading_agent_url}/start") as response:
                if response.status == 200:
                    data = await response.json()
                    return f"✅ {data.get('message', 'Trading started successfully')}"
                else:
                    error_data = await response.json()
                    return f"❌ Failed to start trading: {error_data.get('detail', 'Unknown error')}"
        except Exception as e:
            return f"❌ Error starting trading: {str(e)}"
    
    async def _handle_stop_trading(self) -> str:
        try:
            async with self.session.post(f"{self.trading_agent_url}/stop") as response:
                if response.status == 200:
                    data = await response.json()
                    return f"✅ {data.get('message', 'Trading stopped successfully')}"
                else:
                    error_data = await response.json()
                    return f"❌ Failed to stop trading: {error_data.get('detail', 'Unknown error')}"
        except Exception as e:
            return f"❌ Error stopping trading: {str(e)}"
    
    async def _handle_help(self) -> str:
        return """
🤖 *Aster Trading Bot Commands*

/start - Welcome message
/status - Check current bot status
/start_trading - Start the trading bot
/stop_trading - Stop the trading bot
/help - Show this help message

📊 *Features:*
• Real-time trade notifications
• Bot status monitoring
• Remote control capabilities
• Daily trading summaries

For support, contact your bot administrator.
        """.strip()

# Global instances
telegram_notifier: Optional[TelegramNotifier] = None
telegram_config: Optional[TelegramConfig] = None

def initialize_telegram(bot_token: str, chat_id: str, notification_level: str = "all") -> bool:
    """Initialize Telegram integration"""
    global telegram_notifier, telegram_config
    
    try:
        telegram_config = TelegramConfig(
            bot_token=bot_token,
            chat_id=chat_id,
            enabled=True,
            notification_level=notification_level
        )
        telegram_notifier = TelegramNotifier(telegram_config)
        logger.info("Telegram integration initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Telegram: {e}")
        return False

async def send_trade_notification(trade_data: Dict[str, Any]) -> bool:
    """Send trade notification via Telegram"""
    if telegram_notifier:
        async with telegram_notifier:
            return await telegram_notifier.send_trade_notification(trade_data)
    return False

async def send_error_notification(error: str, context: str = "") -> bool:
    """Send error notification via Telegram"""
    if telegram_notifier:
        async with telegram_notifier:
            return await telegram_notifier.send_error_notification(error, context)
    return False

async def send_status_update(status_data: Dict[str, Any]) -> bool:
    """Send status update via Telegram"""
    if telegram_notifier:
        async with telegram_notifier:
            return await telegram_notifier.send_status_update(status_data)
    return False

async def send_daily_summary(summary_data: Dict[str, Any]) -> bool:
    """Send daily summary via Telegram"""
    if telegram_notifier:
        async with telegram_notifier:
            return await telegram_notifier.send_daily_summary(summary_data)
    return False
