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
        
        emoji = "[BUY]" if side.upper() == "BUY" else "[SELL]"
        pnl_emoji = "[PROFIT]" if pnl > 0 else "[LOSS]" if pnl < 0 else "[FLAT]"
        
        message = f"""
{emoji} *TRADE EXECUTED* {emoji}

[DATA] *Symbol:* {symbol}
[CHART] *Side:* {side}
[QTY] *Quantity:* {qty}
[PRICE] *Price:* ${price:,.4f}
{pnl_emoji} *PnL:* ${pnl:,.2f}
[TIME] *Time:* {timestamp}

[BOT] *Aster Trading Bot*
        """.strip()
        
        return await self.send_message(message)
    
    async def send_error_notification(self, error: str, context: str = "") -> bool:
        """Send error notification"""
        if self.config.notification_level not in ["all", "errors_only"]:
            return False
            
        message = f"""
[ERROR] *BOT ERROR* [ERROR]

[X] *Error:* {error}
[INFO] *Context:* {context}
[TIME] *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

[BOT] *Aster Trading Bot*
        """.strip()
        
        return await self.send_message(message)
    
    async def send_status_update(self, status_data: Dict[str, Any]) -> bool:
        """Send comprehensive bot status update"""
        if self.config.notification_level not in ["all"]:
            return False

        running = status_data.get("running", False)
        status_emoji = "[ACTIVE]" if running else "[STOPPED]"
        
        message = f"""
{status_emoji} *BOT STATUS UPDATE* {status_emoji}

[BOT] *Status:* {"[ACTIVE] Running" if running else "[STOPPED] Stopped"}
[MONEY] *Capital:* ${status_data.get("balance", 0):.2f}
[DATA] *Available:* ${status_data.get("available_balance", 0):.2f}
[TARGET] *Positions:* {status_data.get("positions", 0)}
[CHART] *Total Trades:* {status_data.get("total_trades", 0)}
[TARGET] *Win Rate:* {status_data.get("win_rate", 0):.1%}
[MONEY] *P&L:* ${status_data.get("total_pnl", 0):.2f}

[TIME] *Last Updated:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """.strip()

        return await self.send_message(message)

    async def send_trade_alert(self, trade_data: Dict[str, Any]) -> bool:
        """Send trade execution alert"""
        if self.config.notification_level not in ["all", "trades_only"]:
            return False

        symbol = trade_data.get("symbol", "UNKNOWN")
        side = trade_data.get("side", "UNKNOWN")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)

        emoji = "[UP]" if side.upper() == "BUY" else "[DOWN]"

        message = f"""
{emoji} *TRADE ALERT* {emoji}

[TARGET] *Symbol:* {symbol}
[DATA] *Side:* {side.upper()}
[NUM] *Quantity:* {quantity}
[MONEY] *Price:* ${price:.2f}
[VALUE] *Value:* ${quantity * price:.2f}

[TIME] *Time:* {datetime.now().strftime("%H:%M:%S")}
        """.strip()

        return await self.send_message(message)

    async def send_error_alert(self, error_data: Dict[str, Any]) -> bool:
        """Send error alert"""
        if self.config.notification_level not in ["all", "errors_only"]:
            return False

        error_type = error_data.get("type", "Unknown")
        message_text = error_data.get("message", "No details")

        message = f"""
[ERROR] *ERROR ALERT* [ERROR]

[WARNING] *Type:* {error_type}
[NOTE] *Message:* {message_text}
[TIME] *Time:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """.strip()

        return await self.send_message(message)

    async def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send daily trading summary"""
        total_trades = summary_data.get("total_trades", 0)
        total_pnl = summary_data.get("total_pnl", 0)
        win_rate = summary_data.get("win_rate", 0)
        
        pnl_emoji = "[PROFIT]" if total_pnl > 0 else "[LOSS]" if total_pnl < 0 else "[FLAT]"
        
        message = f"""
[DATA] *DAILY TRADING SUMMARY* [DATA]

[CHART] *Total Trades:* {total_trades}
{pnl_emoji} *Total PnL:* ${total_pnl:,.2f}
[TARGET] *Win Rate:* {win_rate:.1f}%
[DATE] *Date:* {datetime.now().strftime('%Y-%m-%d')}

[BOT] *Aster Trading Bot*
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
            elif command == "/portfolio":
                return await self._handle_portfolio()
            elif command == "/positions":
                return await self._handle_positions()
            elif command == "/metrics":
                return await self._handle_metrics()
            elif command == "/start_trading":
                return await self._handle_start_trading()
            elif command == "/stop_trading":
                return await self._handle_stop_trading()
            elif command == "/restart":
                await self._handle_stop_trading()
                await asyncio.sleep(2)
                return await self._handle_start_trading()
            elif command == "/emergency_stop":
                return await self._handle_stop_trading()
            elif command == "/balance":
                return await self._handle_portfolio()
            elif command == "/architecture":
                return await self._handle_architecture()
            elif command.startswith("/rebalance"):
                return await self._handle_rebalance()
            elif command.startswith("/leverage"):
                parts = command.split()
                leverage = float(parts[1]) if len(parts) > 1 else None
                return await self._handle_leverage(leverage)
            elif command == "/emergency_close_all":
                return await self._handle_emergency_close_all()
            elif command == "/system_status":
                return await self._handle_system_status()
            elif command.startswith("/logs"):
                parts = command.split()
                lines = int(parts[1]) if len(parts) > 1 else 20
                return await self._handle_logs(lines)
            elif command == "/help":
                return await self._handle_help()
            else:
                return "[ERROR] Unknown command. Use /help for available commands."
                
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
            return f"[ERROR] Error processing command: {str(e)}"
    
    async def _handle_start(self) -> str:
        return """
[BOT] *Welcome to Aster Trading Bot!*

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
                    status_emoji = "[ACTIVE]" if running else "[STOPPED]"
                    
                    return f"""
{status_emoji} *Bot Status: {"Running" if running else "Stopped"}*

[DATA] *Details:*
{json.dumps(data, indent=2)}
                    """.strip()
                else:
                    return "[ERROR] Failed to get bot status"
        except Exception as e:
            return f"[ERROR] Error getting status: {str(e)}"
    
    async def _handle_start_trading(self) -> str:
        try:
            async with self.session.post(f"{self.trading_agent_url}/start") as response:
                if response.status == 200:
                    data = await response.json()
                    return f"[OK] {data.get('message', 'Trading started successfully')}"
                else:
                    error_data = await response.json()
                    return f"[ERROR] Failed to start trading: {error_data.get('detail', 'Unknown error')}"
        except Exception as e:
            return f"[ERROR] Error starting trading: {str(e)}"
    
    async def _handle_stop_trading(self) -> str:
        try:
            async with self.session.post(f"{self.trading_agent_url}/stop") as response:
                if response.status == 200:
                    data = await response.json()
                    return f"[OK] {data.get('message', 'Trading stopped successfully')}"
                else:
                    error_data = await response.json()
                    return f"[ERROR] Failed to stop trading: {error_data.get('detail', 'Unknown error')}"
        except Exception as e:
            return f"[ERROR] Error stopping trading: {str(e)}"
    
    async def _handle_help(self) -> str:
        return """
[BOT] *Aster Trading Bot Commands*

*MONITORING:*
/status - Check bot status and metrics
/portfolio - View portfolio balance
/positions - View open positions
/metrics - Trading performance metrics
/system_status - Cloud + local status
/logs [N] - Recent log entries (default 20)

*CONTROL:*
/start_trading - Start trading
/stop_trading - Stop trading
/restart - Restart bot
/emergency_stop - Emergency shutdown
/emergency_close_all - Close all positions NOW

*CONFIGURATION:*
/leverage [X] - Set/check leverage (max 2x)
/rebalance - Rebalance portfolio

*INFO:*
/architecture - System architecture
/balance - Account balance
/help - Show this help

[DATA] *Features:*
* Real-time trade notifications
* Full remote control
* Emergency controls
* Conservative risk limits
* Hybrid cloud/local deployment

[WARNING] *Safety:* Max 2x leverage, 5% daily loss limit
        """.strip()

    async def _handle_portfolio(self) -> str:
        """Handle portfolio command"""
        try:
            async with self.session.get(f"{self.trading_agent_url}/portfolio") as response:
                if response.status == 200:
                    data = await response.json()
                    balance = data.get("balance", 0)
                    equity = data.get("equity", 0)
                    margin_used = data.get("margin_used", 0)

                    return f"""
[MONEY] *PORTFOLIO STATUS* [MONEY]

[VALUE] *Balance:* ${balance:.2f}
[CHART] *Equity:* ${equity:.2f}
[DATA] *Margin Used:* ${margin_used:.2f}
                    """.strip()
                else:
                    return "[ERROR] Failed to retrieve portfolio data"
        except Exception as e:
            return f"[ERROR] Error getting portfolio: {str(e)}"

    async def _handle_positions(self) -> str:
        """Handle positions command"""
        try:
            async with self.session.get(f"{self.trading_agent_url}/positions") as response:
                if response.status == 200:
                    data = await response.json()
                    positions = data.get("positions", [])

                    if not positions:
                        return "[DATA] No open positions"

                    message = "[TARGET] *OPEN POSITIONS* [TARGET]\n\n"
                    for pos in positions[:10]:  # Limit to 10 positions
                        message += f"""
[CHART] *{pos.get('symbol', 'N/A')}*
[MONEY] *Entry:* ${pos.get('entry_price', 0):.2f}
[NUM] *Quantity:* {pos.get('quantity', 0)}
[DATA] *Side:* {pos.get('side', 'N/A').upper()}
                        """

                    return message.strip()
                else:
                    return "[ERROR] Failed to retrieve positions data"
        except Exception as e:
            return f"[ERROR] Error getting positions: {str(e)}"

    async def _handle_metrics(self) -> str:
        """Handle metrics command"""
        try:
            async with self.session.get(f"{self.trading_agent_url}/metrics") as response:
                if response.status == 200:
                    data = await response.json()
                    latest = data.get("latest", {})

                    return f"""
[DATA] *TRADING METRICS* [DATA]

[TARGET] *Total Trades:* {latest.get('total_trades', 0)}
[MONEY] *Total P&L:* ${latest.get('total_pnl', 0):.2f}
[CHART] *Win Rate:* {latest.get('win_rate', 0):.1%}
[NUM] *Active Positions:* {latest.get('active_positions', 0)}
                    """.strip()
                else:
                    return "[ERROR] Failed to retrieve metrics data"
        except Exception as e:
            return f"[ERROR] Error getting metrics: {str(e)}"

    async def _handle_architecture(self) -> str:
        """Handle architecture command"""
        try:
            async with self.session.get(f"{self.trading_agent_url}/system/summary") as response:
                if response.status == 200:
                    data = await response.json()

                    message = "[ARCH] *SYSTEM ARCHITECTURE* [ARCH]\n\n"

                    # Services status
                    message += "*SERVICES:*\n"
                    for service, info in data.get("services", {}).items():
                        status = "[OK]" if info.get("status") == "healthy" else "[ERROR]"
                        message += f"{status} {service.replace('_', ' ').title()}: {info.get('status', 'unknown')}\n"

                    message += "\n*SYSTEM METRICS:*\n"
                    for metric, value in data.get("system_metrics", {}).items():
                        message += f"[DATA] {metric.replace('_', ' ').title()}: {value}\n"

                    return message.strip()
                else:
                    return "[ERROR] Failed to retrieve architecture data"
        except Exception as e:
            return f"[ERROR] Error getting architecture: {str(e)}"
    
    async def _handle_rebalance(self) -> str:
        """Rebalance portfolio based on current market conditions"""
        try:
            async with self.session.post(f"{self.trading_agent_url}/rebalance") as response:
                if response.status == 200:
                    data = await response.json()
                    return f"[OK] Portfolio rebalanced\n{data.get('details', '')}"
                return "[ERROR] Rebalance failed"
        except Exception as e:
            return f"[ERROR] {str(e)}"
    
    async def _handle_leverage(self, leverage: Optional[float]) -> str:
        """Adjust trading leverage"""
        if leverage is None:
            # Get current leverage
            try:
                async with self.session.get(f"{self.trading_agent_url}/config") as response:
                    if response.status == 200:
                        data = await response.json()
                        return f"Current leverage: {data.get('max_leverage', 'N/A')}x"
            except Exception as e:
                return f"[ERROR] {str(e)}"
        
        # Set new leverage (cap at 2x for conservative mode)
        leverage = min(max(leverage, 1.0), 2.0)
        payload = {"max_leverage": leverage}
        try:
            async with self.session.post(f"{self.trading_agent_url}/config/leverage", json=payload) as response:
                if response.status == 200:
                    return f"[OK] Leverage set to {leverage}x (capped at 2x for safety)"
                return "[ERROR] Failed to update leverage"
        except Exception as e:
            return f"[ERROR] {str(e)}"
    
    async def _handle_emergency_close_all(self) -> str:
        """Emergency close all positions"""
        try:
            async with self.session.post(f"{self.trading_agent_url}/emergency/close-all") as response:
                if response.status == 200:
                    data = await response.json()
                    closed = data.get('positions_closed', 0)
                    return f"[WARNING] EMERGENCY STOP\n{closed} positions closed"
                return "[ERROR] Emergency close failed"
        except Exception as e:
            return f"[ERROR] Emergency close error: {str(e)}"
    
    async def _handle_system_status(self) -> str:
        """Get comprehensive system status (cloud + local)"""
        try:
            # Cloud status
            async with self.session.get(f"{self.trading_agent_url}/system/summary") as response:
                if response.status == 200:
                    cloud_data = await response.json()
                else:
                    cloud_data = {}
            
            # Local status (if accessible)
            local_status = "Local PC: [OFFLINE]"
            try:
                async with self.session.get("http://localhost:8081/api/control/status", timeout=2) as response:
                    if response.status == 200:
                        local_data = await response.json()
                        local_status = f"Local PC: [ACTIVE] Trading: {local_data.get('trading_active', False)}"
            except:
                pass
            
            return f"""
[SYSTEM STATUS]

Cloud: {cloud_data.get('services', {}).get('trading_backend', {}).get('status', 'unknown')}
{local_status}

Balance: ${cloud_data.get('trading_metrics', {}).get('current_balance', 0):.2f}
Active Positions: {cloud_data.get('trading_metrics', {}).get('total_positions', 0)}
Win Rate: {cloud_data.get('trading_metrics', {}).get('win_rate', 0):.1%}
            """.strip()
        except Exception as e:
            return f"[ERROR] System status error: {str(e)}"
    
    async def _handle_logs(self, lines: int = 20) -> str:
        """Get recent log entries"""
        try:
            async with self.session.get(f"{self.trading_agent_url}/system/logs?lines={lines}") as response:
                if response.status == 200:
                    data = await response.json()
                    logs = data.get('logs', [])
                    message = "[RECENT LOGS]\n\n"
                    for log in logs[-10:]:  # Last 10 for Telegram limit
                        level = log.get('level', 'INFO')
                        msg = log.get('message', '')
                        timestamp = log.get('timestamp', '')
                        message += f"{timestamp} [{level}] {msg}\n"
                    return message.strip()
                return "[ERROR] Failed to retrieve logs"
        except Exception as e:
            return f"[ERROR] {str(e)}"

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
