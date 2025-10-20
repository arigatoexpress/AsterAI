"""
Telegram Webhook Server for Remote Bot Control
Receives commands from Telegram and executes them on trading system
"""

import os
import asyncio
from fastapi import FastAPI, Request, HTTPException
from telegram_bot import TelegramCommandHandler
import logging
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Telegram Webhook Server", version="1.0.0")

# Initialize command handler
TRADING_AGENT_URL = os.getenv("TRADING_AGENT_URL", "https://aster-self-learning-trader-880429861698.us-central1.run.app")
command_handler = None

@app.on_event("startup")
async def startup():
    """Initialize command handler on startup"""
    global command_handler
    command_handler = TelegramCommandHandler(TRADING_AGENT_URL)
    await command_handler.__aenter__()
    logger.info(f"Telegram webhook server started, connected to {TRADING_AGENT_URL}")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global command_handler
    if command_handler:
        await command_handler.__aexit__(None, None, None)

@app.post("/webhook/{bot_token}")
async def telegram_webhook(bot_token: str, request: Request):
    """Handle incoming Telegram webhook"""
    # Verify bot token
    expected_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not expected_token or bot_token != expected_token:
        logger.warning(f"Invalid bot token attempt: {bot_token[:10]}...")
        raise HTTPException(status_code=403, detail="Invalid bot token")
    
    data = await request.json()
    logger.info(f"Received webhook: {data}")
    
    # Extract message
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")
    
    # Verify chat ID
    expected_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if expected_chat_id and str(chat_id) != str(expected_chat_id):
        logger.warning(f"Unauthorized chat ID: {chat_id}")
        return {"ok": True}  # Silently ignore
    
    if not text.startswith("/"):
        return {"ok": True}
    
    # Process command
    try:
        response_text = await command_handler.handle_command(text, str(chat_id))
        
        # Send response back via Telegram API
        await send_telegram_message(chat_id, response_text)
        
        return {"ok": True}
    except Exception as e:
        logger.error(f"Error processing command: {e}")
        await send_telegram_message(chat_id, f"[ERROR] Failed to process command: {str(e)}")
        return {"ok": False, "error": str(e)}

async def send_telegram_message(chat_id: str, text: str):
    """Send message via Telegram API"""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    # Split long messages (Telegram limit is 4096 characters)
    max_length = 4000
    if len(text) > max_length:
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    else:
        chunks = [text]
    
    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            try:
                async with session.post(url, json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "Markdown"
                }) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send message: {await response.text()}")
            except Exception as e:
                logger.error(f"Error sending message: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "telegram-webhook",
        "handler_initialized": command_handler is not None
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Telegram Webhook Server",
        "version": "1.0.0",
        "status": "running",
        "trading_agent_url": TRADING_AGENT_URL
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Cloud Run sets PORT)
    port = int(os.environ.get("PORT", 8002))
    
    logger.info(f"Starting Telegram Webhook Server on port {port}")
    logger.info(f"Connected to trading agent: {TRADING_AGENT_URL}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

