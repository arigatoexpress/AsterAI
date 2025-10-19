"""
Paper test trade runner (dry-run)

Simulates a ~$5 BTCUSDT trade end-to-end without real API keys by using a
mock Aster client. Exercises market data fetch, signal generation, position
creation, and status reporting in the LiveTradingAgent with dry_run enabled.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

from live_trading_agent import LiveTradingAgent, TradingConfig


class MockAsterClient:
	"""Minimal async mock of AsterClient suitable for dry-run flow."""

	def __init__(self, price: float = 50000.0):
		self._price = float(price)
		self._wallet = 100.0

	async def get_ticker(self, symbol: str) -> Dict[str, Any]:
		# Small random walk around base price
		self._price *= (1.0 + random.uniform(-0.0008, 0.0008))
		return {"symbol": symbol, "lastPrice": f"{self._price:.2f}"}

	async def get_order_book(self, symbol: str) -> Dict[str, Any]:
		best_bid = self._price * 0.999
		best_ask = self._price * 1.001
		bids = [[f"{best_bid - i*5:.2f}", f"{1.0 + i*0.1:.6f}"] for i in range(10)]
		asks = [[f"{best_ask + i*5:.2f}", f"{1.0 + i*0.1:.6f}"] for i in range(10)]
		return {"lastUpdateId": int(datetime.now().timestamp()), "bids": bids, "asks": asks}

	async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[List[Any]]:
		# Generate synthetic 1m klines around price
		now = datetime.utcnow()
		base = self._price
		kl = []
		for i in range(limit):
			open_t = int((now - timedelta(minutes=limit - i)).timestamp() * 1000)
			opn = base * (1.0 + random.uniform(-0.002, 0.002))
			high = opn * (1.0 + random.uniform(0.0, 0.003))
			low = opn * (1.0 - random.uniform(0.0, 0.003))
			cls = random.choice([low, opn, high])
			vol = random.uniform(5.0, 50.0)
			kl.append([open_t, f"{opn:.2f}", f"{high:.2f}", f"{low:.2f}", f"{cls:.2f}", f"{vol:.6f}", open_t+60000, "0", 100, "0", "0", "0"])
		return kl

	async def place_order(self, symbol: str, side: str, type: str, quantity: float) -> Dict[str, Any]:
		# No real side effects in dry-run; emulate instant fill
		return {"symbol": symbol, "orderId": f"DRYRUN-{random.randint(100000,999999)}", "status": "FILLED", "side": side, "type": type, "executedQty": quantity}

	async def get_account_info(self) -> Dict[str, Any]:
		return {"totalWalletBalance": f"{self._wallet:.2f}"}


async def run_once():
	config = TradingConfig(
		initial_capital=100.0,
		max_leverage=3.0,
		position_size_pct=0.05,  # slightly higher for pullback aggressiveness
		stop_loss_pct=0.02,
		take_profit_pct=0.04,
		daily_loss_limit_pct=0.10,
		max_positions=1,
		trading_pairs=["BTCUSDT"],
		dry_run=True,
	)

	client = MockAsterClient(price=50000.0)
	agent = LiveTradingAgent(config, client)  # type: ignore[arg-type]

	# Single iteration of the internal pipeline (no infinite loop)
	market_data = await agent._update_market_data()
	signals = await agent._generate_signals(market_data)
	await agent._execute_trades(signals)
	await agent._update_positions()
	await agent._update_metrics()
	status = agent.get_status()
	print({
		"positions": status["positions"],
		"metrics": status["metrics"],
		"config": status["config"],
	})


def main():
	asyncio.run(run_once())


if __name__ == "__main__":
	main()


