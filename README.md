# MCP AI Trading Protocol (Aster Perps DEX)

Modular Python stack for AI-orchestrated trading research and execution on perps DEX venues.

## Quickstart

1) Create venv
```bash
python3 -m venv .venv && source .venv/bin/activate
```
2) Install deps
```bash
pip install -r requirements.txt
```
3) Run demo backtest
```bash
python scripts/run_backtest.py
```
4) Launch dashboard
```bash
streamlit run dashboard/app.py
```

## Env
Copy `.env.example` to `.env` and set keys when wiring live trading.

## Structure
```
mcp_trader/ ... core libs
agent/ ... live agent
scripts/ ... utilities
```
