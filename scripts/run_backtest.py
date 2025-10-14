import numpy as np
import pandas as pd

from mcp_trader.strategies.rules import generate_positions_sma_crossover
from mcp_trader.backtesting.vectorized_backtester import evaluate_positions
from mcp_trader.ga.engine import run_ga


def main():
    np.random.seed(7)
    n = 1500
    returns = np.random.normal(0.0002, 0.018, size=n)
    close = pd.Series((1 + pd.Series(returns)).cumprod() * 1000.0, name="close")

    best, score = run_ga(close, population_size=30, generations=15)
    print(f"Best SMA params: short={best.short_win}, long={best.long_win}, score={score:.3f}")

    pos = generate_positions_sma_crossover(pd.DataFrame({"close": close}), best.short_win, best.long_win)
    res = evaluate_positions(close, pos, fee_bps=1)
    print("Metrics:")
    for k, v in res["metrics"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
