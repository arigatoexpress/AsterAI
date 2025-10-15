from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from mcp_trader.backtesting.vectorized_backtester import evaluate_positions
from mcp_trader.strategies.rules import generate_positions_sma_crossover


@dataclass
class Chromosome:
    short_win: int
    long_win: int


def random_chromosome(short_range=(5, 50), long_range=(20, 200)) -> Chromosome:
    s = random.randint(*short_range)
    l = random.randint(max(s + 1, long_range[0]), long_range[1])
    return Chromosome(short_win=s, long_win=l)


def mutate(ch: Chromosome, rate: float = 0.1, short_range=(5, 50), long_range=(20, 200)) -> Chromosome:
    s, l = ch.short_win, ch.long_win
    if random.random() < rate:
        s = random.randint(*short_range)
    if random.random() < rate:
        l = random.randint(max(s + 1, long_range[0]), long_range[1])
    return Chromosome(short_win=s, long_win=l)


def crossover(a: Chromosome, b: Chromosome) -> Chromosome:
    return Chromosome(short_win=random.choice([a.short_win, b.short_win]), long_win=random.choice([a.long_win, b.long_win]))


def fitness(close: pd.Series, ch: Chromosome, fee_bps: float = 1.0) -> float:
    pos = generate_positions_sma_crossover(pd.DataFrame({"close": close}), ch.short_win, ch.long_win)
    res = evaluate_positions(close, pos, fee_bps=fee_bps)
    m = res["metrics"]
    score = (m["sharpe"] * 0.5) + (m["calmar"] * 0.3) + (m["profit_factor"] * 0.2)
    return float(score)


def run_ga(
    close: pd.Series,
    population_size: int = 40,
    generations: int = 20,
    mutation_rate: float = 0.15,
    elite_k: int = 4,
) -> tuple[Chromosome, float]:
    population = [random_chromosome() for _ in range(population_size)]

    best_ch, best_fit = None, -np.inf
    for _ in range(generations):
        scored = [(ch, fitness(close, ch)) for ch in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        if scored[0][1] > best_fit:
            best_ch, best_fit = scored[0]

        elites = [ch for ch, _ in scored[:elite_k]]
        new_pop = elites.copy()
        while len(new_pop) < population_size:
            parents = random.sample(elites, 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child, rate=mutation_rate)
            new_pop.append(child)
        population = new_pop

    return best_ch, best_fit

