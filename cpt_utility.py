from typing import Union, Tuple

import cvxpy as cp
import numpy as np


class CPTUtility:
    """
    A utility function that incorporates features from cumulative prospect theory.
    1. Prospect theory utility ('S-shaped'), parametrized by gamma_pos and gamma_neg
    2. Overweighting of extreme outcomes, parametrized by delta_pos and delta_neg
    """

    def __init__(self, gamma_pos: float, gamma_neg: float, delta_pos: float, delta_neg: float):

        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

        self.delta_pos = delta_pos
        self.delta_neg = delta_neg

        self._validate_arguments()

    @staticmethod
    def _weight(p: np.array, delta: float) -> np.array:
        assert delta >= 0.278, (
            f"[utility] weights are only strictly increasing for delta >= 0.278."
            f"{delta=} was passed."
        )
        return (p**delta) / ((p**delta + np.maximum((1 - p), 0) ** delta) ** (1 / delta))

    def cumulative_weights(self, N: int, delta: float) -> np.array:

        pi = -np.diff(self._weight(np.flip(np.cumsum(np.ones(N) / N)), delta))
        pi = np.append(pi, np.array([self._weight(1 / N, delta)]))

        # make monotone
        assert np.sum(np.diff(np.diff(pi) > 0)) <= 1, "[utility] probabilities should be unimodal."
        idx_min = np.argmin(pi)
        pi[:idx_min] = pi[idx_min]
        return pi

    def evaluate(self, weights: np.array, returns: np.array, p_weights: np.array, n_weights: np.array) -> Tuple[float, float, float]:
        portfolio_returns = returns @ weights

        pos_sort = np.sort(np.maximum(portfolio_returns, 0))
        util_p = p_weights @ self.p_util_expression(pos_sort).value

        neg_sort = np.flip(np.sort(np.minimum(portfolio_returns, 0)))
        util_n = n_weights @ self.n_util(neg_sort)

        return util_p - util_n, util_p, util_n

    def _validate_arguments(self) -> None:
        assert self.gamma_neg >= self.gamma_pos > 0, (
            f"[utility] Loss aversion implies gamma_neg >= gamma_pos. "
            f"Here: {self.gamma_neg=}, {self.gamma_pos=}."
        )
        assert self.delta_pos > 0, f"[utility] delta_pos must be positive: {self.delta_pos=}."
        assert self.delta_neg > 0, f"[utility] delta_neg must be positive: {self.delta_neg=}."

    def p_util_expression(self, portfolio_returns: Union[np.array, cp.Expression]) -> cp.Expression:
        return 1 - cp.exp(-self.gamma_pos * portfolio_returns)

    def n_util(self, portfolio_returns: np.array) -> np.array:
        return 1 - np.exp(self.gamma_neg * portfolio_returns)