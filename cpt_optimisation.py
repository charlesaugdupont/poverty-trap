from __future__ import annotations

import abc
import copy
import cvxpy as cp
import numpy as np
from abc import abstractmethod
from typing import Union, Tuple
from cpt_optimisation import CPTUtility
from scipy.optimize import minimize


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


class CPTOptimizer(abc.ABC):
    def __init__(self, utility: CPTUtility, max_iter: int = 1000):
        utility = copy.deepcopy(utility)
        self.utility = utility
        self.max_iter = max_iter
        self._weights = None
        self._weights_history = None
        self._wall_time = None

    @abstractmethod
    def optimize(self, r: np.array, verbose: bool = False) -> None:
        pass

    @property
    def weights(self) -> np.array:
        assert self._weights is not None
        return self._weights

    @property
    def weights_history(self) -> np.array:
        assert self._weights_history is not None
        return self._weights_history

    @property
    def wall_time(self) -> np.array:
        assert self._wall_time is not None
        return self._wall_time - np.min(self._wall_time)


class MeanVarianceFrontierOptimizer(CPTOptimizer):

    def func(self, x, w, var_target, problem, returns, p_weights, n_weights) -> float:
        var_target.value = x[0]**2
        problem.solve()
        return - self.utility.evaluate(w.value, returns, p_weights, n_weights)[0]


    def optimize(self, returns: np.array) -> None:
        mu = np.mean(returns, axis=0)
        Sigma = np.cov(returns, rowvar=False)

        N = len(returns)
        p_weights = self.utility.cumulative_weights(N, delta=self.utility.delta_pos)
        n_weights = self.utility.cumulative_weights(N, delta=self.utility.delta_neg)

        min_vol = np.sqrt(self._get_min_variance(Sigma))
        max_vol = np.sqrt(self._get_max_return_variance(mu))
        delta = max_vol - min_vol

        var_target = cp.Parameter()
        w = cp.Variable(len(mu), nonneg=True)
        objective = cp.Maximize(w @ mu)
        constraints = [cp.sum(w) == 1, cp.quad_form(w, Sigma) <= var_target]
        problem = cp.Problem(objective, constraints)

        util = minimize(self.func, 
                        x0=(min_vol + delta*0.15), 
                        args=(w, var_target, problem, returns, p_weights, n_weights), 
                        method="SLSQP", 
                        bounds=((min_vol, max_vol),))
        
        var_target.value = util.x[0]
        problem.solve()
        self._weights = w.value
        self._best_util = -util.fun


    @staticmethod
    def _get_min_variance(Sigma: np.array) -> np.float:
        w = cp.Variable(Sigma.shape[0], nonneg=True)
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        problem = cp.Problem(objective, [cp.sum(w) == 1])
        problem.solve()
        return max(objective.value, 0)

    @staticmethod
    def _get_max_return_variance(mu: np.array) -> np.float:
        w = cp.Variable(len(mu), nonneg=True)
        objective = cp.Maximize(w @ mu)
        problem = cp.Problem(objective, [cp.sum(w) == 1])
        problem.solve()
        return objective.value