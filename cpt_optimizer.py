from __future__ import annotations

import abc
import copy
from abc import abstractmethod
from typing import Tuple, Optional, Union

import cvxpy as cp
import numpy as np

from cptopt.utility import CPTUtility

from scipy.optimize import minimize


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