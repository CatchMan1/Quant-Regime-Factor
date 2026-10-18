"""JM 数值函数；CPCV 的分组、清洗、标准化与回测步骤见 Notebook。"""

import numpy as np
from sklearn.cluster import KMeans

__all__ = ['viterbi_jump', 'fit_sjm', 'predict_test_set']


def _check_data(y, lam, lengths):
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or min(y.shape) == 0 or not np.isfinite(y).all():
        raise ValueError('特征必须是非空、有限的二维矩阵')
    if not np.isscalar(lam) or not np.isfinite(lam) or lam < 0:
        raise ValueError('lam 必须是有限的非负数')
    if lengths is None:
        lengths = np.array([len(y)])
    else:
        lengths = np.asarray(lengths)
        if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer)
                or np.any(lengths <= 0) or lengths.sum() != len(y)):
            raise ValueError('lengths 必须为正整数，且总和等于样本数')
    return y, lengths


def _check_theta(theta, n_features):
    theta = np.asarray(theta, dtype=float)
    if (theta.ndim != 2 or len(theta) == 0 or theta.shape[1] != n_features
            or not np.isfinite(theta).all()):
        raise ValueError('theta 必须是与特征列数一致的有限状态中心矩阵')
    return theta


def viterbi_jump(y, theta, lam, lengths=None):
    """固定中心后做 Viterbi 解码；每段独立开始，返回状态与累计代价。"""
    y, lengths = _check_data(y, lam, lengths)
    theta = _check_theta(theta, y.shape[1])
    n_samples, n_states = len(y), len(theta)
    loss = ((y[:, None, :] - theta[None, :, :]) ** 2).sum(axis=2)
    penalty = lam * (1 - np.eye(n_states))
    V = np.empty((n_samples, n_states))
    back = np.zeros((n_samples, n_states), dtype=int)
    states = np.empty(n_samples, dtype=int)
    columns = np.arange(n_states)

    start = 0
    for length in lengths:
        stop = start + length
        V[start] = loss[start]
        # 仅保留时间递推循环，所有前后状态的代价一起计算。
        for t in range(start + 1, stop):
            costs = V[t - 1, :, None] + penalty
            back[t] = costs.argmin(axis=0)
            V[t] = loss[t] + costs[back[t], columns]
        states[stop - 1] = V[stop - 1].argmin()
        for t in range(stop - 2, start - 1, -1):
            states[t] = back[t + 1, states[t + 1]]
        start = stop
    return states, V


def fit_sjm(y, K, lam, max_iter, lengths=None):
    """所有训练段共享一组中心；状态转移与 Viterbi 回溯均不跨段。"""
    y, lengths = _check_data(y, lam, lengths)
    if not isinstance(K, (int, np.integer)) or not 1 <= K <= len(y):
        raise ValueError('K 必须是 1 到训练样本数之间的整数')
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError('max_iter 必须是正整数')
    theta = KMeans(n_clusters=K, random_state=42).fit(y).cluster_centers_
    previous_states = None

    for iteration in range(max_iter):
        states, V_grid = viterbi_jump(y, theta, lam, lengths=lengths)
        for k in range(K):
            members = y[states == k]
            # 大惩罚可能使某状态无人归属，此时保留它上一轮的中心。
            if len(members):
                theta[k] = members.mean(axis=0)
        if previous_states is not None and np.array_equal(previous_states, states):
            break
        previous_states = states

    # 达到迭代上限时，仍让输出代价与最终中心一致。
    states, V_grid = viterbi_jump(y, theta, lam, lengths=lengths)
    print(f'JM 迭代次数：{iteration + 1}')
    return states, theta, V_grid


def predict_test_set(y_test, theta, lam, states=None, V_grid=None, lengths=None):
    """逐时点输出最小代价状态；不回溯，也不使用后续测试观测。

    默认每段从零代价开始。旧调用可同时传入训练 states、V_grid，
    仅给第一段热启动；调用方的训练代价矩阵不会被修改。
    """
    y_test, lengths = _check_data(y_test, lam, lengths)
    theta = _check_theta(theta, y_test.shape[1])
    n_states = len(theta)
    penalty = lam * (1 - np.eye(n_states))
    warm_cost = None
    if (states is None) != (V_grid is None):
        raise ValueError('热启动须同时提供 states 和 V_grid')
    if states is not None:
        states = np.asarray(states)
        V_grid = np.asarray(V_grid, dtype=float)
        if (states.ndim != 1 or len(states) == 0
                or not np.issubdtype(states.dtype, np.integer)
                or np.any((states < 0) | (states >= n_states))
                or V_grid.shape != (len(states), n_states)
                or not np.isfinite(V_grid).all()):
            raise ValueError('训练 states 或 V_grid 无效')
        warm_cost = V_grid[-1].copy()
        warm_cost[states[-1]] = 0.0  # 保留旧 Notebook 的热启动规则。

    test_states = np.empty(len(y_test), dtype=int)
    start = 0
    for length in lengths:
        stop = start + length
        v_prev = warm_cost if start == 0 and warm_cost is not None else np.zeros(n_states)
        for t in range(start, stop):
            loss = ((y_test[t] - theta) ** 2).sum(axis=1)
            v_prev = loss + (v_prev[:, None] + penalty).min(axis=0)
            test_states[t] = v_prev.argmin()
        start = stop
    return test_states
