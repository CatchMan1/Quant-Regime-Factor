r"""
Combinatorial Purged Cross-Validation (CPCV) for time series.

Model-agnostic: any object with ``fit(X)`` (or ``fit(X, y)``) and ``predict(X)``
works. The series is split into ``N`` non-overlapping groups; each split holds
out ``k`` groups as test, every contiguous training block is purged + embargoed
and standardised on its own, and ``phi = C(N-1, k-1)`` complete backtest paths
are returned. ``N`` must be a multiple of ``k``.

Usage
-----
::

    from cpcv import run_cpcv

    result = run_cpcv(
        df,                                # DataFrame with time / features / label columns
        feature_cols=['f1', 'f2'],
        model_factory=lambda: MyModel(),   # fit(X) or fit(X, y); predict(X) -> labels
        n_groups=6, k_test_groups=2,
        window_size=200,                   # rolling window used for standardisation
        time_col='DateTime',
    )
    paths = result['backtest_paths']        # {path_id: DataFrame}

Common options
--------------
=================================  ===============================================
supervised model                   target_col='y', label_horizon_ticks=<horizon>,
                                   rank_states=False
scaling                            standardize='rolling' (default) | 'global' | 'none'
state ranking                      return_col='pct' (default) or another column;
                                   rank_states=False to keep raw labels
custom groups                      group_col='g' (integer group ids 0..N-1)
purging / embargo                  purge_ticks=..., embargo_ticks=...  (in rows)
=================================  ===============================================

Or drive the class directly::

    cpcv = CPCVGrouped(n_groups=6, k_test_groups=2, window_size=200)
    result = cpcv.run(df, feature_cols, model_factory, time_col, prediction_col)
    paths = cpcv.build_backtest_paths(result)

Feature engineering is intentionally not part of this module: prepare the
dataframe in the calling code and pass it in. Optional model helpers
(``HMM_fit`` and the Jump Model: ``fit_sjm`` / ``predict_test_set`` /
``SJMRegimeModel``) are included for the example notebooks; the CPCV class
itself does not depend on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover
    KMeans = None

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover
    GaussianHMM = None


# dataclaass装饰器自动为类生成init、repr、eq方法，存储数据
@dataclass(frozen=True)
class GroupRange:
    group_id: int
    start_pos: int
    end_pos: int


@dataclass(frozen=True)
class SplitDefinition:
    split_id: int
    train_groups: Tuple[int, ...]
    test_groups: Tuple[int, ...]


class CPCVGrouped:
    r"""
    Combinatorial Purged Cross-Validation over ordered time-series groups.

    Parameters
    ----------
    n_groups, k_test_groups : int
        Split the series into ``n_groups`` blocks and hold out ``k_test_groups``
        per split.  ``phi = C(n_groups-1, k_test_groups-1)``.  ``n_groups`` must
        be a multiple of ``k_test_groups``.
    window_size : int, optional
        Rolling window length used when ``standardize='rolling'``.
    purge_ticks, embargo_ticks, label_horizon_ticks : int
        Purge overlap, post-test embargo and label horizon, all in ticks (rows).
    min_train_samples : int, optional
        Minimum number of usable training rows per split.
    return_col : str or None
        Column used to rank unsupervised states by in-sample return
        (default ``'pct'``). Required only when ``rank_states=True``.
    rank_states : bool
        Re-rank raw model labels so that rank 0 = worst in-sample return.
        Set ``False`` for supervised models.
    target_col : str or None
        Supervised label column. When given, ``model.fit(X, y)`` is used.
    standardize : {'rolling', 'global', 'none'}
        How training features are scaled. Test features always use training
        statistics only.
    group_col : str or None
        Column with explicit integer group ids (0..n_groups-1). When ``None``
        the series is cut into equal-length contiguous blocks.

    Notes
    -----
    All purging / embargoing / standardisation is done per contiguous training
    block, so no window ever crosses a block or a purged gap.
    """

    def __init__(self, n_groups, k_test_groups, window_size=None, purge_ticks=0,
                 embargo_ticks=0, label_horizon_ticks=1, min_train_samples=None, *,
                 return_col='pct', rank_states=True, target_col=None,
                 standardize='rolling', group_col=None):
        if standardize not in ('rolling', 'global', 'none'):
            raise ValueError("standardize must be 'rolling', 'global' or 'none'")
        if standardize == 'rolling' and not window_size:
            raise ValueError("window_size is required when standardize='rolling'")
        if rank_states and not return_col:
            raise ValueError("return_col is required when rank_states=True")
        if k_test_groups >= n_groups or n_groups % k_test_groups != 0:
            raise ValueError(
                f"n_groups ({n_groups}) must be a multiple of k_test_groups ({k_test_groups}) "
                "so that every CPCV path covers all groups exactly once"
            )

        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        self.window_size = window_size
        self.purge_ticks = purge_ticks
        self.embargo_ticks = embargo_ticks
        self.label_horizon_ticks = label_horizon_ticks
        self.min_train_samples = min_train_samples if min_train_samples is not None else (window_size or 1)
        self.return_col = return_col
        self.rank_states = rank_states
        self.target_col = target_col
        self.standardize = standardize
        self.group_col = group_col
        self.phi = math.comb(n_groups - 1, k_test_groups - 1)  # 任意一个测试组出现在不同组合中的数量
        self.time_col_name = None

    # ------------------------------------------------------------------ data
    def _prepare_dataframe(self, df, time_col) -> pd.DataFrame:
        data = df.copy()
        data[time_col] = pd.to_datetime(data[time_col])
        data = data.sort_values(time_col).reset_index(drop=True)

        self.time_col_name = time_col
        data['_pos'] = np.arange(len(data), dtype=int)
        data['_event_end_pos'] = np.minimum(
            data['_pos'].values + self.label_horizon_ticks - 1,
            len(data) - 1,
        )
        if self.group_col is not None:
            data['_group'] = data[self.group_col].astype(int)
            found = data['_group'].nunique()
            if found != self.n_groups:
                raise ValueError(
                    f"group_col '{self.group_col}' has {found} groups, expected n_groups={self.n_groups}"
                )
        else:
            data['_group'] = self._split_groups(len(data))
        return data.copy()

    def _split_groups(self, n_samples) -> np.ndarray:
        base = n_samples // self.n_groups
        groups = np.empty(n_samples, dtype=int)
        cursor = 0
        for gid in range(self.n_groups - 1):
            groups[cursor:cursor + base] = gid
            cursor += base
        groups[cursor:] = self.n_groups - 1  # 每个样本属于哪个组N
        return groups

    def _group_ranges(self, df) -> Dict[int, GroupRange]:
        ranges: Dict[int, GroupRange] = {}
        for gid, grp in df.groupby('_group', sort=True):
            ranges[int(gid)] = GroupRange(int(gid), int(grp['_pos'].min()), int(grp['_pos'].max()))
        return ranges

    def generate_splits(self) -> List[SplitDefinition]:
        splits: List[SplitDefinition] = []
        group_ids = tuple(range(self.n_groups))
        for split_id, test_groups in enumerate(combinations(group_ids, self.k_test_groups)):
            train_groups = tuple(g for g in group_ids if g not in test_groups)
            splits.append(SplitDefinition(split_id, train_groups, tuple(test_groups)))
        return splits

    def _contiguous_train_blocks(self, train_groups: Sequence[int]) -> List[Tuple[int, ...]]:
        if not train_groups:
            return []
        # 第一个组作为第一段的开头
        blocks: List[List[int]] = [[train_groups[0]]]
        for gid in train_groups[1:]:
            if gid == blocks[-1][-1] + 1:  # 连续
                blocks[-1].append(gid)
            else:                          # 不连续 -> 新开一段
                blocks.append([gid])
        return [tuple(b) for b in blocks]

    # ------------------------------------------------------------- cleaning
    def _purge_embargo_block(
        self,
        block_df: pd.DataFrame,
        test_intervals: Sequence[Tuple[int, int]],
    ) -> pd.DataFrame:
        """
        对单个 train block 单独清洗。
        purge 条件：训练样本事件区间与任一测试区间重叠，或距离测试区间不足 purge_ticks。
        embargo 条件：测试区间结束后 embargo_ticks 内的样本删除。
        """
        keep_mask = np.ones(len(block_df), dtype=bool)
        start = block_df['_pos'].to_numpy()
        end = block_df['_event_end_pos'].to_numpy()

        for t_start, t_end in test_intervals:
            overlap = (start <= t_end + self.purge_ticks) & (end >= t_start - self.purge_ticks)
            embargo = (start > t_end) & (start <= t_end + self.embargo_ticks)
            keep_mask &= ~(overlap | embargo)

        cleaned = block_df.loc[keep_mask].copy()
        return cleaned.reset_index(drop=True)

    def _split_on_gaps(self, cleaned_block_df: pd.DataFrame) -> List[pd.DataFrame]:
        if cleaned_block_df.empty:
            return []
        pos = cleaned_block_df['_pos'].to_numpy()
        break_points = np.where(np.diff(pos) > 1)[0] + 1
        segments = np.split(cleaned_block_df, break_points)
        return [seg.reset_index(drop=True) for seg in segments if len(seg) > 0]

    # ---------------------------------------------------------- standarise
    def _standardize_train_segment(self, seg_df, feature_cols, global_mean,
                                   global_std, keep_cols=()) -> pd.DataFrame:
        """Standardise one contiguous training segment (no window crosses gaps)."""
        if self.standardize == 'rolling' and len(seg_df) < self.window_size:
            return pd.DataFrame()

        feats = seg_df.loc[:, feature_cols]
        if self.standardize == 'rolling':
            rolling_mean = feats.rolling(window=self.window_size, min_periods=self.window_size).mean()
            rolling_std = feats.rolling(window=self.window_size, min_periods=self.window_size).std(ddof=0)
            z = (feats - rolling_mean) / rolling_std.replace(0, np.nan)
        elif self.standardize == 'global':
            z = (feats - global_mean) / global_std
        else:  # 'none'
            z = feats

        out = seg_df[['_pos', '_group']].copy()
        out['time'] = seg_df[self.time_col_name].values
        for col in feature_cols:
            out[col] = z[col].values
        for col in keep_cols:
            out[col] = seg_df[col].values
        return out.dropna().reset_index(drop=True)

    def _build_train_samples(self, df, feature_cols, split_def, test_intervals):
        keep_cols = (self.target_col,) if self.target_col else ()

        raw_blocks = []
        segments = []
        for block_groups in self._contiguous_train_blocks(split_def.train_groups):
            block_df = df[df['_group'].isin(block_groups)].copy().reset_index(drop=True)
            cleaned = self._purge_embargo_block(block_df, test_intervals)
            raw_blocks.append(cleaned.copy())
            segments.extend(self._split_on_gaps(cleaned))

        if not segments:
            raise ValueError(f"split {split_def.split_id} 在 purging/embargo 后无可用训练样本")

        cleaned_train_raw = pd.concat(raw_blocks, ignore_index=True) if raw_blocks else pd.DataFrame(columns=df.columns)
        if cleaned_train_raw.empty:
            raise ValueError(f"split {split_def.split_id} 清洗后训练原始样本为空")

        if self.standardize == 'none':
            global_mean = pd.Series(0.0, index=list(feature_cols))
            global_std = pd.Series(1.0, index=list(feature_cols))
        else:
            global_mean = cleaned_train_raw.loc[:, feature_cols].mean()
            global_std = cleaned_train_raw.loc[:, feature_cols].std(ddof=0).replace(0, np.nan).fillna(1.0)

        normalized_blocks = []
        for seg in segments:
            seg_norm = self._standardize_train_segment(seg, feature_cols, global_mean, global_std, keep_cols)
            if not seg_norm.empty:
                normalized_blocks.append(seg_norm)
        if not normalized_blocks:
            raise ValueError(f"split {split_def.split_id} 标准化后无可用训练样本")

        train_samples = pd.concat(normalized_blocks, ignore_index=True)
        if len(train_samples) < self.min_train_samples:
            raise ValueError(
                f"split {split_def.split_id} 训练样本不足: {len(train_samples)} < {self.min_train_samples}"
            )
        return train_samples, global_mean, global_std, raw_blocks

    def _build_test_samples(self, df, feature_cols, test_groups, global_mean, global_std):
        test_df = df[df['_group'].isin(test_groups)].copy().reset_index(drop=True)
        out = test_df[['_pos', '_group']].copy()
        out['time'] = test_df[self.time_col_name].values
        if self.standardize == 'none':
            out.loc[:, feature_cols] = test_df.loc[:, feature_cols].values
        else:
            out.loc[:, feature_cols] = (test_df.loc[:, feature_cols] - global_mean) / global_std
        if self.target_col:
            out[self.target_col] = test_df[self.target_col].values
        return out.dropna().reset_index(drop=True)

    # ------------------------------------------------------------ validation
    def _assert_no_leakage(self, split_def, train_blocks_raw, train_samples, test_samples,
                           feature_cols, global_mean, global_std, test_intervals) -> None:
        for raw_block in train_blocks_raw:
            if raw_block.empty:
                continue
            assert not raw_block['_group'].isin(split_def.test_groups).any(), '训练块中混入测试组'
            start = raw_block['_pos'].to_numpy()
            end = raw_block['_event_end_pos'].to_numpy()
            for t_start, t_end in test_intervals:
                overlap = (start <= t_end + self.purge_ticks) & (end >= t_start - self.purge_ticks)
                embargo = (start > t_end) & (start <= t_end + self.embargo_ticks)
                assert not (overlap | embargo).any(), 'purging/embargo 失败，训练块仍含泄漏样本'
        assert train_samples[feature_cols].notna().all(axis=1).all(), '训练样本存在 NaN 特征'
        assert not test_samples[feature_cols].isna().any().any(), '测试特征存在 NaN'
        if self.standardize != 'none':
            assert global_mean.index.tolist() == list(feature_cols), '训练全局均值列顺序异常'
            assert global_std.index.tolist() == list(feature_cols), '训练全局标准差列顺序异常'
        assert test_samples['_group'].isin(split_def.test_groups).all(), '测试样本包含非测试组'
        assert set(train_samples['_group'].unique()).issubset(set(split_def.train_groups)), '训练样本跨到了测试组'

    def _build_paths_assignment(self, splits: Sequence[SplitDefinition]) -> Dict[int, int]:
        group_to_splits = {g: [] for g in range(self.n_groups)}
        for split_def in splits:
            for g in split_def.test_groups:
                group_to_splits[g].append(split_def.split_id)
        for g, ids in group_to_splits.items():
            assert len(ids) == self.phi, f'group {g} 出现次数应为 phi={self.phi}'

        assigned: Dict[int, int] = {}
        used = {(g, p): False for g in range(self.n_groups) for p in range(self.phi)}
        split_map = {s.split_id: s for s in splits}

        def candidates(split_id) -> List[int]:
            split = split_map[split_id]
            valid = []
            for p in range(self.phi):
                if all(not used[(g, p)] for g in split.test_groups):
                    valid.append(p)
            return valid

        remaining = set(split_map.keys())

        def backtrack() -> bool:
            if not remaining:
                return True

            split_id = min(remaining, key=lambda sid: len(candidates(sid)))
            cand = candidates(split_id)
            if not cand:
                return False

            remaining.remove(split_id)
            split = split_map[split_id]
            for p in cand:
                assigned[split_id] = p
                for g in split.test_groups:
                    used[(g, p)] = True
                if backtrack():
                    return True
                for g in split.test_groups:
                    used[(g, p)] = False
                assigned.pop(split_id, None)
            remaining.add(split_id)
            return False

        ok = backtrack()
        if not ok:
            raise RuntimeError('无法完成 CPCV φ 路径分配，请检查 N 和 k 是否可行')

        for p in range(self.phi):
            covered = []
            for split_id, path_id in assigned.items():
                if path_id == p:
                    covered.extend(split_map[split_id].test_groups)
            assert sorted(covered) == list(range(self.n_groups)), f'path {p} 未完整覆盖所有分组'
        return assigned

    def _rank_states_by_train_return(self, train_samples, train_states, data) -> Dict[int, int]:
        if self.return_col not in data.columns:
            raise KeyError(f"return_col '{self.return_col}' not found in the input dataframe")
        train_eval = train_samples[['_pos']].copy()
        train_eval['raw_state'] = train_states
        train_eval = train_eval.merge(data[['_pos', self.return_col]], on='_pos', how='left')

        ret_array = train_eval[self.return_col].astype(float).values
        state_final_nets = {}
        for raw_state in np.unique(train_states):
            signal = (train_eval['raw_state'] == raw_state).astype(float).values
            state_ret_array = signal[:-1] * ret_array[1:]
            final_net = float(np.prod(1 + state_ret_array))
            state_final_nets[int(raw_state)] = final_net

        ordered_states = [s for s, _ in sorted(state_final_nets.items(), key=lambda x: x[1])]
        return {int(raw_state): rank for rank, raw_state in enumerate(ordered_states)}

    # ------------------------------------------------------------------ main
    def run(self, df, feature_cols, model_factory, time_col, prediction_col) -> Dict[str, object]:
        """
        Run CPCV.

        Parameters
        ----------
        df : DataFrame
            Ordered or unordered table; it is sorted by ``time_col``.
        feature_cols : list of str
            Model input columns.
        model_factory : callable
            Returns a fresh model with ``fit`` / ``predict``.
        time_col : str
            Timestamp column used for ordering.
        prediction_col : str
            Name of the output label column.
        """
        data = self._prepare_dataframe(df, time_col)
        group_ranges = self._group_ranges(data)
        splits = self.generate_splits()
        split_to_path = self._build_paths_assignment(splits)

        path_outputs: Dict[int, List[pd.DataFrame]] = {p: [] for p in range(self.phi)}

        for split_def in splits:
            test_intervals = [(group_ranges[g].start_pos, group_ranges[g].end_pos) for g in split_def.test_groups]
            train_samples, global_mean, global_std, train_blocks_raw = self._build_train_samples(
                data, feature_cols, split_def, test_intervals,
            )
            test_samples = self._build_test_samples(
                data, feature_cols, split_def.test_groups, global_mean, global_std,
            )

            self._assert_no_leakage(
                split_def, train_blocks_raw, train_samples, test_samples,
                feature_cols, global_mean, global_std, test_intervals,
            )

            X_train = train_samples.loc[:, feature_cols].to_numpy()
            X_test = test_samples.loc[:, feature_cols].to_numpy()

            model = model_factory()
            if self.target_col:
                model.fit(X_train, train_samples.loc[:, self.target_col].to_numpy())
            else:
                model.fit(X_train)

            if self.rank_states:
                raw_train_states = model.predict(X_train)
                state_map = self._rank_states_by_train_return(train_samples, raw_train_states, data)
                raw_test_states = model.predict(X_test)
                preds = np.array([state_map[int(state)] for state in raw_test_states], dtype=int)
            else:
                preds = np.asarray(model.predict(X_test))

            path_id = split_to_path[split_def.split_id]
            pred_df = test_samples[['time', '_pos', '_group']].copy()
            pred_df[prediction_col] = preds
            pred_df['split_id'] = split_def.split_id
            pred_df['path_id'] = path_id
            pred_df['is_test'] = True

            path_outputs[path_id].append(pred_df)

        complete_paths: Dict[int, pd.DataFrame] = {}
        for path_id, pieces in path_outputs.items():
            path_df = pd.concat(pieces, ignore_index=True).sort_values('_pos').reset_index(drop=True)
            assert path_df['_group'].nunique() == self.n_groups, f'path {path_id} 没有覆盖全部分组'
            assert len(path_df['_group']) == len(path_df['_pos'].unique()), f'path {path_id} 存在重复样本'
            complete_paths[path_id] = path_df

        return {
            'prepared_data': data,
            'splits': splits,
            'phi': self.phi,
            'split_to_path': split_to_path,
            'paths': complete_paths,
        }

    def build_backtest_paths(self, result, prediction_col='state') -> Dict[int, pd.DataFrame]:
        """Attach the original columns to each complete path, indexed by time."""
        prepared = result['prepared_data'].copy()
        time_col = self.time_col_name or 'DateTime'
        path_frames: Dict[int, pd.DataFrame] = {}

        for path_id, path_pred in result['paths'].items():
            merged = prepared.merge(
                path_pred[['_pos', prediction_col, 'split_id', 'path_id']],
                on='_pos',
                how='inner',
            )
            merged = merged.sort_values('_pos').reset_index(drop=True)
            merged = merged.set_index(time_col)
            path_frames[path_id] = merged

        return path_frames


def run_cpcv(df, feature_cols, model_factory, *, time_col='DateTime',
             prediction_col='state', **cpcv_kwargs) -> Dict[str, object]:
    """
    One-call CPCV helper.

    Any extra keyword argument (``n_groups``, ``k_test_groups``, ``window_size``,
    ``purge_ticks``, ``embargo_ticks``, ``label_horizon_ticks``,
    ``min_train_samples``, ``return_col``, ``rank_states``, ``target_col``,
    ``standardize``, ``group_col``) is forwarded to :class:`CPCVGrouped`.

    Returns the run result with ``result['backtest_paths']`` attached.
    """
    cpcv = CPCVGrouped(**cpcv_kwargs)
    result = cpcv.run(
        df=df,
        feature_cols=feature_cols,
        model_factory=model_factory,
        time_col=time_col,
        prediction_col=prediction_col,
    )
    result['backtest_paths'] = cpcv.build_backtest_paths(result, prediction_col=prediction_col)
    return result


# ==========================================================================
# Hidden Markov Model (HMM) helpers
# ==========================================================================

def HMM_fit(n_components, covariance_type, n_iter, tol, random_state):
    return GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )


# ==========================================================================
# Statistical Jump Model (JM) helpers
# ==========================================================================

# 迭代算法
def viterbi_jump(y, theta, lam):
    """
    动态规划部分：在固定 theta 的情况下寻找最优路径
    y: (T, D) 特征矩阵
    theta: (K, D) 状态中心
    lam: 跳跃惩罚系数
    """
    T = len(y)  # 时间步数
    K = len(theta)  # 状态数量

    # V[t, k] 记录到时刻 t 停在状态 k 的最小代价
    V = np.zeros((T, K))
    # B[t, k] 记录回溯最优状态
    B = np.zeros((T, K), dtype=int)

    # 初始化 t=0
    for k in range(K):
        V[0, k] = np.sum((y[0] - theta[k]) ** 2)

    # 正向递推
    for t in range(1, T):
        for k in range(K):
            loss_t_k = np.sum((y[t] - theta[k]) ** 2)
            costs = []
            for j in range(K):
                penalty = 0 if j == k else lam
                costs.append(V[t - 1, j] + penalty)
            best_prev_state = np.argmin(costs)
            V[t, k] = loss_t_k + costs[best_prev_state]
            B[t, k] = best_prev_state

    # 反向回溯
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmin(V[-1, :])  # 在最后一天选出损失最小的状态
    for t in range(T - 2, -1, -1):
        states[t] = B[t + 1, states[t + 1]]

    return states, V


def fit_sjm(y, K, lam, max_iter):
    """坐标下降法训练 SJM"""
    T, D = y.shape
    # 随机初始化状态中心 (可以用 KMeans 初始化)
    kmeans = KMeans(n_clusters=K, random_state=42).fit(y)
    theta = kmeans.cluster_centers_

    states = np.zeros(T, dtype=int)
    V_new = np.zeros((T, K))
    count = 0
    for i in range(max_iter):
        old_states = states.copy()

        # Step A: 固定 theta, 优化状态序列 s (动态规划)
        states, V = viterbi_jump(y, theta, lam)

        # Step B: 固定 states, 优化 theta (取均值)
        for k in range(K):
            theta[k] = np.mean(y[states == k], axis=0)
        count += 1
        if np.array_equal(old_states, states):
            V_new = V
            print("最大迭代次数：", count)
            break
        V_new = V
    print("最大迭代次数：", count)
    return states, theta, V_new


# 测试集推理代码
def predict_test_set(y_test, theta, lam, states, V_grid):
    """
    y_test: 测试集特征 (T_test, D)
    theta: 训练好的状态中心 (K, D)
    lam: 训练时选定的最优惩罚项
    states: 训练集的最优状态序列
    V_grid: 训练集的动态规划代价矩阵
    """
    T_test = len(y_test)
    K = len(theta)

    test_states = np.zeros(T_test, dtype=int)

    # 记录前一天的累积代价（训练集中最后一天的结果）
    v_prev = V_grid[-1, :]
    v_prev[states[-1]] = 0

    for t in range(T_test):
        v_current = np.zeros(K)
        for k in range(K):
            dist_sq = np.sum((y_test[t] - theta[k]) ** 2)
            costs = []
            for j in range(K):
                penalty = 0 if j == k else lam
                costs.append(v_prev[j] + penalty)
            v_current[k] = dist_sq + min(costs)
        test_states[t] = np.argmin(v_current)
        v_prev = v_current

    return test_states

class SJMRegimeModel:
    """Scikit-learn style wrapper around the Statistical Jump Model."""

    def __init__(self, n_states=4, lam=150.0, max_iter=1000):
        self.n_states = n_states
        self.lam = lam
        self.max_iter = max_iter
        self.theta_ = None
        self.train_states_ = None
        self.v_grid_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.train_states_, self.theta_, self.v_grid_ = fit_sjm(
            X, K=self.n_states, lam=self.lam, max_iter=self.max_iter
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return predict_test_set(X, self.theta_, self.lam, self.train_states_, self.v_grid_)

