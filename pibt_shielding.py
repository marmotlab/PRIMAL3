import numpy as np
import torch

from pibt.pypibt.pibt import PIBT
from pibt_utils import get_neighbors


class PIBTInstance(PIBT):
    def __init__(
        self,
        grid,
        starts,
        goals,
        moves,
        sampling_method="deterministic",
        seed=0,
        priority_offsets=None,
    ):
        super().__init__(grid, starts, goals, seed)

        # Base priority is the standard distance-to-goal term (in [0, 1)), which acts
        # as a fine tie-breaker. An optional per-agent offset (e.g. an SVO/aging bias
        # supplied by the env) is added on top as the coarse, dominant component.
        self.priorities = []
        for i in range(self.N):
            base = self.dist_tables[i].get(self.starts[i]) / self.grid.size
            if priority_offsets is not None:
                base += float(priority_offsets[i])
            self.priorities.append(base)

        self.state = list(starts)
        self.reached_goals = False
        self.moves = moves
        self.sampling_method = sampling_method

    def _update_priorities(self):
        flg_fin = True
        for i in range(self.N):
            if self.state[i] != self.goals[i]:
                flg_fin = False
                self.priorities[i] += 1
            else:
                self.priorities[i] -= np.floor(self.priorities[i])
        self.reached_goals = flg_fin

    def funcPIBT(self, Q_from, Q_to, i, transition_probabilities, pibt_ids=None):
        C, move_idx, mask = get_neighbors(self.grid, Q_from[i], self.moves)

        if (pibt_ids is not None) and (pibt_ids[i] is not None):
            ids = pibt_ids[i]

        elif self.sampling_method == "deterministic":
            ids = np.arange(len(C))
            self.rng.shuffle(ids)
            ids = sorted(
                ids,
                key=lambda u: (
                    -transition_probabilities[i][move_idx[u]],
                    self.dist_tables[i].get(C[u]),
                ),
            )

        elif self.sampling_method == "probabilistic":
            cur_trans_probs = transition_probabilities[i][mask].astype(np.float64)
            s = np.sum(cur_trans_probs)
            if s <= 0:
                cur_trans_probs = np.ones_like(cur_trans_probs, dtype=np.float64)
                cur_trans_probs /= np.sum(cur_trans_probs)
            else:
                cur_trans_probs /= s

            ids = np.arange(len(C))
            ids = self.rng.choice(
                ids, size=len(C), replace=False, p=cur_trans_probs, shuffle=False
            )
        else:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}")

        for id_ in ids:
            v = C[id_]

            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.actions[i] = move_idx[id_]
            self.occupied_nxt[v] = i

            # priority inheritance
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (
                    not self.funcPIBT(
                        Q_from, Q_to, j, transition_probabilities, pibt_ids
                    )
                )
            ):
                continue

            return True

        # failed -> stay
        Q_to[i] = Q_from[i]
        self.actions[i] = 0
        self.occupied_nxt[Q_from[i]] = i
        return False

    def _step(self, Q_from, priorities, transition_probabilities, pibt_ids=None):
        N = len(Q_from)
        Q_to = []

        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        order = sorted(range(N), key=lambda i: priorities[i], reverse=True)
        for i in order:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, i, transition_probabilities, pibt_ids)

        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        return Q_to

    def step(self, transition_probabilities, pibt_ids=None):
        self.actions = np.zeros(self.N, dtype=np.int64)
        if self.reached_goals:
            return self.actions

        self.state = self._step(
            self.state, self.priorities, transition_probabilities, pibt_ids
        )
        self._update_priorities()
        return self.actions


class PIBTShielding:
    """
    Single-step PIBT collision shielding.

    支持:
    - hard actions: [N]
    - logits/probs: [N, A]
    """

    def __init__(
        self,
        obstacle_map,
        moves,
        seed=0,
        sampling_method="deterministic",
    ):
        self.grid = obstacle_map
        self.moves = moves
        self.seed = seed
        self.sampling_method = sampling_method

    def _build_instance(self, starts, goals, priority_offsets=None):
        return PIBTInstance(
            grid=self.grid,
            starts=starts,
            goals=goals,
            moves=self.moves,
            sampling_method=self.sampling_method,
            seed=self.seed,
            priority_offsets=priority_offsets,
        )

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _hard_actions_to_preferences(self, actions, n_actions):
        prefs = np.full((len(actions), n_actions), 1e-6, dtype=np.float64)
        for i, a in enumerate(actions):
            prefs[i, int(a)] = 1.0
        return prefs

    def shield(self, actions, starts, goals, input_type="actions", priority_offsets=None):
        starts = [tuple(s) for s in starts]
        goals = [tuple(g) for g in goals]

        pibt = self._build_instance(
            starts=starts, goals=goals, priority_offsets=priority_offsets
        )

        x = self._to_numpy(actions)
        n_actions = len(self.moves)

        if input_type == "actions":
            if x.ndim != 1:
                raise ValueError("Hard actions must have shape [N].")

            x = x.astype(np.int64)
            prefs = self._hard_actions_to_preferences(x, n_actions)
            safe_actions = pibt.step(prefs)
            return safe_actions.astype(np.int64)

        elif input_type in ("logits", "probs"):
            if x.ndim != 2:
                raise ValueError("Logits/probs must have shape [N, A].")

            x = x[:, :n_actions].astype(np.float64)

            if input_type == "probs":
                row_sum = np.sum(x, axis=-1, keepdims=True)
                row_sum[row_sum <= 0] = 1.0
                x = x / row_sum

            safe_actions = pibt.step(x)
            return safe_actions.astype(np.int64)

        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
        
