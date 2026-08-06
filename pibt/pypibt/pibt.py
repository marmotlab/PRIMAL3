import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Configs, Coord, Grid, get_neighbors


class PIBT:
    def __init__(self, grid: Grid, starts: Config, goals: Config, seed: int = 0):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.N = len(self.starts)

        # distance table
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: self.dist_tables[i].get(u))

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(self, Q_from: Config, priorities: list[float]) -> Config:
        # setup
        N = len(Q_from)
        Q_to: Config = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, i)

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        return Q_to

    def run(self, max_timestep: int = 1000) -> Configs:
        # define priorities
        priorities: list[float] = []
        for i in range(self.N):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / self.grid.size)

        # main loop, generate sequence of configurations
        configs = [self.starts]
        while len(configs) <= max_timestep:
            # obtain new configuration
            Q = self.step(configs[-1], priorities)
            configs.append(Q)

            # update priorities & goal check
            flg_fin = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += 1
                else:
                    priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break  # goal

        return configs


class PIBTDistanceBased(PIBT):
    def get_manhattan_distance(self, i: int, u: Coord) -> int:
        gx, gy = self.goals[i]
        ux, uy = u
        return abs(gx - ux) + abs(gy - uy)

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: self.get_manhattan_distance(i, u))

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False


class PIBTSamplingBased(PIBT):
    def __init__(
        self,
        grid,
        starts,
        goals,
        seed=0,
        sampling_temperature=1.0,
        sampling_strategy="softmax",
        sample_iteratively=False,
    ):
        super().__init__(grid, starts, goals, seed)
        self.sampling_temperature = sampling_temperature
        self.sampling_strategy = sampling_strategy
        self.sample_iteratively = sample_iteratively

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        cost_to_gos = np.array([self.dist_tables[i].get(u) for u in C])

        if self.sampling_temperature == 0:
            ids = np.argsort(cost_to_gos)
        elif self.sample_iteratively:
            ids = np.arange(len(cost_to_gos))
            mask = np.ones(len(cost_to_gos), dtype=bool)

            ans = []
            for _ in range(len(cost_to_gos)):
                oids = self.get_n_samples(cost_to_gos[mask], 1, ids[mask])
                ans.append(oids)
                mask[oids] = False
            ids = np.concatenate(ans)
        else:
            ids = self.get_n_samples(cost_to_gos, len(cost_to_gos))

        C = [C[i] for i in ids]

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def get_sample_distribution(self, cost_to_gos):
        if self.sampling_strategy in ["softmax", "exponential"]:
            cost_to_gos = cost_to_gos - np.min(cost_to_gos)
            cost_to_gos = np.exp(-cost_to_gos / self.sampling_temperature)
        elif self.sampling_strategy == "reciprocal":
            cost_to_gos = cost_to_gos - np.min(cost_to_gos) + self.sampling_temperature
            cost_to_gos = 1 / cost_to_gos
        else:
            raise NotImplementedError(
                f"Sampling strategy {self.sampling_strategy} not implemented yet."
            )
        return cost_to_gos

    def get_n_samples(self, cost_to_gos, num_samples, ids=None):
        cost_to_gos = self.get_sample_distribution(cost_to_gos)
        if ids is None:
            ids = np.arange(len(cost_to_gos))
        try:
            cost_to_gos = cost_to_gos / np.sum(cost_to_gos)

            ids = self.rng.choice(
                ids,
                size=num_samples,
                replace=False,
                p=cost_to_gos,
                shuffle=False,
            )
        except:
            # Potential error due to zeroing of some probs
            EPSILON = 1e-6

            cost_to_gos = cost_to_gos + EPSILON
            cost_to_gos = cost_to_gos / np.sum(cost_to_gos)

            ids = self.rng.choice(
                ids,
                size=num_samples,
                replace=False,
                p=cost_to_gos,
                shuffle=False,
            )
        return ids
