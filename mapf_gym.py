import numpy as np
from map_generator import *
from alg_parameters import *
import matplotlib.pyplot as plt
from util import getFreeCell, returnAsType, renderWorld, Status
from itertools import combinations
from collections import deque
import math

from pibt_shielding import PIBTShielding

class Agent():
    dirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 
               4: (-1, 0), 5: (1, 1), 6: (1, -1), 7: (-1, -1), 8: (-1, 1)}  # x,y operation for corresponding action
    
    oppositeAction = {0:-1, 1:3, 2:4, 3:1, 4:2}

    def __init__(self, world):
        self.__position = np.array([-1,-1])
        self.__goal = np.array([-1,-1])
        self.__emulatedStep = np.array([-1,-1])

        self.bfsMap = None
        self.heuristicMap = None

        self.svo_avg = None
        self.svo_exe = None
        self.svo_ipm = None
        self.svo_distri = 1 / EnvParameters.N_SVOs * np.ones(EnvParameters.N_SVOs)

        # Aging term for PIBT shielding priority: grows while the agent is away from
        # its goal, resets on arrival. This keeps a fresh per-step PIBTInstance from
        # discarding PIBT's usual priority aging.
        self.pibt_age = 0.0

        self.invalidActions = []
        # static invalid Actions

        self.world = world

        self.restrictedAction = dict()
        # otherAgent restricted Actions (represented as {x:[[a, y], ...]} meaning x action is invalid if agent 'a' takes y action simultaneously)

        self.previousAction = -1

        self.unconditionallyGoodActions = list()

        self.bfsMap = None
        self.heuristicMap = None

        self.fixed_conflict_neighbor = -1
        self.conflict_overlap = None
        self.fixed_same_neighbor = -1
        self.same_overlap = None

    def setGoodActions(self, actions):
        self.unconditionallyGoodActions = actions

    def setInvalidActions(self, actions):
        self.invalidActions = actions

    def updateRestrictedPosition(self, action, newRestriction):
        if action in self.restrictedAction:
            if newRestriction not in self.restrictedAction[action]:
                self.restrictedAction[action].append(newRestriction)
        else:
            self.restrictedAction[action] = [newRestriction]

    def setPos(self, pos):
        self.__position = np.array(pos)
        self.invalidActions = [[],[],[],[]]
        self.restrictedAction = dict()
        self.unconditionallyGoodActions = list()
        self.previousAction = -1
        # NOTE: pibt_age is intentionally NOT reset here. setPos is also the per-step
        # move primitive (takeStep), so resetting it here would wipe aging every step.
        # It is initialised in __init__ and reset on goal arrival in jointStep.


    def getPos(self, type='np'):
        return returnAsType(self.__position, type)

    def setGoal(self, goal):
        self.__goal = np.array(goal)

    def getGoal(self, type='np'):
        return returnAsType(self.__goal, type)
    
    def getEmulatedStep(self, type='np'):
        return returnAsType(self.__emulatedStep, type)

    def emulateStep(self, action):
        step = np.array(self.dirDict[action])
        self.__emulatedStep = np.add(self.getPos(), step)
    
    def takeStep(self, action):
        step = np.array(self.dirDict[action])
        self.setPos(np.add(self.getPos(), step))
        self.previousAction = self.oppositeAction[action]


class MapfGym():

    def __init__(self, restore = False, restorePath = 'savedEnv', num_agents=None):
        self.n_agents = num_agents if num_agents is not None else EnvParameters.N_AGENTS
        if not restore:
            # self.obstacleMap = maze_generator(env_size=EnvParameters.WORLD_SIZE, obstacle_density=EnvParameters.OBSTACLE_PROB).astype(int)
            self.obstacleMap = random_generator(SIZE_O=EnvParameters.WORLD_SIZE, PROB_O=EnvParameters.OBSTACLE_PROB).astype(int)
            # self.obstacleMap = house_generator(env_size=EnvParameters.WORLD_SIZE).astype(int)
            self.cutVertexMap = self.get_cutVertexMap(self.obstacleMap)
            self.degreeMap = self.get_degreeMap(self.obstacleMap)
            self.deadEndMap = self.get_deadEndMap(self.obstacleMap)
            self.agentList = [Agent(self.obstacleMap) for i in range(self.n_agents)]
            # assign agents and their goals in the map
            self.populateMap()
            # init the svo stuff
            conflict_neighbor, array_conflict_graph, same_neighbor, array_same_graph = self.get_agent_neighbor_and_overlap()
            for i in range(self.n_agents):
                agent = self.agentList[i]
                agent.fixed_conflict_neighbor = conflict_neighbor[i]
                agent.conflict_overlap = array_conflict_graph[i, :]
                agent.fixed_same_neighbor = same_neighbor[i]
                agent.same_overlap = array_same_graph[i, :]
                agent.svo_avg = np.dot(np.arange(EnvParameters.N_SVOs), agent.svo_distri)
                agent.svo_ipm = np.random.choice(range(EnvParameters.N_SVOs), p=agent.svo_distri.ravel())
                agent.svo_exe = (agent.svo_ipm * 5) / 180 * np.pi

        else:
            loaded = np.load(restorePath+'.npz')
            self.obstacleMap = loaded['map']

            self.agentList = [Agent(self.obstacleMap) for i in range(self.n_agents)]
            for idx, agent in enumerate(self.agentList):
                agent.setPos(loaded['agents'][idx])
                agent.setGoal(loaded['goals'][idx])

        self.allGoodActions = self.getUnconditionallyGoodActions(returnIsNeeded=True)

        self.pibt_MOVES = [
            (0, 0),   # 0
            (0, 1),   # 1
            (1, 0),   # 2
            (0, -1),  # 3
            (-1, 0),  # 4
        ]

        self.pibt_shield = PIBTShielding(
                            obstacle_map=(self.obstacleMap == 0),   # True 表示 free cell
                            moves=self.pibt_MOVES,
                            seed=SetupParameters.SEED,
                            sampling_method="deterministic",
                        )
        self.pibt_shield_calls = 0
        self.pibt_shield_skips = 0


    def replicate(self, world, agentPos, goalPos):
        self.n_agents = len(agentPos)
        self.obstacleMap = world
        self.cutVertexMap = self.get_cutVertexMap(self.obstacleMap)
        self.degreeMap = self.get_degreeMap(self.obstacleMap)
        self.deadEndMap = self.get_deadEndMap(self.obstacleMap)
        self.agentList = [Agent(self.obstacleMap) for i in range(self.n_agents)]
        for agentIdx, agent in enumerate(self.agentList):
            agent.setPos(tuple(agentPos[agentIdx]))
            agent.setGoal(tuple(goalPos[agentIdx]))
        # init the svo stuff
        conflict_neighbor, array_conflict_graph, same_neighbor, array_same_graph = self.get_agent_neighbor_and_overlap()
        for i in range(self.n_agents):
            agent = self.agentList[i]
            agent.fixed_conflict_neighbor = conflict_neighbor[i]
            agent.conflict_overlap = array_conflict_graph[i, :]
            agent.fixed_same_neighbor = same_neighbor[i]
            agent.same_overlap = array_same_graph[i, :]
            agent.svo_avg = np.dot(np.arange(EnvParameters.N_SVOs), agent.svo_distri)
            agent.svo_ipm = np.random.choice(range(EnvParameters.N_SVOs), p=agent.svo_distri.ravel())
            agent.svo_exe = (agent.svo_ipm * 5) / 180 * np.pi
        self.allGoodActions = self.getUnconditionallyGoodActions(returnIsNeeded=True)

        self.pibt_MOVES = [
            (0, 0),   # 0
            (0, 1),   # 1
            (1, 0),   # 2
            (0, -1),  # 3
            (-1, 0),  # 4
        ]

        self.pibt_shield = PIBTShielding(
                            obstacle_map=(self.obstacleMap == 0),   # True 表示 free cell
                            moves=self.pibt_MOVES,
                            seed=SetupParameters.SEED,
                            sampling_method="deterministic",
                        )
        self.pibt_shield_calls = 0
        self.pibt_shield_skips = 0

    def saveEnv(self, savePath = 'savedEnv'):
        map = self.obstacleMap
        goals = []
        agents = []
        for i in self.agentList:
            agents.append(i.getPos())
            goals.append(i.getGoal())
        agents = np.array(agents)
        goals = np.array(goals)

        np.savez_compressed(savePath, map=map, agents=agents, goals = goals)

    def populateMap(self, edgePoints = None):
        if edgePoints is None:
            tempMap = np.copy(self.obstacleMap)
            
            for i in self.agentList:
                i.setPos(getFreeCell(tempMap))
                tempMap[i.getPos(type='mat')] = 2
            tempMap = np.copy(self.obstacleMap)
            for i in self.agentList:
                i.setGoal(getFreeCell(tempMap))
                tempMap[i.getGoal(type='mat')] = 3
        else:
            np.random.shuffle(edgePoints)
            agentPos = copy.deepcopy(edgePoints)
            while True:
                np.random.shuffle(edgePoints)
                temp = True
                for i in range(self.n_agents):
                    if(np.array_equal(agentPos[i], edgePoints[i])):
                        temp = False
                        break 
                if(temp):
                    break
            for agentIdx, agent in enumerate(self.agentList):
                agent.setPos(agentPos[agentIdx])
                agent.setGoal(edgePoints[(agentIdx)])

    def getAgentsSVOexe(self):
        agentsSVOexe = np.zeros(self.n_agents, dtype=np.int64)
        for agentIdx in range(self.n_agents):
            agent = self.agentList[agentIdx]
            agentsSVOexe[agentIdx] = agent.svo_ipm
        return agentsSVOexe

    def worldWithAgents(self):
        world = np.copy(self.obstacleMap)
        for i,agent in enumerate(self.agentList):
            if not np.any(agent.getPos()<0):
                world[agent.getPos(type='mat')] = i+1

        return world
    
    def worldWithGoals(self):
        world = np.copy(self.obstacleMap)
        for i,agent in enumerate(self.agentList):
            if agent.getGoal()[0]>=0 and agent.getGoal()[1]>=0:
                world[agent.getGoal(type='mat')] = i+1

        return world
    
    def worldWithAgentsAndGoals(self):
        world = np.copy(self.obstacleMap)
        
        for i,agent in enumerate(self.agentList):
            if not np.any(agent.getPos()<0):
                world[agent.getPos(type='mat')] = i+1
            if not np.any(agent.getGoal()<0):
                world[agent.getGoal(type='mat')] = i+1

        return world

    def makeBfsMap(self, agent:Agent):
        # Deque-based BFS with O(1) visited checks; the previous version's
        # `not in openedList` scans made it O(cells^2) and dominated the first
        # step at large scales. Output layout is unchanged: reachable free cell
        # -> BFS distance from the goal, obstacle -> 1e6, unreachable free -> -2.
        bfsMap = np.copy(self.obstacleMap)

        bfsMap[bfsMap==0] = -2
        size = bfsMap.shape

        goal = agent.getGoal('mat')
        queue = deque()
        if bfsMap[goal] == -2:
            bfsMap[goal] = 0
            queue.append(goal)
        while queue:
            x, y = queue.popleft()
            value = bfsMap[x, y] + 1
            if x > 0 and bfsMap[x - 1, y] == -2:
                bfsMap[x - 1, y] = value
                queue.append((x - 1, y))
            if x + 1 < size[0] and bfsMap[x + 1, y] == -2:
                bfsMap[x + 1, y] = value
                queue.append((x + 1, y))
            if y > 0 and bfsMap[x, y - 1] == -2:
                bfsMap[x, y - 1] = value
                queue.append((x, y - 1))
            if y + 1 < size[1] and bfsMap[x, y + 1] == -2:
                bfsMap[x, y + 1] = value
                queue.append((x, y + 1))
        bfsMap[bfsMap==-1] = 1e6
        agent.bfsMap = bfsMap

    def getHeuristicMap(self, agent:Agent):
        if(agent.bfsMap is None):
            self.makeBfsMap(agent)

        # Vectorised form of the original per-cell loops, identical output:
        # heuristicMap[k, x, y] = 1 iff the k-neighbour of free cell (x, y) is
        # exactly one BFS step closer to the goal (the original's `<` check
        # implies equality with bfs-1, which its asserts enforced).
        bfs = agent.bfsMap
        H, W = self.obstacleMap.shape
        free = self.obstacleMap == 0
        sentinel = np.int64(10 ** 9)  # never equals bfs - 1

        heuristicMap = np.zeros((4, H, W)).astype(int)
        shifted = np.full((H, W), sentinel, dtype=np.int64)
        shifted[1:, :] = bfs[:-1, :]
        heuristicMap[0][free & (shifted == bfs - 1)] = 1
        shifted = np.full((H, W), sentinel, dtype=np.int64)
        shifted[:-1, :] = bfs[1:, :]
        heuristicMap[1][free & (shifted == bfs - 1)] = 1
        shifted = np.full((H, W), sentinel, dtype=np.int64)
        shifted[:, 1:] = bfs[:, :-1]
        heuristicMap[2][free & (shifted == bfs - 1)] = 1
        shifted = np.full((H, W), sentinel, dtype=np.int64)
        shifted[:, :-1] = bfs[:, 1:]
        heuristicMap[3][free & (shifted == bfs - 1)] = 1

        correction = np.ones((4, H, W)).astype(int)
        heuristicMap = heuristicMap * -1 + correction
        agent.heuristicMap = heuristicMap

    def _shortestPathFromBfs(self, agent: Agent, start):
        """FAST_PATHS replacement for astar_4(self.obstacleMap, start, goal):
        walks the agent's cached goal-BFS map downhill, O(path length) instead
        of a full search. Returns [start, ..., goal] with both endpoints, like
        astar_4; the route is equally short, but tie-breaking between
        equal-length routes may differ from A*."""
        if agent.bfsMap is None:
            self.makeBfsMap(agent)
        bfs = agent.bfsMap
        H, W = bfs.shape
        x, y = int(start[0]), int(start[1])
        d = int(bfs[x, y])
        if d < 0 or d >= 1000000:
            raise ValueError("No Path Exists")
        path = [(x, y)]
        while d > 0:
            d -= 1
            if x > 0 and bfs[x - 1, y] == d:
                x -= 1
            elif x + 1 < H and bfs[x + 1, y] == d:
                x += 1
            elif y > 0 and bfs[x, y - 1] == d:
                y -= 1
            elif y + 1 < W and bfs[x, y + 1] == d:
                y += 1
            else:
                raise ValueError("BFS descent failed")  # impossible on a valid bfsMap
            path.append((x, y))
        return path

    def _getFastPaths(self):
        """Per-agent shortest paths [start, ..., goal] from the cached BFS
        maps; reuses the copy built by get_agent_paths when the positions it
        was built for are still current."""
        positions = self.get_positions()
        if (getattr(self, '_fastPathsCache', None) is not None
                and getattr(self, '_fastPathsCachePos', None) == positions):
            return self._fastPathsCache
        paths = []
        for i in range(self.n_agents):
            agent = self.agentList[i]
            start = (int(agent.getPos()[0]), int(agent.getPos()[1]))
            goal = (int(agent.getGoal()[0]), int(agent.getGoal()[1]))
            paths.append([goal] if start == goal else self._shortestPathFromBfs(agent, start))
        self._fastPathsCache = paths
        self._fastPathsCachePos = positions
        return paths

    def get_positions(self):
        result = []
        for indexOfAgent in range(0, self.n_agents):
            agent = self.agentList[indexOfAgent]
            result.append(tuple(agent.getPos()))
        return result

    def get_goals(self):
        result = []
        for indexOfAgent in range(0, self.n_agents):
            agent = self.agentList[indexOfAgent]
            result.append(tuple(agent.getGoal()))
        return result

    def get_agent_neighbor_and_overlap(self):

        def fill_rows(list_of_rows, fill_coordinates):
            max_length = max(
                len(row) for row in list_of_rows
            )  # Find the length of the longest row

            # Iterate through each row in the list
            for i, row in enumerate(list_of_rows):
                if len(row) < max_length:  # If the length is smaller than the longest row
                    fill_count = max_length - len(row)
                    fill_values = fill_coordinates[
                        i % len(fill_coordinates)
                        ]  # Select the coordinates based on row index
                    row.extend(
                        [fill_values] * fill_count
                    )  # Add the coordinates to fill the remaining positions

            return list_of_rows

        def get_agent_paths(world, agent_coords, agent_goals):
            """
            Returns A* path for each agent.

            Args:
                world (2D list): The world grid representing the environment.
                agent_coords (list): List of agent coordinates in the format [[x1, y1], [x2, y2], ...].
                agent_goals (list): List of agent goals in the format [[x1, y1], [x2, y2], ...].

            Returns:
                list: A list of paths, where each path is a list of coordinates [[x1, y1], [x2, y2], ...].

            """
            paths = []  # list of paths for each agent
            forward_paths = []  # [start, ..., goal] per agent, kept for getAgentsMsgs
            for i in range(self.n_agents):
                start = tuple(agent_coords[i])
                goal = tuple(agent_goals[i])
                if start != goal:
                    if EnvParameters.FAST_PATHS:
                        path = self._shortestPathFromBfs(self.agentList[i], start)
                    else:
                        path, _ = astar_4(world, start, goal)
                    forward_paths.append(list(path))
                    path.reverse()
                    path.extend([goal])
                else:
                    path = [goal]
                    forward_paths.append([goal])
                paths.append(path)
            if EnvParameters.FAST_PATHS:
                # snapshot BEFORE fill_rows pads the rows in place, so
                # getAgentsMsgs can reuse the clean paths this timestep
                self._fastPathsCache = forward_paths
                self._fastPathsCachePos = list(agent_coords)
            tuple_goal = list(map(tuple, agent_goals))
            filled_path = fill_rows(paths, tuple_goal)
            return filled_path

        # def create_direction(paths):
        #     """
        #     DIRECTION MAPPING: [1, 2, 3, 4]: [0, 1], [-1, 0], [0, -1], [1, 0]
        #     """

        #     def check(prev, curr):
        #         """
        #         Prev: Coordinate of the path ex: [3, 4]
        #         Curr: After coordinate of the path, ex: [4, 4]
        #         """
        #         if curr[0] - prev[0] == 0 and curr[1] - prev[1] == 1:
        #             return 1
        #         elif curr[0] - prev[0] == -1 and curr[1] - prev[1] == 0:
        #             return 2
        #         elif curr[0] - prev[0] == 0 and curr[1] - prev[1] == -1:
        #             return 3
        #         elif curr[0] - prev[0] == 1 and curr[1] - prev[1] == 0:
        #             return 4
        #         else:  # no change in direction?
        #             return 0

        #     # create a zero list whose size is same as paths
        #     directions = [[0 for _ in path] for path in paths]

        #     for index, path in enumerate(paths):
        #         length = len(path)
        #         if not path or length <= 1:
        #             continue  # path
        #         for i in range(1, length):  # [0, 1], [1, 0], [-1, 0], [0, -1]
        #             directions[index][i - 1] = check(path[i], path[i - 1])
        #         directions[index][length - 1] = 0

        #     return directions

        def create_direction(paths):
            """
            DIRECTION MAPPING:
                1 -> [0, 1]   (right)
                2 -> [-1, 0]  (up)
                3 -> [0, -1]  (left)
                4 -> [1, 0]   (down)
                0 -> stay / terminal padding
            """
            def check(prev, curr):
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]

                if dx == 0 and dy == 1:
                    return 1
                elif dx == -1 and dy == 0:
                    return 2
                elif dx == 0 and dy == -1:
                    return 3
                elif dx == 1 and dy == 0:
                    return 4
                else:
                    return 0

            directions = [[0 for _ in path] for path in paths]

            for index, path in enumerate(paths):
                length = len(path)
                if length <= 1:
                    continue
                for i in range(1, length):
                    directions[index][i - 1] = check(path[i - 1], path[i])  # fixed
                directions[index][length - 1] = 0

            return directions

        # def overlapping_squares(paths, directions):
        #     """
        #     (1) Add all paths to each coordinate (x,y): [(agent, direction), (agent, direction), ... ]
        #     (2) For each pair of agents (agent1, agent2) add # of collisions,

        #     """
        #     # for all agents, calculate its discounted detour caused by other agents
        #     weighted_sum_detour = np.zeros((self.n_agents, self.n_agents, 1))

        #     # iterate through all paths, finding the agents at each square
        #     mapping = {}
        #     for index, path in enumerate(paths):
        #         agent_num = index + 1  # zero indexed
        #         for j, (x, y) in enumerate(path):
        #             if (x, y) not in mapping:
        #                 mapping[(x, y)] = set()
        #             mapping[(x, y)].add((agent_num, directions[index][j]))

        #     for overlapping_pos, agents in mapping.items():
        #         if len(agents) <= 1:
        #             continue  # no overlapping paths
        #         for x, y in combinations(list(agents), 2):
        #             agent1, dir1 = x
        #             agent2, dir2 = y
        #             if dir1 == dir2 or agent1 == agent2:
        #                 continue  # same direction does not add to detour

        #             dist_to_agent1 = paths[agent1 - 1].index(overlapping_pos)
        #             dist_to_agent2 = paths[agent2 - 1].index(overlapping_pos)
        #             weighted_sum_detour[agent1 - 1][agent2 - 1] += EnvParameters.OVERLAP_DECAY ** dist_to_agent1 + EnvParameters.OVERLAP_DECAY ** dist_to_agent2
        #             weighted_sum_detour[agent2 - 1][agent1 - 1] += EnvParameters.OVERLAP_DECAY ** dist_to_agent1 + EnvParameters.OVERLAP_DECAY ** dist_to_agent2

        #     return weighted_sum_detour

        def overlapping_squares_directed(paths, directions):
            """
            Returns
            -------
            directed_conflict : np.ndarray, shape (N, N, 1)
                Directed mismatch-overlap score.
                [i, j, 0] means overlap cost imposed on agent i by agent j.

                - opposite direction: coefficient = 1.0
                - different but not opposite: coefficient = 0.5

            directed_same_dir : np.ndarray, shape (N, N, 1)
                Directed same-direction overlap score.
                [i, j, 0] means same-direction overlap imposed on agent i by agent j.
            """
            n_agents = self.n_agents
            gamma_conflict = EnvParameters.CONFLICT_OVERLAP_DECAY
            gamma_same = EnvParameters.SAME_OVERLAP_DECAY

            directed_conflict = np.zeros((n_agents, n_agents, 1), dtype=np.float32)
            directed_same_dir = np.zeros((n_agents, n_agents, 1), dtype=np.float32)

            # mapping[(x, y)] = [(agent_id, direction, step_idx), ...]
            mapping = {}
            for agent_idx, path in enumerate(paths):
                agent_id = agent_idx + 1
                for step_idx, pos in enumerate(path):
                    if pos not in mapping:
                        mapping[pos] = []
                    mapping[pos].append((agent_id, directions[agent_idx][step_idx], step_idx))

            def is_opposite(dir1, dir2):
                opposite_pairs = {
                    (1, 3), (3, 1),  # right <-> left
                    (2, 4), (4, 2),  # up <-> down
                }
                return (dir1, dir2) in opposite_pairs

            for overlapping_pos, visits in mapping.items():
                if len(visits) <= 1:
                    continue

                for a, b in combinations(visits, 2):
                    agent1, dir1, dist1 = a
                    agent2, dir2, dist2 = b

                    if agent1 == agent2:
                        continue

                    # ignore terminal/stay direction
                    if dir1 == 0 and dir2 == 0:
                        continue

                    w1_conflict = gamma_conflict ** dist1
                    w2_conflict = gamma_conflict ** dist2

                    w1_same = gamma_same ** dist1
                    w2_same = gamma_same ** dist2

                    if dir1 == dir2:
                        directed_same_dir[agent1 - 1, agent2 - 1, 0] += w1_same
                        directed_same_dir[agent2 - 1, agent1 - 1, 0] += w2_same
                    elif dir1 == 0 or dir2 == 0:
                        # think about it: currently this part setting only for maze map
                        # in the random map, we might do not need this 
                        directed_conflict[agent1 - 1, agent2 - 1, 0] += 5 * w1_conflict
                        directed_conflict[agent2 - 1, agent1 - 1, 0] += 5 * w2_conflict
                    else:
                        coef = 1.0 if is_opposite(dir1, dir2) else 0.5
                        directed_conflict[agent1 - 1, agent2 - 1, 0] += coef * w1_conflict
                        directed_conflict[agent2 - 1, agent1 - 1, 0] += coef * w2_conflict

            return directed_conflict, directed_same_dir

        paths = get_agent_paths(self.obstacleMap, self.get_positions(), self.get_goals())
        directions = create_direction(paths)
        # weighted_sum_detour = overlapping_squares(paths, directions)
        directed_conflict, directed_same_dir = overlapping_squares_directed(paths, directions)
        # based on the weighted sum detour, we determine the neighbor we are going to select
        # reshape the weighted_sum_detour as a np.array with size num_agent * num_agents
        array_wsd = np.array(directed_conflict).squeeze()
        neighbor = []
        count_self_neighbor = 0
        for row in array_wsd:
            if np.all(row == 0):  # if the row only contains zeros, no agent will have overlap path with it
                neighbor.append(np.where((array_wsd == row).all(axis=1))[0][count_self_neighbor])
                count_self_neighbor = count_self_neighbor + 1
            else:
                # Find indices of the maximum values, choose one randomly if there are multiple
                max_indices = np.where(row == row.max())[0]
                random_index = np.random.choice(max_indices)
                neighbor.append(random_index)
        assert len(neighbor) == self.n_agents
        """
        The neighbor is a np.array like [agent_id_1. agent_id_2. agent_id_3. agent_id_4]
        """
        neighbor = np.array(neighbor)
        neighbor = neighbor + 1

        # directed_neighbor and array_directed_same_dir
        array_directed_same_dir = np.array(directed_same_dir).squeeze()
        directed_same_neighbor = []
        count_directed_same_neighbor = 0
        for row in array_directed_same_dir:
            if np.all(row == 0):
                directed_same_neighbor.append(np.where((array_directed_same_dir == row).all(axis=1))[0][count_directed_same_neighbor])
                count_directed_same_neighbor = count_directed_same_neighbor + 1
            else:
                max_indices = np.where(row == row.max())[0]
                random_index = np.random.choice(max_indices)
                directed_same_neighbor.append(random_index)
        assert len(directed_same_neighbor) == self.n_agents
        directed_same_neighbor = np.array(directed_same_neighbor)
        directed_same_neighbor = directed_same_neighbor + 1

        return neighbor, array_wsd, directed_same_neighbor, array_directed_same_dir

    def observe(self, indexOfAgent=-1, world=None):
        agent = self.agentList[indexOfAgent]

        if agent.heuristicMap is None:
            self.getHeuristicMap(agent)

        #PART 1: FOV Observations

        top_left = (agent.getPos()[0] - EnvParameters.FOV_SIZE // 2, agent.getPos()[1] - EnvParameters.FOV_SIZE // 2)  # (top, left)
        top_left_heuristic = (agent.getPos()[0] - EnvParameters.FOV_Heuristic // 2,
                              agent.getPos()[1] - EnvParameters.FOV_Heuristic // 2)
        bottom_right_heuristic = (
            top_left_heuristic[0] + EnvParameters.FOV_Heuristic, top_left_heuristic[1] + EnvParameters.FOV_Heuristic)

        observations = np.zeros((NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE))  #observations per parameters and FOV Size
        observations[5:, :, :] = 1
        # 0: obs map
        # 1: other Agents
        # 2: own goal
        # 3: agents' in Fov goals
        # 4: svo Other Agents
        # 5: heuri_map1
        # 6: heuri_map2
        # 7: heuri_map3
        # 8: heuri_map4

        if world is None:  # caller may pass the shared per-step copy
            world = self.worldWithAgents()
        size = world.shape

        visibleAgents = list()

        for i in range(top_left[0], top_left[0] + EnvParameters.FOV_SIZE):  # top and bottom
            for j in range(top_left[1], top_left[1] + EnvParameters.FOV_SIZE):  # left and right
                
                if i >= size[0] or i < 0 or j >= size[1] or j < 0:
                    # out of boundaries (in obstacle map)
                    observations[0,i - top_left[0], j - top_left[1]] = 1
                    continue
                elif world[i,j] == -1:
                    #obstacle (in obstacle map)
                    observations[0,i - top_left[0], j - top_left[1]] = 1

                elif world[i,j] >0 and world[i,j]== indexOfAgent+1:
                    #self Position (in obstacle map)
                    observations[1,i - top_left[0], j - top_left[1]] = 1
                
                elif world[i,j]>0:
                    # other agents in FOV (in agent Map)
                    visibleAgents.append(world[i,j])
                    observations[1,i - top_left[0], j - top_left[1]] = 1 
                    
                    otherAgent = world[i,j]-1
                    # todo: this part should use svo or svo_avg? or we should ignore this channel?
                    if(self.agentList[otherAgent].svo_avg < agent.svo_avg):
                        observations[4,i - top_left[0], j - top_left[1]] = 1

                if top_left_heuristic[0] <= i <= bottom_right_heuristic[0] and top_left_heuristic[1] <= j <= bottom_right_heuristic[1] and 0 <= i < self.obstacleMap.shape[0] and 0 <= j < self.obstacleMap.shape[1]:
                    observations[5, i - top_left[0], j - top_left[1]] = agent.heuristicMap[0, i, j]
                    observations[6, i - top_left[0], j - top_left[1]] = agent.heuristicMap[1, i, j]
                    observations[7, i - top_left[0], j - top_left[1]] = agent.heuristicMap[2, i, j]
                    observations[8, i - top_left[0], j - top_left[1]] = agent.heuristicMap[3, i, j]

                if self.cutVertexMap[i, j] == 1:
                    observations[9, i - top_left[0], j - top_left[1]] = 1

                if self.degreeMap[i, j] == 2:
                    observations[10, i - top_left[0], j - top_left[1]] = 1



        if(top_left[0]<= agent.getGoal()[0]<top_left[0] + EnvParameters.FOV_SIZE and top_left[1]<= agent.getGoal()[1]<top_left[1] + EnvParameters.FOV_SIZE):
            # own goal in FOV (in own goal frame)
            observations[2,agent.getGoal()[0] - top_left[0], agent.getGoal()[1] - top_left[1]] = 1

        for agentIndex in visibleAgents:
            # print(agentIndex)
            x, y = self.agentList[agentIndex-1].getGoal()
            # projection of visible agents' goal in FOV (in others' goal frame)
            min_node = (max(top_left[0], min(top_left[0] + EnvParameters.FOV_SIZE - 1, x)),
                        max(top_left[1], min(top_left[1] + EnvParameters.FOV_SIZE - 1, y)))
            observations[3,min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        #PART2: Goal Vector and Prev Action
        vector = np.zeros(NetParameters.VECTOR_LEN)

        vector[0] = agent.getGoal()[0] - agent.getPos()[0]  # distance on x axes
        vector[1] = agent.getGoal()[1] - agent.getPos()[1]  # distance on y axes
        vector[2] = (vector[0] ** 2 + vector[1] ** 2) ** .5  # total distance
        if vector[2] != 0:  # normalized
            vector[0] = vector[0] / vector[2]
            vector[1] = vector[1] / vector[2]
        if EnvParameters.CLIP:
            vector[2] = min(vector[2], EnvParameters.CLIP_VECTOR_DIST_MAX)

        # if(agent.previousAction is None ):
        #     prevAction = 0
        # else:
        prevAction = agent.previousAction
        
        vector[3] = prevAction

        # social value orientation
        svo = agent.svo_distri

        return observations, vector, svo
    
    def get_degreeMap(self, obstacle_map):
        degree_map = np.full(obstacle_map.shape, -1, dtype=int)
        rows, cols = obstacle_map.shape
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Iterate over each cell in the grid
        for r in range(rows):
            for c in range(cols):
                if obstacle_map[r, c] == 0:  # Only compute degrees for free cells
                    degree = 0
                    # Check each of the four possible neighbors
                    for dr, dc in neighbors:
                        rr, cc = r + dr, c + dc
                        # Check if the neighbor is within bounds and is a free cell
                        if 0 <= rr < rows and 0 <= cc < cols and obstacle_map[rr, cc] == 0:
                            degree += 1
                    # Store the computed degree in the degree_map
                    degree_map[r, c] = degree
        return degree_map
    
    def get_deadEndMap(self, obstacle_map):
        """
        Returns a map where:
          1  = cell is in a dead end (degree-1 tip or degree-2 corridor leading to a degree-1 tip)
          0  = free cell, not in a dead end
         -1  = obstacle cell
        A dead end is a chain of degree-2 cells terminating in a degree-1 cell.
        """
        rows, cols = obstacle_map.shape
        _neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        deg = self.degreeMap
        dead_end_map = np.where(obstacle_map == 0, 0, -1)

        for r in range(rows):
            for c in range(cols):
                if deg[r, c] != 1:
                    continue
                prev, curr = None, (r, c)
                while True:
                    dead_end_map[curr] = 1
                    cr, cc = curr
                    nxt = None
                    for dr, dc in _neighbors:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and obstacle_map[nr, nc] == 0:
                            if (nr, nc) != prev:
                                nxt = (nr, nc)
                                break
                    if nxt is None or deg[nxt[0], nxt[1]] != 2:
                        break
                    prev, curr = curr, nxt

        return dead_end_map

    def get_cutVertexMap(self, obstacle_map):
        """
        In cutvertexMap:
        0 means this cell is not a cut vertex;
        1 means this cell is a cut vertex;
        -1 means this cell is an obstacle cell.
        """
        cutVertexMap = copy.deepcopy(obstacle_map)
        _, num_regions_pre = skimage.measure.label(obstacle_map, background=-1, connectivity=1, return_num=True)
        for i in range(obstacle_map.shape[0]):
            for j in range(obstacle_map.shape[1]):
                if obstacle_map[i, j] != -1:
                    temp_obstacle_map = copy.deepcopy(obstacle_map)
                    temp_obstacle_map[i, j] = -1
                    _, num_regions_aft = skimage.measure.label(temp_obstacle_map, background=-1, connectivity=1, return_num=True)
                    if num_regions_pre != num_regions_aft:
                        cutVertexMap[i, j] = 1
        return cutVertexMap
    
    def getAgentsMsgs(self):
        agent_messages = np.zeros((self.n_agents, NetParameters.MSG_LEN))

        fast_cell_count = None
        if EnvParameters.FAST_PATHS:
            # softBlocked support: count, per cell, how many moving agents'
            # shortest paths cross it (same fellow qualification as the slow
            # per-fellow astar_4 loop below; shortest paths never revisit a
            # cell, so each fellow contributes at most 1 per cell).
            fast_paths = self._getFastPaths()
            fast_cell_count = {}
            for j in range(self.n_agents):
                fellow = self.agentList[j]
                if (fellow.getPos()[0] != fellow.getGoal()[0]) or (fellow.getPos()[1] != fellow.getGoal()[1]):
                    for cell in fast_paths[j]:
                        key = (int(cell[0]), int(cell[1]))
                        fast_cell_count[key] = fast_cell_count.get(key, 0) + 1

        for i in range(self.n_agents):
            agent = self.agentList[i]
            # determine agent current position's distance to the goal - e
            dist2goal = agent.bfsMap[agent.getPos()[0], agent.getPos()[1]]
            # determine whether the goal position is a cut vertex - a
            cutGoal = self.cutVertexMap[agent.getGoal()[0], agent.getGoal()[1]]
            # calculate how many cutvertices it have to cross - b
            numCutvertex = 0
            if (agent.getPos()[0], agent.getPos()[1]) != (agent.getGoal()[0], agent.getGoal()[1]):
                if EnvParameters.FAST_PATHS:
                    path = self._shortestPathFromBfs(agent, (agent.getPos()[0], agent.getPos()[1]))
                else:
                    path, _ = astar_4(self.obstacleMap, (agent.getPos()[0], agent.getPos()[1]), (agent.getGoal()[0], agent.getGoal()[1]))
                for waypoint in path:
                    if self.cutVertexMap[waypoint[0], waypoint[1]] == 1:
                        numCutvertex = numCutvertex + 1
                if cutGoal:
                    numCutvertex = numCutvertex + 1
            else:
                if cutGoal:
                    numCutvertex = numCutvertex + 1
            # how many agents will be blocked if there is an obstacle on the agent's goal - c
            hardBlocked = 0
            if cutGoal:
                temp_goal_as_obstacle_map = copy.deepcopy(self.obstacleMap)
                temp_goal_as_obstacle_map[agent.getGoal()[0], agent.getGoal()[1]] = -1
                labeled_map, _ = skimage.measure.label(temp_goal_as_obstacle_map, background=-1, connectivity=1, return_num=True)
                for fellow_id in range(self.n_agents):
                    if EnvParameters.CLIP and hardBlocked >= EnvParameters.CLIP_MSG_HARD_BLOCKED_MAX:
                        break
                    if fellow_id != i:
                        fellow_agent = self.agentList[fellow_id]
                        if labeled_map[fellow_agent.getPos()[0], fellow_agent.getPos()[1]] != labeled_map[fellow_agent.getGoal()[0], fellow_agent.getGoal()[1]]:
                            hardBlocked = hardBlocked + 1
            # how many agents overlap with the ego agent if the ego agent stays on its goal (is cutGoal)
            softBlocked = 0
            if cutGoal:
                if fast_cell_count is not None:
                    softBlocked = fast_cell_count.get((int(agent.getPos()[0]), int(agent.getPos()[1])), 0)
                    if (agent.getPos()[0] != agent.getGoal()[0]) or (agent.getPos()[1] != agent.getGoal()[1]):
                        softBlocked = softBlocked - 1  # exclude the agent's own path
                else:
                    for fellow_id in range(self.n_agents):
                        if EnvParameters.CLIP and softBlocked >= EnvParameters.CLIP_MSG_SOFT_BLOCKED_MAX:
                            break
                        if fellow_id != i:
                            fellow_agent = self.agentList[fellow_id]
                            if (fellow_agent.getPos()[0], fellow_agent.getPos()[1]) != (fellow_agent.getGoal()[0], fellow_agent.getGoal()[1]):
                                fellow_path, _ = astar_4(self.obstacleMap, (fellow_agent.getPos()[0], fellow_agent.getPos()[1]), (fellow_agent.getGoal()[0], fellow_agent.getGoal()[1]))
                                for fellow_waypoint in fellow_path:
                                    if (fellow_waypoint[0] == agent.getPos()[0]) and (fellow_waypoint[1] == agent.getPos()[1]):
                                        softBlocked = softBlocked + 1
            # whether the agent’s goal is in a dead end
            goalInDeadEnd = 1 if self.deadEndMap[agent.getGoal()[0], agent.getGoal()[1]] == 1 else 0
            # whether the agent’s current position is in a dead end
            agentInDeadEnd = 1 if self.deadEndMap[agent.getPos()[0], agent.getPos()[1]] == 1 else 0
            # clip the scale-dependent entries to the maxima observable in training
            if EnvParameters.CLIP:
                numCutvertex = min(numCutvertex, EnvParameters.CLIP_MSG_NUM_CUTVERTEX_MAX)
                hardBlocked = min(hardBlocked, EnvParameters.CLIP_MSG_HARD_BLOCKED_MAX)
                softBlocked = min(softBlocked, EnvParameters.CLIP_MSG_SOFT_BLOCKED_MAX)
                dist2goal = min(dist2goal, EnvParameters.CLIP_MSG_DIST2GOAL_MAX)
            # Collate all the heuristic data obtained as the agent’s message
            agent_messages[i, :] = np.array([cutGoal, numCutvertex, hardBlocked, softBlocked, dist2goal, goalInDeadEnd, agentInDeadEnd])
        """
        cutGoal: binary value indicating whether the goal is a cut vertex
        numCutvertex: number of cut vertices on the path to the goal
        hardBlocked: number of agents that would be blocked if there is an obstacle on the agent’s goal
        softBlocked: number of agents whose optimal path would overlap with the agent if the agent stays on its goal
        dist2goal: distance from the agent’s current position to its goal
        goalInDeadEnd: 1 if the agent’s goal is in a dead end (degree-1 tip or degree-2 corridor leading to one), else 0
        agentInDeadEnd: 1 if the agent’s current position is in a dead end, else 0
        """
        return agent_messages


    def getAllObservations(self):
        allObs = np.zeros((1, self.n_agents, NetParameters.NUM_CHANNEL , EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE), dtype=np.float32)
        allVectors = np.zeros((1, self.n_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
        allSvos = np.zeros((1, self.n_agents, EnvParameters.N_SVOs), dtype=np.float32)
        allConflictIndex = np.zeros((1, self.n_agents, self.n_agents), dtype=np.float32)
        allSameIndex = np.zeros((1, self.n_agents, self.n_agents), dtype=np.float32)
        allMsgs = np.zeros((1, self.n_agents, NetParameters.MSG_LEN), dtype=np.float32)

        # calculate the updated neighbor for all agents
        conflict_neighbor, array_conflict_graph, same_neighbor, array_same_graph = self.get_agent_neighbor_and_overlap()
        # updated the self.fixed_neighbor according to the current neighbor and weighted_detour
        # only if we solve the previous neighbor, we can select a new neighbor
        for i in range(self.n_agents):
            agent = self.agentList[i]
            if array_conflict_graph[i][agent.fixed_conflict_neighbor - 1] == 0:
                agent.fixed_conflict_neighbor = conflict_neighbor[i]
            else:
                pass
            if array_same_graph[i][agent.fixed_same_neighbor - 1] == 0:
                agent.fixed_same_neighbor = same_neighbor[i]
            else:
                pass

        sharedWorld = self.worldWithAgents()  # identical for every agent this step
        for i in range(0, self.n_agents):
            observations, vector, svo = self.observe(i, world=sharedWorld)

            allObs[:, i] = observations
            allVectors[:, i] = vector
            allSvos[:, i] = svo

        # According to all agent's neighbor, we determine the comms graph
        conflict_adj_mat = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            agent = self.agentList[i]
            conflict_adj_mat[i][agent.fixed_conflict_neighbor - 1] = 1
            conflict_adj_mat[agent.fixed_conflict_neighbor - 1][i] = 1
        for i in range(self.n_agents):
            allConflictIndex[:, i, :] = conflict_adj_mat[i]

        same_adj_mat = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            agent = self.agentList[i]
            same_adj_mat[i][agent.fixed_same_neighbor - 1] = 1
            same_adj_mat[agent.fixed_same_neighbor - 1][i] = 1
        for i in range(self.n_agents):
            allSameIndex[:, i, :] = same_adj_mat[i]

        # get the heuristic msgs for all agents
        agent_messages = self.getAgentsMsgs()
        for i in range(self.n_agents):
            allMsgs[:, i, :] = agent_messages[i]
        
        return allObs, allVectors, allSvos, allConflictIndex, allSameIndex, allMsgs

    def getInvalidActions(self):

            for agent in self.agentList:
                
                staticInvalidAction = list()

                for i in range(0, EnvParameters.N_ACTIONS):
                    agent.emulateStep(i)
                    pos = agent.getEmulatedStep('mat')

                    if not ((0<= pos[0] < self.obstacleMap.shape[0]) and (0<= pos[1] <self.obstacleMap.shape[1])): ## Falling out of map
                        staticInvalidAction.append(i)
                        continue
                    elif(self.obstacleMap[pos] !=0): ## Running into walls
                        staticInvalidAction.append(i)
                        continue
                agent.setInvalidActions(staticInvalidAction)
            
    def getRestrictedActions(self):
        np.zeros((self.n_agents, self.n_agents, EnvParameters.N_ACTIONS, EnvParameters.N_ACTIONS))
        # Get set of codependent restricted actions

        #Part1: get possible agents that can collide

        agentsAtRisk = list()

        # Vectorised (chunked) version of the original O(N^2) python pair loop;
        # produces the identical [i, j] (i < j) list in the identical order.
        positions = np.array([agent.getPos() for agent in self.agentList], dtype=np.int64)
        chunk = max(1, int(4e6 // max(self.n_agents, 1)))
        for row_start in range(0, self.n_agents, chunk):
            row_end = min(row_start + chunk, self.n_agents)
            diff = positions[row_start:row_end, None, :] - positions[None, :, :]
            d2 = (diff * diff).sum(axis=2)
            for i_off, j in np.argwhere(d2 <= 4):
                i = row_start + int(i_off)
                j = int(j)
                if i < j:
                    agentsAtRisk.append([i, j])

        #Part2: get simultaneous actions which cause collision
        for agentOneIndex,agentTwoIndex in agentsAtRisk:

            agentOne = self.agentList[agentOneIndex]
            agentTwo = self.agentList[agentTwoIndex]
            currentDistance = np.sum(np.square(agentOne.getPos() - agentTwo.getPos()))

            for i in range(EnvParameters.N_ACTIONS):
                agentOne.emulateStep(i)

                #Collision is only possible if the agents get closer or atleast stay at the same distance
                if np.sum(np.square(agentOne.getEmulatedStep() - agentTwo.getPos()))<=currentDistance: 
                    
                    #Now check which corressponding action(if any) of agentTwo causes a vertex collision
                    for j in range(EnvParameters.N_ACTIONS):
                        agentTwo.emulateStep(j)

                        if(np.array_equal(agentOne.getEmulatedStep(), agentTwo.getEmulatedStep())):
                            
                            # Add the to the lists
                            agentOne.updateRestrictedPosition(i, [agentTwoIndex, j])
                            agentTwo.updateRestrictedPosition(j, [agentOneIndex, i])

                    # Also account for swapping collision
                    if(np.array_equal(agentOne.getEmulatedStep(), agentTwo.getPos())):
                        agentOne.updateRestrictedPosition(i, [agentTwoIndex, Agent.oppositeAction[i]])
                        agentTwo.updateRestrictedPosition(Agent.oppositeAction[i], [agentOneIndex, i])
        
    def getUnconditionallyGoodActions(self, returnIsNeeded = False):
        # First get bad actions
        self.getInvalidActions()
        self.getRestrictedActions()

        if(returnIsNeeded):
            allGoodActions = list()

        for agent in self.agentList:
            badActions = list()

            badActions += agent.invalidActions

            for i in agent.restrictedAction:
                badActions.append(i)
            

            goodActions = np.setdiff1d(np.arange(EnvParameters.N_ACTIONS),badActions)
            agent.setGoodActions(goodActions)

            if(returnIsNeeded):
                allGoodActions.append(goodActions)
        
        if(returnIsNeeded):
            return allGoodActions

    def getStaticColl(self, actionStatus):
        return np.where(actionStatus==Status.STATIC_COLLISION.value)[0]

    def _set_agent_svo(self, agent, svo_distri, executed_svo=None):
        svo_distri = np.asarray(svo_distri, dtype=np.float64).copy()
        total = np.sum(svo_distri)
        if total <= 0:
            svo_distri = np.ones(EnvParameters.N_SVOs, dtype=np.float64) / EnvParameters.N_SVOs
        else:
            svo_distri /= total

        agent.svo_distri = svo_distri
        agent.svo_avg = float(np.dot(np.arange(EnvParameters.N_SVOs), agent.svo_distri))
        if executed_svo is None:
            agent.svo_ipm = int(np.random.choice(range(EnvParameters.N_SVOs), p=agent.svo_distri.ravel()))
        else:
            agent.svo_ipm = int(np.clip(executed_svo, 0, EnvParameters.N_SVOs - 1))
        agent.svo_exe = (agent.svo_ipm * 5) / 180 * np.pi

    def _update_svo_from_model(self, svo_output):
        svo_output = np.squeeze(np.asarray(svo_output, dtype=np.float64))
        if svo_output.ndim == 1:
            svo_output = np.reshape(svo_output, (self.n_agents, EnvParameters.N_SVOs))

        for i in range(self.n_agents):
            agent = self.agentList[i]
            if (agent.getPos()[0] != agent.getGoal()[0]) or (agent.getPos()[1] != agent.getGoal()[1]):
                self._set_agent_svo(agent, svo_output[i])
            else:
                selfless = np.zeros(EnvParameters.N_SVOs, dtype=np.float64)
                selfless[-1] = 1.0
                self._set_agent_svo(agent, selfless, executed_svo=EnvParameters.N_SVOs - 1)

    def _pibt_priority_offsets(self):
        svo_weight = EnvParameters.PIBT_SVO_WEIGHT
        age_weight = EnvParameters.PIBT_AGE_WEIGHT
        if self.n_agents > EnvParameters.PIBT_SVO_MAX_AGENTS:
            svo_weight = EnvParameters.PIBT_LARGE_AGENT_SVO_WEIGHT

        # print([agent.pibt_age for agent in self.agentList])
        # print([svo_weight * ((EnvParameters.N_SVOs - 1) - agent.svo_avg) for agent in self.agentList])

        return np.array(
            [
                age_weight * agent.pibt_age
                + svo_weight * ((EnvParameters.N_SVOs - 1) - agent.svo_avg)
                for agent in self.agentList
            ],
            dtype=np.float64,
        )

    # def getActionStatus(self, actions, svo_output):
    #     # update the svo_distri/svo_avg of all agents
    #     for i in range(self.n_agents):
    #         agent = self.agentList[i]
    #         if (agent.getPos()[0] != agent.getGoal()[0]) or (agent.getPos()[1] != agent.getGoal()[1]):
    #             agent.svo_distri = svo_output[i]
    #             agent.svo_avg = np.dot(np.arange(EnvParameters.N_SVOs), agent.svo_distri)
    #             agent.svo_ipm = np.random.choice(range(EnvParameters.N_SVOs), p=svo_output[i].ravel())
    #             agent.svo_exe = (agent.svo_ipm * 5) / 180 * np.pi
    #         else:
    #             selfless = np.zeros(EnvParameters.N_SVOs)
    #             selfless[-1] = 1
    #             agent.svo_distri = selfless
    #             agent.svo_avg = np.dot(np.arange(EnvParameters.N_SVOs), agent.svo_distri)
    #             agent.svo_ipm = np.random.choice(range(EnvParameters.N_SVOs), p=agent.svo_distri.ravel())
    #             agent.svo_exe = (agent.svo_ipm * 5) / 180 * np.pi
    #     # -------------------------------------------
    #     assert(len(actions)==self.n_agents)
    #     actions = np.copy(actions)
    #     agentList = list(self.getSvoOrder()[::-1])
    #     actionStatus = np.full(shape=self.n_agents, fill_value=10)
        
    #     while not len(agentList) == 0:
    #         indexOfAgent = agentList.pop(0)
    #         agent = self.agentList[indexOfAgent]
    #         action = actions[indexOfAgent]

    #         try:
    #             assert(action in range(EnvParameters.N_ACTIONS))
    #         except:
    #             print(actions, indexOfAgent, action)
    #             raise Exception("Well, Shit")
            
    #         if action in agent.invalidActions: ##This caluses a static collision
    #             actionStatus[indexOfAgent] = min(Status.STATIC_COLLISION.value, actionStatus[indexOfAgent])
    #             actions[indexOfAgent] = 0
    #             if 0 in agent.restrictedAction:
    #                 for fellowAgent,_ in agent.restrictedAction[0]:
    #                     if(fellowAgent not in agentList):
    #                         agentList.append(fellowAgent)
            
    #         elif action in agent.restrictedAction:
    #             for fellowAgent,agentAction in agent.restrictedAction[action]: ##Check if this is a restricted action and a collision is being caused due to it
    #                     if(actions[fellowAgent]==agentAction):

    #                         if(agent.svo_avg >= self.agentList[fellowAgent].svo_avg):
    #                             actionStatus[indexOfAgent] = min(Status.AGENT_COLLSION.value, actionStatus[indexOfAgent])
    #                         if(self.agentList[fellowAgent].svo_avg >= agent.svo_avg):
    #                             actionStatus[fellowAgent] = min(Status.AGENT_COLLSION.value, actionStatus[fellowAgent])

    #                         actions[indexOfAgent] = 0
    #                         if 0 in agent.restrictedAction:
    #                             for fellowAgent,_ in agent.restrictedAction[0]:
    #                                 if(fellowAgent not in agentList):
    #                                     agentList.append(fellowAgent)
                
    #         if (actionStatus[indexOfAgent] == 10): ## This means this is a valid action. It might have been restricted but the other agent might be performing some other action, hence it is valid.
    #             actionStatus[indexOfAgent] = Status.VALID.value
                
    #         if(actionStatus[indexOfAgent] == Status.VALID.value):
    #             agent.emulateStep(action)
    #             if(np.array_equal(agent.getEmulatedStep(), agent.getGoal())):
    #                 actionStatus[indexOfAgent] = Status.REACH_GOAL.value
                
    #             elif(np.array_equal(agent.getPos(), agent.getGoal()) and actions[indexOfAgent]!=0):
    #                 actionStatus[indexOfAgent] = Status.LEAVE_GOAL.value
                
    #             elif(actions[indexOfAgent]==agent.previousAction):
    #                 actionStatus[indexOfAgent] = Status.REPEAT_ACTION.value
        
    #     assert not np.any((actionStatus)==10) # I'll cry if this is an error

    #     return actionStatus, actions

    def getActionStatus(self, actions, svo_output, ps=None):
        self._update_svo_from_model(svo_output)

        actions = np.asarray(actions).copy().astype(np.int64)
        assert len(actions) == self.n_agents

        starts = [tuple(agent.getPos().tolist()) for agent in self.agentList]
        goals = [tuple(agent.getGoal().tolist()) for agent in self.agentList]
        priority_offsets = self._pibt_priority_offsets()

        n_actions = len(self.pibt_MOVES)
        pibt_actions = actions.copy()
        invalid_action_ids = (pibt_actions < 0) | (pibt_actions >= n_actions)
        pibt_actions[invalid_action_ids] = 0

        shield_input_type = EnvParameters.PIBT_SHIELD_INPUT_TYPE
        if shield_input_type not in ("actions", "probs"):
            raise ValueError(f"Unsupported PIBT_SHIELD_INPUT_TYPE: {shield_input_type}")

        if shield_input_type == "probs" and ps is not None:
            prefs = np.squeeze(np.asarray(ps, dtype=np.float64))
            if prefs.ndim == 1:
                prefs = np.reshape(prefs, (self.n_agents, -1))
            prefs = prefs.copy()[:, :n_actions]
            prefs = np.maximum(prefs, 0.0)
            row_max = prefs.max(axis=1, keepdims=True)
            for i in range(self.n_agents):
                prefs[i, int(pibt_actions[i])] = row_max[i, 0] + 1.0
            safe_actions = self.pibt_shield.shield(
                actions=prefs,
                starts=starts,
                goals=goals,
                input_type="probs",
                priority_offsets=priority_offsets,
            )
        else:
            safe_actions = self.pibt_shield.shield(
                actions=pibt_actions,
                starts=starts,
                goals=goals,
                input_type="actions",
                priority_offsets=priority_offsets,
            )

        self.pibt_shield_calls += 1

        actionStatus = np.full(shape=self.n_agents, fill_value=Status.VALID.value)

        for i in range(self.n_agents):
            agent = self.agentList[i]
            orig_action = int(actions[i])
            final_action = int(safe_actions[i])

            if orig_action in agent.invalidActions:
                actionStatus[i] = Status.STATIC_COLLISION.value
            elif final_action != orig_action:
                actionStatus[i] = Status.AGENT_COLLSION.value

            step = np.array(agent.dirDict[final_action])
            next_pos = agent.getPos() + step

            if np.array_equal(next_pos, agent.getGoal()):
                actionStatus[i] = Status.REACH_GOAL.value
            elif np.array_equal(agent.getPos(), agent.getGoal()) and final_action != 0:
                actionStatus[i] = Status.LEAVE_GOAL.value
            elif final_action == agent.previousAction:
                actionStatus[i] = Status.REPEAT_ACTION.value

        return actionStatus, safe_actions

    def getNonInvalidActions(self):
        nonInvalidAction = []
        for agentIdx, agent in enumerate(self.agentList):
            nonInvalidAction.append(np.setdiff1d(np.arange(5),agent.invalidActions))

        return nonInvalidAction

    # TODO: Maybe we can remove Blocking if we set the priority of an agent who has reached its goal to minimum
    def getBlockingReward(self, indexOfAgent):
        def astar(world, start, goal, robots):
            """A* function for single agent"""
            # print(start, goal)
            for (i, j) in robots:
                world[i, j] = -1
            try:
                path = astar_4(world, start, goal)
            except Exception as e:
                # print(e)
                path = None
            for (i, j) in robots:
                world[i, j] = 0
            return path
        
        other_agents = []
        other_locations = []
        inflation = 10

        agent = self.agentList[indexOfAgent]

        top_left = (agent.getPos()[0] - EnvParameters.FOV_SIZE // 2,
                    agent.getPos()[1] - EnvParameters.FOV_SIZE // 2)
        bottom_right = (top_left[0] + EnvParameters.FOV_SIZE, top_left[1] + EnvParameters.FOV_SIZE)
        for agentIdx, fellowAgent in enumerate(self.agentList):
            if agentIdx == indexOfAgent:
                continue
            x, y = fellowAgent.getPos()
            if x < top_left[0] or x >= bottom_right[0] or y >= bottom_right[1] or y < top_left[1]:
                # exclude agent not in FOV
                continue
            other_agents.append(agentIdx)
            other_locations.append((x, y))
        num_blocking = 0
        world = np.copy(self.obstacleMap)
        for agentIdx in other_agents:
            fellowAgent = self.agentList[agentIdx]
            other_locations.remove(fellowAgent.getPos('mat'))
            # print(agentIdx, other_locations)
            # before removing
            path_before = astar(world, fellowAgent.getPos('mat'), fellowAgent.getGoal('mat'),
                                     robots=other_locations + [agent.getPos('mat')])
            # print(path_before)
            # after removing
            path_after = astar(world, fellowAgent.getPos('mat'), fellowAgent.getGoal('mat'),
                                     robots=other_locations)
            # print(path_after)



            other_locations.append(fellowAgent.getPos('mat'))
            if path_before is None and path_after is None:
                continue
            if path_before is not None and path_after is None:
                continue
            if (path_before is None and path_after is not None) or (len(path_before) > len(path_after) + inflation):
                num_blocking += 1
        return num_blocking * EnvParameters.BLOCKING_COST, num_blocking

    def calculateReward(self, actions, actionStatus):
        svo_post_rewards = np.zeros((1, self.n_agents), dtype=np.float32)
        action_post_rewards = np.zeros((1, self.n_agents), dtype=np.float32)
        baseRewards = np.zeros((1, self.n_agents), dtype=np.float32)
        blockings = np.zeros((1, self.n_agents), dtype=np.float32)
        leaveGoals = np.zeros((1, self.n_agents), dtype=np.float32)
        numCollide = np.zeros((1, self.n_agents), dtype=np.float32)

        for agentIdx, agent in enumerate(self.agentList):

            if(actions[agentIdx]==0):
                if actionStatus[agentIdx]==Status.REACH_GOAL.value:
                    baseRewards[:,agentIdx] = EnvParameters.GOAL_REWARD
                    if self.n_agents < 32:  # do not calculate A* for increasing speed
                        blockingReward, num_blocking = self.getBlockingReward(agentIdx)
                        # print(agentIdx, num_blocking)
                        baseRewards[:, agentIdx] += blockingReward
                        if blockingReward < 0:
                            blockings[:, agentIdx] = num_blocking
                elif actionStatus[agentIdx]==Status.REPEAT_ACTION.value:
                    baseRewards[:,agentIdx] = EnvParameters.IDLE_COST
                elif actionStatus[agentIdx]==Status.VALID.value:
                    baseRewards[:, agentIdx] = EnvParameters.IDLE_COST
                else:
                    baseRewards[:, agentIdx] = EnvParameters.COLLISION_COST
                    numCollide[:, agentIdx]+=1
            
            else:
                if actionStatus[agentIdx] == Status.REACH_GOAL.value:
                    baseRewards[:, agentIdx] = EnvParameters.GOAL_REWARD
                
                elif actionStatus[agentIdx] == Status.STATIC_COLLISION.value or \
                            actionStatus[agentIdx] == Status.AGENT_COLLSION.value:
                    baseRewards[:, agentIdx] = EnvParameters.COLLISION_COST
                    numCollide[:, agentIdx]+=1
                elif actionStatus[agentIdx]==Status.REPEAT_ACTION.value:
                    baseRewards[:,agentIdx] = EnvParameters.ACTION_COST
                else:
                    baseRewards[:, agentIdx] = EnvParameters.ACTION_COST
                    if actionStatus[agentIdx] == Status.LEAVE_GOAL.value:
                        leaveGoals[:, agentIdx] += 1

        # re-assign the rewards according to the svo
        for IndexOfAgent in range(self.n_agents):
            agent = self.agentList[IndexOfAgent]
            other_agent_rewards = baseRewards[0][agent.fixed_conflict_neighbor - 1]
            svo_post_rewards[:, IndexOfAgent] = (baseRewards[0][IndexOfAgent] + other_agent_rewards) / EnvParameters.IMPORTANCE_SVO
            action_post_rewards[:, IndexOfAgent] = math.cos(agent.svo_exe) * baseRewards[0][IndexOfAgent] + math.sin(agent.svo_exe) * other_agent_rewards

        return svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide
    
    def getTrainValid(self, actions, actionStatus):
        trainValid = np.ones((self.n_agents, EnvParameters.N_ACTIONS), dtype=np.float32)

        for idx, agent in enumerate(self.agentList):
            for action in agent.invalidActions:
                trainValid[idx, action] = 0
            if(actionStatus[idx]==Status.AGENT_COLLSION.value):
                trainValid[idx, int(actions[idx])] = 0
            
            if(agent.previousAction!=-1):
                trainValid[idx, agent.previousAction] = 0

        return trainValid


    def isConflict(self, agentActionPairs, nextAgentIdx, nextAgentAction):
        if(nextAgentAction in self.agentList[nextAgentIdx].restrictedAction and len(np.array([x for x in set(tuple(x) for x in agentActionPairs) & set(tuple(x) for x in self.agentList[nextAgentIdx].restrictedAction[nextAgentAction])]))!=0):
                return True
        return False    

    def getSvoOrder(self):
        order = []
        for i in self.agentList:
            order.append(i.svo_avg)
        return np.argsort(order)

    def jointStep(self, actions= None, actionStatus=None):
        # if actionStatus is None:
        #     actionStatus, fixedActions = self.getActionStatus(actions)
        #     actions = fixedActions

        goalsReached = np.zeros(self.n_agents)

        for agentIdx, agent in enumerate(self.agentList):
            agent.takeStep(actions[agentIdx])

            if(np.array_equal(agent.getPos(), agent.getGoal())):
                goalsReached[agentIdx] = 1
                # reached goal -> reset PIBT aging (mirrors PIBT's priority reset at goal)
                agent.pibt_age = 0.0
            else:
                # away from goal -> age, so a repeatedly-yielding agent eventually
                # accumulates enough priority to override the SVO bias (starvation-free)
                agent.pibt_age += 1.0

        self.allGoodActions = self.getUnconditionallyGoodActions(returnIsNeeded=False)

        if(np.array_equal(goalsReached, np.ones_like(goalsReached))):
            done = True
        else:
            done = False

        return goalsReached, done

    def _render(self):
        goals = []
        agents = []

        for i in self.agentList:
            agents.append(i.getPos('mat'))
            goals.append(i.getGoal('mat'))
            

        return renderWorld(world=self.obstacleMap, agents=agents,goals=goals,svoOrder=self.getSvoOrder()+1,n_agents=self.n_agents)

        
