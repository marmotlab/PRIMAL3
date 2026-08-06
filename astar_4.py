from heapq import heappop, heappush


def construct_path_from_dict(parents, goal, start):
    current = goal
    parent = parents[current]
    path = [current, parent]

    while parent is not start:
        temp = parent
        parent = parents[temp]
        path.append(parent)

    path.reverse()

    return path


def manhattan_distance(a, b):  # heuristic function
    (x1, y1) = a
    (x2, y2) = b

    return abs(x1 - x2) + abs(y1 - y2)


def astar_4(grid_world, start, goal):
    size_h = grid_world.shape[0] - 1
    size_w = grid_world.shape[1] - 1
    open_list = []
    complete_closed_list = []
    closed_list = set()
    g_scores = {}
    parents = {}
    heappush(open_list, (0, 0, start, None))  # f, g, cell, parent

    while open_list:
        # Move to the best cell
        cell = heappop(open_list)
        previous_cost = cell[1]
        # previous_cell = cell[2]
        current_cell = cell[2]

        if current_cell == goal:
            complete_closed_list.append(cell)
            path = construct_path_from_dict(parents, goal, start)
            return path, closed_list

        if current_cell in closed_list:
            continue  # ignore cells already evaluated

        complete_closed_list.append(cell)
        closed_list.add(current_cell)

        x = current_cell[0]
        y = current_cell[1]

        if y > 0 and grid_world[x][y - 1] != -1 and (x, y - 1) not in closed_list:
            left = (x, y - 1)
            new_g_score = previous_cost + 1

            if left in g_scores and g_scores[(x, y - 1)] < new_g_score:
                parent = parents[left]
            else:
                g_scores[left] = new_g_score
                parents[left] = current_cell
                parent = current_cell
            f_score = manhattan_distance(left, goal) + g_scores[left]  # f(n) = g(n) + h(n)
            heappush(open_list, (f_score, g_scores[left], left, parent))

        if x > 0 and grid_world[x - 1][y] != -1 and (x - 1, y) not in closed_list:
            up = (x - 1, y)
            new_g_score = previous_cost + 1

            if up in g_scores and g_scores[(x - 1, y)] < new_g_score:
                parent = parents[up]
            else:
                g_scores[up] = new_g_score
                parents[up] = current_cell
                parent = current_cell

            f_score = manhattan_distance(up, goal) + g_scores[up]  # f(n) = g(n) + h(n)
            heappush(open_list, (f_score, g_scores[up], up, parent))

        if y < size_w and grid_world[x][y + 1] != -1 and (x, y + 1) not in closed_list:
            right = (x, y + 1)
            new_g_score = previous_cost + 1

            if right in g_scores and g_scores[(x, y + 1)] < new_g_score:
                parent = parents[right]
            else:
                g_scores[right] = new_g_score
                parents[right] = current_cell
                parent = current_cell

            f_score = manhattan_distance(right, goal) + g_scores[right]  # f(n)  g(n) + h(n)
            heappush(open_list, (f_score, g_scores[right], right, parent))

        if x < size_h and grid_world[x + 1][y] != -1 and (x + 1, y) not in closed_list:
            down = (x + 1, y)
            new_g_score = previous_cost + 1

            if down in g_scores and g_scores[(x + 1, y)] < new_g_score:
                parent = parents[down]
            else:
                g_scores[down] = new_g_score
                parents[down] = current_cell
                parent = current_cell

            f_score = manhattan_distance(down, goal) + g_scores[down]  # f(n)  g(n) + h(n)
            heappush(open_list, (f_score, g_scores[down], down, parent))

    raise ValueError("No Path Exists")