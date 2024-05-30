#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/30 14:22
# @Author : Sunx
# @Last Modified by: Sunx
# @Software: PyCharm

import heapq
import math
from collections import namedtuple

class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                           (-1, 1), (1, 1), (1, -1), (-1, -1)]
        self.Node = namedtuple('Node', ['x', 'y'])
        print("rows: ", self.rows)
        print("cols: ", self.cols)

    def Euli_heuristic(self, sp, tp):
        h = math.sqrt((sp[0] - tp[0]) ** 2 + (sp[1] - tp[1]) ** 2)
        return h

    def Manh_heuristic(self, sp, tp):
        return math.fabs(tp[0] - sp[0]) + math.fabs(tp[1] - sp[1])

    def Octi_heuristic(self, sp, tp):
        h = math.sqrt(2)*min(math.fabs(tp[0] - sp[0]), math.fabs(tp[1] - sp[1]))
        s = abs(abs(tp[0] - sp[0]) - abs(tp[0] - sp[0]))
        return h + s

    def Cheb_herustic(self, sp, tp):
        return max(abs(tp[0] - sp[0]), abs(tp[1] - sp[1]))

    def distance(self, sp, tp):
        d = math.sqrt((sp[0] - tp[0]) ** 2 + (sp[1] - tp[1]) ** 2)
        return d

    def in_bounds_and_passable(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x][y] == 0

    def neighbors(self, node):
        for dx, dy in self.directions:
            x, y = node.x + dx, node.y + dy
            if self.in_bounds_and_passable(x, y):
                yield self.Node(x, y)

    def reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append((current.x, current.y))
            current = came_from[current]
        path.append((start.x, start.y))
        path.reverse()
        return path

    def search(self, start, goal):
        start_node = self.Node(*start)
        goal_node = self.Node(*goal)
        frontier = []
        heapq.heappush(frontier, (0, start_node))
        came_from = {start_node: None}
        cost_so_far = {start_node: 0}

        while frontier:
            cost, current = heapq.heappop(frontier)

            if current == goal_node:
                cost = cost
                return cost, self.reconstruct_path(came_from, start_node, goal_node)

            current_cost = cost_so_far[current]
            for next_node in self.neighbors(current):
                new_cost = current_cost + self.distance(next_node, current)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.Euli_heuristic(next_node, goal_node)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current

        return None  # return None if no path is found

# Example usage:
# if __name__ == "__main__":
#     grid = [
#         [0, 1, 0, 0, 0],
#         [0, 1, 0, 1, 0],
#         [0, 0, 0, 1, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, 0, 1, 0]
#     ]
#
#     astar = AStar(grid)
#     start = (0, 0)
#     goal = (4, 4)
#     path = astar.search(start, goal)
#     print("Path found:", path)
