"""
Dijkstra算法实现
用于与MCTS进行对比
"""

import heapq
from typing import List, Tuple, Dict, Optional
from environment.grid import GridMap


class DijkstraPathFinder:
    """Dijkstra路径规划算法"""

    def __init__(self, grid_map: GridMap):
        """
        初始化Dijkstra路径规划器

        Args:
            grid_map: 栅格地图
        """
        self.grid_map = grid_map
        self.nodes_explored = 0
        self.path_length = 0

    def search(self) -> Tuple[Optional[List[Tuple[int, int]]], Dict]:
        """
        执行Dijkstra搜索

        Returns:
            (路径, 统计信息)
        """
        start = self.grid_map.start
        goal = self.grid_map.goal

        # 优先队列: (distance, counter, position)
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start))

        # 记录每个节点的父节点
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # 从起点到每个节点的最短距离
        distance: Dict[Tuple[int, int], float] = {start: 0}

        # 已探索的节点
        closed_set = set()

        self.nodes_explored = 0

        while open_set:
            # 取出距离最小的节点
            current_dist, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            self.nodes_explored += 1
            closed_set.add(current)

            # 到达目标
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                self.path_length = len(path)
                return path, self.get_statistics()

            # 扩展邻居节点
            for neighbor in self.grid_map.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # 计算从起点经过current到neighbor的距离
                move_cost = self._get_move_cost(current, neighbor)
                tentative_dist = distance[current] + move_cost

                if neighbor not in distance or tentative_dist < distance[neighbor]:
                    # 找到更短的路径
                    came_from[neighbor] = current
                    distance[neighbor] = tentative_dist
                    counter += 1
                    heapq.heappush(open_set, (tentative_dist, counter, neighbor))

        # 没有找到路径
        return None, self.get_statistics()

    def _get_move_cost(self, from_pos: Tuple[int, int],
                       to_pos: Tuple[int, int]) -> float:
        """计算移动代价"""
        dy = abs(to_pos[0] - from_pos[0])
        dx = abs(to_pos[1] - from_pos[1])

        # 对角线移动代价为sqrt(2)，直线移动代价为1
        if dy + dx == 2:  # 对角线移动
            return 1.414
        return 1.0

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]],
                          current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """从目标回溯重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # 反转路径

    def get_statistics(self) -> dict:
        """获取搜索统计信息"""
        return {
            'nodes_explored': self.nodes_explored,
            'path_length': self.path_length,
        }
