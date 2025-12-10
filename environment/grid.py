"""
2D栅格地图环境模块
支持生成不同规模的栅格地图，随机放置障碍物
"""

import numpy as np
from typing import Tuple, List, Optional
import random


class GridMap:
    """2D栅格地图类"""

    # 地图元素常量
    FREE = 0        # 可通行区域
    OBSTACLE = 1    # 障碍物
    START = 2       # 起点
    GOAL = 3        # 终点

    # 四方向移动（上下左右）
    DIRECTIONS_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    # 八方向移动（包含对角线）
    DIRECTIONS_8 = [(0, 1), (0, -1), (1, 0), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(self, width: int = 30, height: int = 30,
                 obstacle_density: float = 0.2,
                 start: Optional[Tuple[int, int]] = None,
                 goal: Optional[Tuple[int, int]] = None,
                 seed: Optional[int] = None,
                 allow_diagonal: bool = False):
        """
        初始化栅格地图

        Args:
            width: 地图宽度
            height: 地图高度
            obstacle_density: 障碍物密度 (0.1-0.3)
            start: 起点坐标，默认左上角
            goal: 终点坐标，默认右下角
            seed: 随机种子
            allow_diagonal: 是否允许对角线移动
        """
        self.width = width
        self.height = height
        self.obstacle_density = np.clip(obstacle_density, 0.1, 0.3)
        self.allow_diagonal = allow_diagonal

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 初始化地图
        self.grid = np.zeros((height, width), dtype=np.int8)

        # 设置起点和终点
        self.start = start if start else (0, 0)
        self.goal = goal if goal else (height - 1, width - 1)

        # 生成障碍物
        self._generate_obstacles()

        # 确保起点和终点可达
        self._ensure_path_exists()

    def _generate_obstacles(self):
        """随机生成障碍物"""
        total_cells = self.width * self.height
        num_obstacles = int(total_cells * self.obstacle_density)

        # 获取所有可用位置（排除起点和终点）
        available_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if (y, x) != self.start and (y, x) != self.goal:
                    available_positions.append((y, x))

        # 随机选择障碍物位置
        obstacle_positions = random.sample(available_positions,
                                          min(num_obstacles, len(available_positions)))

        for pos in obstacle_positions:
            self.grid[pos[0], pos[1]] = self.OBSTACLE

    def _ensure_path_exists(self):
        """确保起点到终点存在可行路径，使用BFS检查"""
        if self._path_exists():
            return

        # 如果不存在路径，清除一些障碍物直到路径存在
        while not self._path_exists():
            # 随机移除一些障碍物
            obstacles = np.argwhere(self.grid == self.OBSTACLE)
            if len(obstacles) == 0:
                break
            # 移除10%的障碍物
            num_to_remove = max(1, len(obstacles) // 10)
            indices = np.random.choice(len(obstacles), num_to_remove, replace=False)
            for idx in indices:
                y, x = obstacles[idx]
                self.grid[y, x] = self.FREE

    def _path_exists(self) -> bool:
        """使用BFS检查是否存在从起点到终点的路径"""
        from collections import deque

        visited = set()
        queue = deque([self.start])
        visited.add(self.start)

        directions = self.DIRECTIONS_8 if self.allow_diagonal else self.DIRECTIONS_4

        while queue:
            current = queue.popleft()
            if current == self.goal:
                return True

            for dy, dx in directions:
                ny, nx = current[0] + dy, current[1] + dx
                if self.is_valid_position(ny, nx) and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append((ny, nx))

        return False

    def is_valid_position(self, y: int, x: int) -> bool:
        """检查位置是否有效（在地图内且不是障碍物）"""
        return (0 <= y < self.height and
                0 <= x < self.width and
                self.grid[y, x] != self.OBSTACLE)

    def is_obstacle(self, y: int, x: int) -> bool:
        """检查位置是否是障碍物"""
        return self.grid[y, x] == self.OBSTACLE

    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取某位置的所有有效邻居"""
        y, x = position
        neighbors = []
        directions = self.DIRECTIONS_8 if self.allow_diagonal else self.DIRECTIONS_4

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if self.is_valid_position(ny, nx):
                # 对角线移动时检查是否被障碍物阻挡
                if self.allow_diagonal and abs(dy) + abs(dx) == 2:
                    # 检查两个相邻格子是否都有障碍物
                    if self.grid[y + dy, x] == self.OBSTACLE and self.grid[y, x + dx] == self.OBSTACLE:
                        continue
                neighbors.append((ny, nx))

        return neighbors

    def get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """计算两点之间的距离（曼哈顿距离或欧几里得距离）"""
        if self.allow_diagonal:
            # 欧几里得距离
            return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        else:
            # 曼哈顿距离
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_heuristic(self, position: Tuple[int, int]) -> float:
        """获取从当前位置到目标的启发式估计"""
        return self.get_distance(position, self.goal)

    def copy(self) -> 'GridMap':
        """创建地图的副本"""
        new_map = GridMap.__new__(GridMap)
        new_map.width = self.width
        new_map.height = self.height
        new_map.obstacle_density = self.obstacle_density
        new_map.allow_diagonal = self.allow_diagonal
        new_map.grid = self.grid.copy()
        new_map.start = self.start
        new_map.goal = self.goal
        return new_map

    def __str__(self) -> str:
        """返回地图的字符串表示"""
        symbols = {
            self.FREE: '.',
            self.OBSTACLE: '#',
            self.START: 'S',
            self.GOAL: 'G'
        }

        result = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (y, x) == self.start:
                    row.append(symbols[self.START])
                elif (y, x) == self.goal:
                    row.append(symbols[self.GOAL])
                else:
                    row.append(symbols[self.grid[y, x]])
            result.append(' '.join(row))

        return '\n'.join(result)


def create_test_maps() -> List[GridMap]:
    """创建一组测试地图"""
    maps = []

    # 30x30 地图，不同障碍物密度
    for density in [0.1, 0.2, 0.3]:
        maps.append(GridMap(30, 30, obstacle_density=density, seed=42))

    # 50x50 地图
    maps.append(GridMap(50, 50, obstacle_density=0.2, seed=42))

    return maps


if __name__ == "__main__":
    # 测试地图生成
    grid_map = GridMap(20, 20, obstacle_density=0.2, seed=42)
    print("Generated 20x20 map with 20% obstacles:")
    print(grid_map)
    print(f"\nStart: {grid_map.start}, Goal: {grid_map.goal}")
    print(f"Path exists: {grid_map._path_exists()}")
