"""
MCTS树搜索实现
包含选择、扩展、模拟、回溯四个核心步骤
"""

import random
import math
from typing import List, Tuple, Optional, Callable
from .node import MCTSNode
from environment.grid import GridMap


class MCTSPathFinder:
    """使用MCTS进行路径规划"""

    def __init__(self, grid_map: GridMap,
                 exploration_constant: float = 1.414,
                 max_iterations: int = 10000,
                 max_rollout_depth: int = 100,
                 reward_goal: float = 100.0,
                 reward_step: float = -0.1,
                 reward_closer: float = 1.0):
        """
        初始化MCTS路径规划器

        Args:
            grid_map: 栅格地图
            exploration_constant: UCB1探索常数
            max_iterations: 最大迭代次数
            max_rollout_depth: 最大rollout深度
            reward_goal: 到达目标的奖励
            reward_step: 每步的惩罚
            reward_closer: 接近目标的奖励系数
        """
        self.grid_map = grid_map
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
        self.max_rollout_depth = max_rollout_depth
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_closer = reward_closer

        # 统计信息
        self.iterations_used = 0
        self.nodes_created = 0

        # 回调函数用于可视化
        self.on_iteration: Optional[Callable[[int, MCTSNode], None]] = None

    def search(self) -> Tuple[List[Tuple[int, int]], MCTSNode]:
        """
        执行MCTS搜索

        Returns:
            (最优路径, 根节点) 用于可视化
        """
        # 创建根节点
        root = MCTSNode(self.grid_map.start)
        root.untried_actions = self._get_valid_actions(root.position)
        self.nodes_created = 1

        best_path = None
        best_path_length = float('inf')

        for iteration in range(self.max_iterations):
            self.iterations_used = iteration + 1

            # * 1. Selection - 选择
            node = self._select(root)

            # * 2. Expansion - 扩展
            if not node.is_terminal and not node.is_fully_expanded:
                node = self._expand(node)

            # * 3. Simulation - 模拟 (Rollout)
            reward, reached_goal, path = self._simulate(node)

            # * 4. Backpropagation - 回溯
            self._backpropagate(node, reward)

            # 更新最佳路径
            if reached_goal:
                full_path = node.get_path_positions() + path[1:]  # 避免重复当前位置
                if len(full_path) < best_path_length:
                    best_path = full_path
                    best_path_length = len(full_path)

            # 调用可视化回调
            if self.on_iteration is not None:
                self.on_iteration(iteration, root)

            # 早期终止：如果找到足够好的解
            if best_path is not None and iteration > 1000:
                # 检查是否收敛
                if self._is_converged(root):
                    break

        # 如果没有通过rollout找到路径，尝试从树中提取最佳路径
        if best_path is None:
            best_path = self._extract_best_path(root)

        return best_path, root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        选择阶段：从根节点向下选择，直到找到未完全扩展的节点

        使用UCB1策略平衡探索和利用
        """
        while node.is_fully_expanded and node.children:
            if node.is_terminal:
                break
            node = node.best_child(self.exploration_constant)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        扩展阶段：从未尝试的动作中选择一个，创建新的子节点
        """
        if not node.untried_actions:
            return node

        # 随机选择一个未尝试的动作
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # 计算新位置
        new_position = (node.position[0] + action[0],
                       node.position[1] + action[1])

        # 创建子节点
        child = node.add_child(new_position, action)
        child.untried_actions = self._get_valid_actions(new_position,
                                                        exclude=node.position)
        self.nodes_created += 1

        # 检查是否到达目标
        if new_position == self.grid_map.goal:
            child.is_terminal = True

        return child

    def _simulate(self, node: MCTSNode) -> Tuple[float, bool, List[Tuple[int, int]]]:
        """
        模拟阶段：使用随机策略进行rollout

        Returns:
            (奖励值, 是否到达目标, 模拟路径)
        """
        current_position = node.position
        path = [current_position]
        visited = set(node.get_path_positions())
        visited.add(current_position)

        total_reward = 0.0
        initial_distance = self.grid_map.get_heuristic(current_position)

        for step in range(self.max_rollout_depth):
            # 检查是否到达目标
            if current_position == self.grid_map.goal:
                total_reward += self.reward_goal
                return total_reward, True, path

            # 获取可用动作（排除已访问位置）
            neighbors = self.grid_map.get_neighbors(current_position)
            valid_neighbors = [n for n in neighbors if n not in visited]

            if not valid_neighbors:
                # 死路，给予惩罚
                total_reward -= 10.0
                break

            # ! 随机策略选择下一步（带有启发式偏向）
            next_position = self._rollout_policy(current_position, valid_neighbors)

            # 计算奖励
            total_reward += self.reward_step
            new_distance = self.grid_map.get_heuristic(next_position)
            if new_distance < initial_distance:
                total_reward += self.reward_closer * (initial_distance - new_distance)
            initial_distance = new_distance

            visited.add(next_position)
            path.append(next_position)
            current_position = next_position

        # 未到达目标，根据距离给予部分奖励
        final_distance = self.grid_map.get_heuristic(current_position)
        max_distance = self.grid_map.get_heuristic(self.grid_map.start)
        progress_reward = (max_distance - final_distance) / max_distance * 10.0
        total_reward += progress_reward

        return total_reward, False, path

    def _rollout_policy(self, current: Tuple[int, int],
                        neighbors: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Rollout策略：带有启发式偏向的随机策略

        以一定概率选择更接近目标的位置，否则随机选择
        """
        # 80%概率使用贪心策略，20%随机探索
        if random.random() < 0.8:
            # 选择距离目标最近的邻居
            return min(neighbors, key=lambda n: self.grid_map.get_heuristic(n))
        else:
            return random.choice(neighbors)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        回溯阶段：从当前节点向上更新所有祖先节点的统计信息
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def _get_valid_actions(self, position: Tuple[int, int],
                          exclude: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """获取某位置的所有有效动作"""
        neighbors = self.grid_map.get_neighbors(position)
        actions = []

        for neighbor in neighbors:
            if exclude and neighbor == exclude:
                continue
            dy = neighbor[0] - position[0]
            dx = neighbor[1] - position[1]
            actions.append((dy, dx))

        return actions

    def _is_converged(self, root: MCTSNode, threshold: float = 0.01) -> bool:
        """检查搜索是否收敛"""
        if not root.children:
            return False

        # 检查最佳子节点的访问比例是否足够高
        best_child = root.best_action_child()
        visit_ratio = best_child.visits / root.visits

        return visit_ratio > (1 - threshold)

    def _extract_best_path(self, root: MCTSNode) -> List[Tuple[int, int]]:
        """从MCTS树中提取最佳路径"""
        path = [root.position]
        node = root

        while node.children:
            # 选择访问次数最多的子节点
            node = node.best_action_child()
            path.append(node.position)

            if node.position == self.grid_map.goal:
                break

        return path

    def get_statistics(self) -> dict:
        """获取搜索统计信息"""
        return {
            'iterations': self.iterations_used,
            'nodes_created': self.nodes_created,
        }
