"""
MCTS树节点实现
"""

import math
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.grid import GridMap


class MCTSNode:
    """MCTS树节点类"""

    def __init__(self, position: Tuple[int, int],
                 parent: Optional['MCTSNode'] = None,
                 action: Optional[Tuple[int, int]] = None):
        """
        初始化MCTS节点

        Args:
            position: 当前位置坐标 (y, x)
            parent: 父节点
            action: 从父节点到达此节点的动作（移动方向）
        """
        self.position = position
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []

        # * 核心统计信息
        self.visits = 0          # 访问次数 N(s)
        self.value = 0.0         # 累计价值 Q(s)
        self.untried_actions: List[Tuple[int, int]] = []  # 未尝试的动作

    @property
    def is_fully_expanded(self) -> bool:
        """检查节点是否完全扩展"""
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        """检查是否是终端节点（需要在外部设置）"""
        return hasattr(self, '_is_terminal') and self._is_terminal

    @is_terminal.setter
    def is_terminal(self, value: bool):
        self._is_terminal = value

    @property
    def average_value(self) -> float:
        """获取平均价值"""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """
        计算UCB1值

        UCB1 = Q(s)/N(s) + c * sqrt(ln(N(parent)) / N(s))

        Args:
            exploration_constant: 探索常数 c，默认sqrt(2)

        Returns:
            UCB1值
        """
        if self.visits == 0:
            return float('inf')  # 未访问节点优先探索

        if self.parent is None:
            return self.average_value

        exploitation = self.average_value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """
        选择最佳子节点（UCB1最大）

        Args:
            exploration_constant: 探索常数

        Returns:
            UCB1值最大的子节点
        """
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))

    def best_action_child(self) -> 'MCTSNode':
        """
        选择最佳动作对应的子节点（访问次数最多）
        用于最终决策

        Returns:
            访问次数最多的子节点
        """
        return max(self.children, key=lambda c: c.visits)

    def add_child(self, position: Tuple[int, int],
                  action: Tuple[int, int]) -> 'MCTSNode':
        """
        添加子节点

        Args:
            position: 子节点位置
            action: 到达子节点的动作

        Returns:
            新创建的子节点
        """
        child = MCTSNode(position, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, reward: float):
        """
        更新节点统计信息（回溯时调用）

        Args:
            reward: 获得的奖励
        """
        self.visits += 1
        self.value += reward

    def get_path_to_root(self) -> List['MCTSNode']:
        """获取从当前节点到根节点的路径"""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]  # 反转，从根到当前

    def get_path_positions(self) -> List[Tuple[int, int]]:
        """获取从根节点到当前节点的位置序列"""
        return [node.position for node in self.get_path_to_root()]

    def __repr__(self) -> str:
        return (f"MCTSNode(pos={self.position}, "
                f"visits={self.visits}, "
                f"value={self.value:.2f}, "
                f"avg={self.average_value:.3f})")
