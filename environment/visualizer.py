"""
可视化模块
动态展示MCTS搜索树的生长过程和最终路径
"""

import platform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import List, Tuple, Optional, Dict
from environment.grid import GridMap
from mcts.node import MCTSNode

# 跨平台中文字体设置
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['LXGW WenKai', 'Noto Serif CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class PathVisualizer:
    """路径可视化器"""

    def __init__(self, grid_map: GridMap, figsize: Tuple[int, int] = (12, 10)):
        """
        初始化可视化器

        Args:
            grid_map: 栅格地图
            figsize: 图形大小
        """
        self.grid_map = grid_map
        self.figsize = figsize

        # 颜色定义
        self.colors = {
            'free': '#FFFFFF',       # 白色 - 可通行
            'obstacle': '#2C3E50',   # 深蓝灰 - 障碍物
            'start': '#27AE60',      # 绿色 - 起点
            'goal': '#E74C3C',       # 红色 - 终点
            'path': '#3498DB',       # 蓝色 - 路径
            'explored': '#F39C12',   # 橙色 - 已探索
            'mcts_node': '#9B59B6',  # 紫色 - MCTS节点
        }

    def plot_map(self, ax: Optional[plt.Axes] = None,
                 title: str = "Grid Map") -> plt.Axes:
        """绘制基础地图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # 创建地图图像
        map_image = np.zeros((self.grid_map.height, self.grid_map.width, 3))

        for y in range(self.grid_map.height):
            for x in range(self.grid_map.width):
                if self.grid_map.grid[y, x] == GridMap.OBSTACLE:
                    map_image[y, x] = self._hex_to_rgb(self.colors['obstacle'])
                else:
                    map_image[y, x] = self._hex_to_rgb(self.colors['free'])

        ax.imshow(map_image, origin='upper')

        # 标记起点和终点
        start_y, start_x = self.grid_map.start
        goal_y, goal_x = self.grid_map.goal

        ax.plot(start_x, start_y, 'o', color=self.colors['start'],
                markersize=15, label='Start', markeredgecolor='white', markeredgewidth=2)
        ax.plot(goal_x, goal_y, 's', color=self.colors['goal'],
                markersize=15, label='Goal', markeredgecolor='white', markeredgewidth=2)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right')

        # 添加网格线
        ax.set_xticks(np.arange(-0.5, self.grid_map.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_map.height, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        return ax

    def plot_path(self, path: List[Tuple[int, int]],
                  ax: Optional[plt.Axes] = None,
                  color: str = None,
                  label: str = "Path",
                  linewidth: float = 3,
                  alpha: float = 0.8) -> plt.Axes:
        """绘制路径"""
        if ax is None:
            ax = self.plot_map()

        if not path:
            return ax

        if color is None:
            color = self.colors['path']

        # 绘制路径线
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, '-', color=color, linewidth=linewidth,
                alpha=alpha, label=label, zorder=5)

        # 绘制路径点
        ax.scatter(path_x[1:-1], path_y[1:-1], c=color, s=30, alpha=alpha, zorder=6)

        ax.legend(loc='upper right')
        return ax

    def plot_mcts_tree(self, root: MCTSNode,
                       ax: Optional[plt.Axes] = None,
                       max_depth: int = 10,
                       show_values: bool = True) -> plt.Axes:
        """
        绘制MCTS搜索树

        Args:
            root: MCTS根节点
            ax: matplotlib轴
            max_depth: 最大显示深度
            show_values: 是否显示节点值
        """
        if ax is None:
            ax = self.plot_map()

        # 收集所有节点
        nodes_by_depth = self._collect_nodes(root, max_depth)

        # 创建热力图显示访问次数
        visit_map = np.zeros((self.grid_map.height, self.grid_map.width))
        value_map = np.zeros((self.grid_map.height, self.grid_map.width))

        for depth, nodes in nodes_by_depth.items():
            for node in nodes:
                y, x = node.position
                visit_map[y, x] = max(visit_map[y, x], node.visits)
                value_map[y, x] = max(value_map[y, x], node.average_value)

        # 归一化并绘制热力图
        if visit_map.max() > 0:
            visit_map_normalized = visit_map / visit_map.max()
            # 创建自定义颜色映射
            cmap = LinearSegmentedColormap.from_list(
                'visits', ['#FFFFFF', '#FFE5B4', '#FFA500', '#FF4500'], N=256
            )
            # 只在有访问的地方显示热力
            masked_visits = np.ma.masked_where(visit_map == 0, visit_map_normalized)
            im = ax.imshow(masked_visits, cmap=cmap, alpha=0.6, origin='upper')
            plt.colorbar(im, ax=ax, label='Normalized Visit Count', shrink=0.6)

        # 绘制树的边
        for depth, nodes in nodes_by_depth.items():
            for node in nodes:
                if node.parent is not None:
                    parent_y, parent_x = node.parent.position
                    node_y, node_x = node.position
                    ax.plot([parent_x, node_x], [parent_y, node_y],
                           'b-', alpha=0.3, linewidth=1)

        # 标记高访问节点
        for depth, nodes in nodes_by_depth.items():
            for node in nodes:
                if node.visits > root.visits * 0.1:  # 访问次数超过10%的节点
                    y, x = node.position
                    ax.plot(x, y, 'o', color=self.colors['mcts_node'],
                           markersize=8, alpha=0.7)
                    if show_values:
                        ax.annotate(f'{node.visits}',
                                   (x, y), textcoords="offset points",
                                   xytext=(0, 5), ha='center', fontsize=6)

        ax.set_title(f"MCTS Tree (Total iterations: {root.visits})",
                    fontsize=14, fontweight='bold')
        return ax

    def _collect_nodes(self, root: MCTSNode,
                       max_depth: int) -> Dict[int, List[MCTSNode]]:
        """收集节点按深度分组"""
        nodes_by_depth = {}

        def dfs(node: MCTSNode, depth: int):
            if depth > max_depth:
                return
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)
            for child in node.children:
                dfs(child, depth + 1)

        dfs(root, 0)
        return nodes_by_depth

    def compare_paths(self, paths: Dict[str, List[Tuple[int, int]]],
                      title: str = "Path Comparison") -> plt.Figure:
        """比较多个算法的路径"""
        fig, ax = plt.subplots(figsize=self.figsize)
        self.plot_map(ax, title)

        colors = ['#3498DB', '#E74C3C', '#27AE60', '#9B59B6', '#F39C12']

        for i, (name, path) in enumerate(paths.items()):
            if path:
                color = colors[i % len(colors)]
                self.plot_path(path, ax, color=color, label=f"{name} ({len(path)} steps)")

        ax.legend(loc='upper right')
        plt.tight_layout()
        return fig

    def create_animation(self, grid_map: GridMap,
                         iteration_data: List[Tuple[int, MCTSNode]],
                         interval: int = 100) -> animation.FuncAnimation:
        """
        创建MCTS搜索过程的动画

        Args:
            grid_map: 栅格地图
            iteration_data: 每次迭代的数据 (iteration, root)
            interval: 帧间隔(毫秒)

        Returns:
            动画对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        def init():
            ax.clear()
            self.plot_map(ax, "MCTS Search Animation")
            return []

        def update(frame):
            ax.clear()
            iteration, root = iteration_data[frame]
            self.plot_map(ax, f"MCTS Iteration: {iteration}")
            self.plot_mcts_tree(root, ax, show_values=False)
            return []

        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=len(iteration_data),
            interval=interval, blit=False
        )

        return anim

    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def visualize_results(grid_map: GridMap,
                      mcts_path: Optional[List[Tuple[int, int]]],
                      astar_path: Optional[List[Tuple[int, int]]],
                      dijkstra_path: Optional[List[Tuple[int, int]]],
                      mcts_root: Optional[MCTSNode] = None,
                      save_path: Optional[str] = None):
    """
    综合可视化结果

    Args:
        grid_map: 栅格地图
        mcts_path: MCTS路径
        astar_path: A*路径
        dijkstra_path: Dijkstra路径
        mcts_root: MCTS根节点（用于显示搜索树）
        save_path: 保存路径
    """
    visualizer = PathVisualizer(grid_map)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 子图1: 基础地图
    visualizer.plot_map(axes[0, 0], "Grid Map")

    # 子图2: MCTS路径和搜索树
    visualizer.plot_map(axes[0, 1], "MCTS Path")
    if mcts_root:
        visualizer.plot_mcts_tree(mcts_root, axes[0, 1], show_values=False)
    if mcts_path:
        visualizer.plot_path(mcts_path, axes[0, 1], color='#E74C3C',
                            label=f'MCTS ({len(mcts_path)} steps)', linewidth=4)

    # 子图3: A*路径
    visualizer.plot_map(axes[1, 0], "A* Path")
    if astar_path:
        visualizer.plot_path(astar_path, axes[1, 0], color='#3498DB',
                            label=f'A* ({len(astar_path)} steps)')

    # 子图4: 路径比较
    visualizer.plot_map(axes[1, 1], "Path Comparison")
    if mcts_path:
        visualizer.plot_path(mcts_path, axes[1, 1], color='#E74C3C',
                            label=f'MCTS ({len(mcts_path)} steps)', linewidth=3)
    if astar_path:
        visualizer.plot_path(astar_path, axes[1, 1], color='#3498DB',
                            label=f'A* ({len(astar_path)} steps)', linewidth=2)
    if dijkstra_path:
        visualizer.plot_path(dijkstra_path, axes[1, 1], color='#27AE60',
                            label=f'Dijkstra ({len(dijkstra_path)} steps)', linewidth=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # 测试可视化
    from environment.grid import GridMap

    grid_map = GridMap(20, 20, obstacle_density=0.2, seed=42)
    visualizer = PathVisualizer(grid_map)

    fig, ax = plt.subplots(figsize=(10, 10))
    visualizer.plot_map(ax)
    plt.show()
