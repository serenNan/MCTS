"""
性能测试和对比分析模块
测试MCTS、A*、Dijkstra在不同地图复杂度下的表现
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tabulate import tabulate

from environment.grid import GridMap
from mcts.tree import MCTSPathFinder
from algorithms.astar import AStarPathFinder
from algorithms.dijkstra import DijkstraPathFinder


class Benchmark:
    """性能测试类"""

    def __init__(self):
        self.results: List[Dict] = []

    def run_single_test(self, grid_map: GridMap,
                        mcts_iterations: int = 5000) -> Dict:
        """
        在单个地图上运行所有算法

        Args:
            grid_map: 栅格地图
            mcts_iterations: MCTS最大迭代次数

        Returns:
            测试结果字典
        """
        result = {
            'map_size': f"{grid_map.width}x{grid_map.height}",
            'obstacle_density': grid_map.obstacle_density,
        }

        # MCTS
        mcts = MCTSPathFinder(grid_map, max_iterations=mcts_iterations)
        start_time = time.time()
        mcts_path, mcts_root = mcts.search()
        mcts_time = time.time() - start_time

        result['mcts'] = {
            'time': mcts_time,
            'path_length': len(mcts_path) if mcts_path else None,
            'iterations': mcts.iterations_used,
            'nodes_created': mcts.nodes_created,
            'found_path': mcts_path is not None,
            'path': mcts_path,
            'root': mcts_root,
        }

        # A*
        astar = AStarPathFinder(grid_map)
        start_time = time.time()
        astar_path, astar_stats = astar.search()
        astar_time = time.time() - start_time

        result['astar'] = {
            'time': astar_time,
            'path_length': len(astar_path) if astar_path else None,
            'nodes_explored': astar_stats['nodes_explored'],
            'found_path': astar_path is not None,
            'path': astar_path,
        }

        # Dijkstra
        dijkstra = DijkstraPathFinder(grid_map)
        start_time = time.time()
        dijkstra_path, dijkstra_stats = dijkstra.search()
        dijkstra_time = time.time() - start_time

        result['dijkstra'] = {
            'time': dijkstra_time,
            'path_length': len(dijkstra_path) if dijkstra_path else None,
            'nodes_explored': dijkstra_stats['nodes_explored'],
            'found_path': dijkstra_path is not None,
            'path': dijkstra_path,
        }

        self.results.append(result)
        return result

    def run_comprehensive_test(self, num_trials: int = 3) -> List[Dict]:
        """
        运行综合测试

        Args:
            num_trials: 每种配置的测试次数

        Returns:
            所有测试结果
        """
        # 测试配置
        map_sizes = [(30, 30), (50, 50)]
        obstacle_densities = [0.1, 0.2, 0.3]

        all_results = []

        for size in map_sizes:
            for density in obstacle_densities:
                print(f"\n测试配置: 地图大小={size}, 障碍物密度={density}")

                for trial in range(num_trials):
                    seed = 42 + trial  # 不同的随机种子
                    grid_map = GridMap(size[0], size[1],
                                      obstacle_density=density,
                                      seed=seed)

                    result = self.run_single_test(grid_map)
                    result['trial'] = trial
                    all_results.append(result)

                    # 打印简要结果
                    print(f"  试验 {trial + 1}: MCTS={result['mcts']['time']:.3f}s, "
                          f"A*={result['astar']['time']:.3f}s, "
                          f"Dijkstra={result['dijkstra']['time']:.3f}s")

        return all_results

    def print_summary(self):
        """打印测试结果摘要"""
        if not self.results:
            print("没有测试结果")
            return

        # 按地图大小和障碍物密度分组
        grouped = {}
        for r in self.results:
            key = (r['map_size'], r['obstacle_density'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)

        # 计算平均值并生成表格
        table_data = []
        headers = ['地图大小', '障碍物密度',
                   'MCTS时间(s)', 'MCTS路径长度', 'MCTS节点数',
                   'A*时间(s)', 'A*路径长度', 'A*探索节点',
                   'Dijkstra时间(s)', 'Dijkstra路径长度', 'Dijkstra探索节点']

        for (size, density), results in sorted(grouped.items()):
            # 计算平均值
            mcts_times = [r['mcts']['time'] for r in results]
            mcts_lengths = [r['mcts']['path_length'] for r in results if r['mcts']['path_length']]
            mcts_nodes = [r['mcts']['nodes_created'] for r in results]

            astar_times = [r['astar']['time'] for r in results]
            astar_lengths = [r['astar']['path_length'] for r in results if r['astar']['path_length']]
            astar_explored = [r['astar']['nodes_explored'] for r in results]

            dijkstra_times = [r['dijkstra']['time'] for r in results]
            dijkstra_lengths = [r['dijkstra']['path_length'] for r in results if r['dijkstra']['path_length']]
            dijkstra_explored = [r['dijkstra']['nodes_explored'] for r in results]

            row = [
                size, f"{density:.0%}",
                f"{np.mean(mcts_times):.3f}",
                f"{np.mean(mcts_lengths):.1f}" if mcts_lengths else "N/A",
                f"{np.mean(mcts_nodes):.0f}",
                f"{np.mean(astar_times):.3f}",
                f"{np.mean(astar_lengths):.1f}" if astar_lengths else "N/A",
                f"{np.mean(astar_explored):.0f}",
                f"{np.mean(dijkstra_times):.3f}",
                f"{np.mean(dijkstra_lengths):.1f}" if dijkstra_lengths else "N/A",
                f"{np.mean(dijkstra_explored):.0f}",
            ]
            table_data.append(row)

        print("\n" + "=" * 100)
        print("性能测试结果摘要")
        print("=" * 100)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))

    def plot_comparison(self, save_path: str = None):
        """绘制性能对比图"""
        if not self.results:
            print("没有测试结果")
            return

        # 提取数据
        grouped = {}
        for r in self.results:
            key = (r['map_size'], r['obstacle_density'])
            if key not in grouped:
                grouped[key] = {'mcts': [], 'astar': [], 'dijkstra': []}
            grouped[key]['mcts'].append(r['mcts'])
            grouped[key]['astar'].append(r['astar'])
            grouped[key]['dijkstra'].append(r['dijkstra'])

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        configs = list(grouped.keys())
        x = np.arange(len(configs))
        width = 0.25

        # 1. 计算时间对比
        ax = axes[0, 0]
        mcts_times = [np.mean([r['time'] for r in grouped[c]['mcts']]) for c in configs]
        astar_times = [np.mean([r['time'] for r in grouped[c]['astar']]) for c in configs]
        dijkstra_times = [np.mean([r['time'] for r in grouped[c]['dijkstra']]) for c in configs]

        ax.bar(x - width, mcts_times, width, label='MCTS', color='#E74C3C')
        ax.bar(x, astar_times, width, label='A*', color='#3498DB')
        ax.bar(x + width, dijkstra_times, width, label='Dijkstra', color='#27AE60')

        ax.set_ylabel('时间 (秒)')
        ax.set_title('计算时间对比')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c[0]}\n{c[1]:.0%}" for c in configs])
        ax.legend()
        ax.set_xlabel('地图配置 (大小, 障碍物密度)')

        # 2. 路径长度对比
        ax = axes[0, 1]
        mcts_lengths = [np.mean([r['path_length'] for r in grouped[c]['mcts'] if r['path_length']]) for c in configs]
        astar_lengths = [np.mean([r['path_length'] for r in grouped[c]['astar'] if r['path_length']]) for c in configs]
        dijkstra_lengths = [np.mean([r['path_length'] for r in grouped[c]['dijkstra'] if r['path_length']]) for c in configs]

        ax.bar(x - width, mcts_lengths, width, label='MCTS', color='#E74C3C')
        ax.bar(x, astar_lengths, width, label='A*', color='#3498DB')
        ax.bar(x + width, dijkstra_lengths, width, label='Dijkstra', color='#27AE60')

        ax.set_ylabel('路径长度 (步数)')
        ax.set_title('路径长度对比')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c[0]}\n{c[1]:.0%}" for c in configs])
        ax.legend()
        ax.set_xlabel('地图配置 (大小, 障碍物密度)')

        # 3. 搜索节点数对比
        ax = axes[1, 0]
        mcts_nodes = [np.mean([r['nodes_created'] for r in grouped[c]['mcts']]) for c in configs]
        astar_nodes = [np.mean([r['nodes_explored'] for r in grouped[c]['astar']]) for c in configs]
        dijkstra_nodes = [np.mean([r['nodes_explored'] for r in grouped[c]['dijkstra']]) for c in configs]

        ax.bar(x - width, mcts_nodes, width, label='MCTS', color='#E74C3C')
        ax.bar(x, astar_nodes, width, label='A*', color='#3498DB')
        ax.bar(x + width, dijkstra_nodes, width, label='Dijkstra', color='#27AE60')

        ax.set_ylabel('节点数')
        ax.set_title('搜索节点数对比')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c[0]}\n{c[1]:.0%}" for c in configs])
        ax.legend()
        ax.set_xlabel('地图配置 (大小, 障碍物密度)')

        # 4. 路径质量对比 (相对于最优路径)
        ax = axes[1, 1]
        # A*和Dijkstra找到的是最优路径
        mcts_quality = []
        for c in configs:
            mcts_lens = [r['path_length'] for r in grouped[c]['mcts'] if r['path_length']]
            astar_lens = [r['path_length'] for r in grouped[c]['astar'] if r['path_length']]
            if mcts_lens and astar_lens:
                # 计算MCTS路径相对于A*路径的比率
                ratio = np.mean(mcts_lens) / np.mean(astar_lens)
                mcts_quality.append(ratio)
            else:
                mcts_quality.append(1.0)

        ax.bar(x, mcts_quality, width * 2, label='MCTS/A*', color='#9B59B6')
        ax.axhline(y=1.0, color='r', linestyle='--', label='最优 (A*)')

        ax.set_ylabel('路径长度比率')
        ax.set_title('MCTS路径质量 (相对于A*最优路径)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c[0]}\n{c[1]:.0%}" for c in configs])
        ax.legend()
        ax.set_xlabel('地图配置 (大小, 障碍物密度)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到 {save_path}")

        plt.show()


def run_benchmark(num_trials: int = 3, save_results: bool = True):
    """运行基准测试"""
    print("开始性能测试...")
    print("=" * 60)

    benchmark = Benchmark()
    benchmark.run_comprehensive_test(num_trials=num_trials)
    benchmark.print_summary()

    if save_results:
        benchmark.plot_comparison(save_path='benchmark_results.png')

    return benchmark


if __name__ == "__main__":
    run_benchmark(num_trials=3)
