"""
MCTS路径规划主程序
蒙特卡洛树搜索算法解决2D栅格地图中的静态路径规划问题
"""

import argparse
import sys
import time
import matplotlib.pyplot as plt

from environment.grid import GridMap
from environment.visualizer import PathVisualizer, visualize_results
from mcts.tree import MCTSPathFinder
from algorithms.astar import AStarPathFinder
from algorithms.dijkstra import DijkstraPathFinder


def run_demo(map_size: int = 30,
             obstacle_density: float = 0.2,
             mcts_iterations: int = 5000,
             seed: int = 42,
             visualize: bool = True,
             save_figure: bool = False):
    """
    运行MCTS路径规划演示

    Args:
        map_size: 地图大小
        obstacle_density: 障碍物密度 (0.1-0.3)
        mcts_iterations: MCTS最大迭代次数
        seed: 随机种子
        visualize: 是否显示可视化
        save_figure: 是否保存图片
    """
    print("=" * 60)
    print("MCTS路径规划演示")
    print("=" * 60)

    # 1. 创建地图
    print(f"\n[1] 创建 {map_size}x{map_size} 栅格地图 (障碍物密度: {obstacle_density:.0%})")
    grid_map = GridMap(map_size, map_size,
                       obstacle_density=obstacle_density,
                       seed=seed)
    print(f"    起点: {grid_map.start}")
    print(f"    终点: {grid_map.goal}")

    # 2. 运行MCTS
    print(f"\n[2] 运行MCTS算法 (最大迭代次数: {mcts_iterations})")
    mcts = MCTSPathFinder(grid_map, max_iterations=mcts_iterations)

    start_time = time.time()
    mcts_path, mcts_root = mcts.search()
    mcts_time = time.time() - start_time

    if mcts_path:
        print(f"    ✓ 找到路径! 长度: {len(mcts_path)} 步")
    else:
        print("    ✗ 未找到路径")
    print(f"    迭代次数: {mcts.iterations_used}")
    print(f"    创建节点数: {mcts.nodes_created}")
    print(f"    计算时间: {mcts_time:.3f} 秒")

    # 3. 运行A*
    print("\n[3] 运行A*算法")
    astar = AStarPathFinder(grid_map)

    start_time = time.time()
    astar_path, astar_stats = astar.search()
    astar_time = time.time() - start_time

    if astar_path:
        print(f"    ✓ 找到路径! 长度: {len(astar_path)} 步")
    else:
        print("    ✗ 未找到路径")
    print(f"    探索节点数: {astar_stats['nodes_explored']}")
    print(f"    计算时间: {astar_time:.3f} 秒")

    # 4. 运行Dijkstra
    print("\n[4] 运行Dijkstra算法")
    dijkstra = DijkstraPathFinder(grid_map)

    start_time = time.time()
    dijkstra_path, dijkstra_stats = dijkstra.search()
    dijkstra_time = time.time() - start_time

    if dijkstra_path:
        print(f"    ✓ 找到路径! 长度: {len(dijkstra_path)} 步")
    else:
        print("    ✗ 未找到路径")
    print(f"    探索节点数: {dijkstra_stats['nodes_explored']}")
    print(f"    计算时间: {dijkstra_time:.3f} 秒")

    # 5. 对比分析
    print("\n" + "=" * 60)
    print("对比分析结果")
    print("=" * 60)
    print(f"{'算法':<15} {'路径长度':<12} {'搜索节点':<12} {'计算时间(s)':<12}")
    print("-" * 60)

    mcts_len = len(mcts_path) if mcts_path else "N/A"
    astar_len = len(astar_path) if astar_path else "N/A"
    dijkstra_len = len(dijkstra_path) if dijkstra_path else "N/A"

    print(f"{'MCTS':<15} {str(mcts_len):<12} {mcts.nodes_created:<12} {mcts_time:<12.3f}")
    print(f"{'A*':<15} {str(astar_len):<12} {astar_stats['nodes_explored']:<12} {astar_time:<12.3f}")
    print(f"{'Dijkstra':<15} {str(dijkstra_len):<12} {dijkstra_stats['nodes_explored']:<12} {dijkstra_time:<12.3f}")

    # 路径质量分析
    if mcts_path and astar_path:
        quality_ratio = len(mcts_path) / len(astar_path)
        print(f"\nMCTS路径质量: {quality_ratio:.2%} (相对于A*最优路径)")
        if quality_ratio <= 1.1:
            print("  → 接近最优路径!")
        elif quality_ratio <= 1.3:
            print("  → 路径质量良好")
        else:
            print("  → 可以增加迭代次数以提高路径质量")

    # 6. 可视化
    if visualize:
        print("\n[5] 生成可视化...")
        save_path = 'mcts_demo_result.png' if save_figure else None
        visualize_results(grid_map, mcts_path, astar_path, dijkstra_path,
                         mcts_root, save_path)

    return {
        'grid_map': grid_map,
        'mcts': {'path': mcts_path, 'root': mcts_root, 'time': mcts_time},
        'astar': {'path': astar_path, 'time': astar_time},
        'dijkstra': {'path': dijkstra_path, 'time': dijkstra_time}
    }


def run_mcts_animation(map_size: int = 20,
                       obstacle_density: float = 0.15,
                       iterations: int = 500,
                       seed: int = 42):
    """
    运行MCTS搜索过程动画

    Args:
        map_size: 地图大小
        obstacle_density: 障碍物密度
        iterations: 迭代次数
        seed: 随机种子
    """
    print("生成MCTS搜索动画...")

    grid_map = GridMap(map_size, map_size,
                       obstacle_density=obstacle_density,
                       seed=seed)

    # 收集每次迭代的数据
    iteration_data = []

    def on_iteration(iteration, root):
        if iteration % 10 == 0:  # 每10次迭代记录一次
            iteration_data.append((iteration, root))

    mcts = MCTSPathFinder(grid_map, max_iterations=iterations)
    mcts.on_iteration = on_iteration
    mcts.search()

    # 创建动画
    visualizer = PathVisualizer(grid_map)
    anim = visualizer.create_animation(grid_map, iteration_data, interval=200)

    plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MCTS路径规划 - 蒙特卡洛树搜索算法演示'
    )

    parser.add_argument('--size', type=int, default=30,
                        help='地图大小 (默认: 30)')
    parser.add_argument('--density', type=float, default=0.2,
                        help='障碍物密度 (0.1-0.3, 默认: 0.2)')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='MCTS最大迭代次数 (默认: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='不显示可视化')
    parser.add_argument('--save', action='store_true',
                        help='保存结果图片')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行性能基准测试')
    parser.add_argument('--animation', action='store_true',
                        help='显示MCTS搜索过程动画')

    args = parser.parse_args()

    if args.benchmark:
        from benchmark import run_benchmark
        run_benchmark(num_trials=3)
    elif args.animation:
        run_mcts_animation(
            map_size=min(args.size, 20),  # 动画用较小的地图
            obstacle_density=args.density,
            iterations=min(args.iterations, 500),
            seed=args.seed
        )
    else:
        run_demo(
            map_size=args.size,
            obstacle_density=args.density,
            mcts_iterations=args.iterations,
            seed=args.seed,
            visualize=not args.no_visualize,
            save_figure=args.save
        )


if __name__ == "__main__":
    main()
