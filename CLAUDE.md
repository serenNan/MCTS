# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements the Monte Carlo Tree Search (MCTS) algorithm for solving static path planning problems in 2D grid maps. The goal is to find optimal paths for robots navigating from a start point to a goal while avoiding obstacles.

## Project Requirements

### Core Implementation
1. **Environment Setup**: Generate 2D grid maps (e.g., 30×30, 50×50) with random obstacles (10%-30% density), defined start and goal points
2. **MCTS Algorithm**: Implement the four core steps - Selection, Expansion, Simulation (rollout with random policy), and Backpropagation
3. **Visualization**: Dynamic display of MCTS tree growth including node visit counts, value estimates, and the final optimal path
4. **Performance Analysis**: Test algorithm performance under different map complexities, record iteration counts, planning time, and path length relationships
5. **Comparative Study**: Compare with A* and Dijkstra algorithms on path quality, search node count, and computation time
6. **Grid Map Generation**: Self-implement code to generate grid maps with obstacles

## Development Commands

```bash
# Run the main MCTS demo (with visualization)
conda run -n base python main.py

# Run without visualization
conda run -n base python main.py --no-visualize

# Custom map configuration
conda run -n base python main.py --size 50 --density 0.3 --iterations 10000

# Run performance benchmark
conda run -n base python main.py --benchmark

# Show MCTS search animation
conda run -n base python main.py --animation

# Save result figure
conda run -n base python main.py --save
```

## Architecture

```
MCTS/
├── main.py              # Entry point with CLI arguments
├── benchmark.py         # Performance comparison tests
├── requirements.txt     # Python dependencies
├── mcts/
│   ├── __init__.py
│   ├── node.py          # MCTSNode class with UCB1, statistics
│   └── tree.py          # MCTSPathFinder with select/expand/simulate/backprop
├── environment/
│   ├── __init__.py
│   ├── grid.py          # GridMap class for 2D grid generation
│   └── visualizer.py    # PathVisualizer for tree and path display
└── algorithms/
    ├── __init__.py
    ├── astar.py         # A* algorithm for comparison
    └── dijkstra.py      # Dijkstra algorithm for comparison
```

## Key MCTS Concepts

- **Selection**: Use UCB1 (Upper Confidence Bound) to balance exploration vs exploitation
- **Expansion**: Add new child nodes for unexplored moves
- **Simulation**: Random rollout from new node to terminal state
- **Backpropagation**: Update visit counts and values along the path back to root
