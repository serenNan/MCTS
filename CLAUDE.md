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
# Activate conda environment (use appropriate env or base)
conda activate base

# Run the main MCTS implementation
python main.py

# Run with visualization
python main.py --visualize

# Run comparison tests
python benchmark.py
```

## Architecture

```
MCTS/
├── main.py              # Entry point
├── mcts/
│   ├── node.py          # MCTS tree node implementation
│   ├── tree.py          # MCTS tree with selection, expansion, simulation, backpropagation
│   └── policy.py        # Selection and rollout policies (UCB1, random)
├── environment/
│   ├── grid.py          # 2D grid map generation with obstacles
│   └── visualizer.py    # Path and tree visualization
├── algorithms/
│   ├── astar.py         # A* algorithm for comparison
│   └── dijkstra.py      # Dijkstra algorithm for comparison
└── benchmark.py         # Performance comparison tests
```

## Key MCTS Concepts

- **Selection**: Use UCB1 (Upper Confidence Bound) to balance exploration vs exploitation
- **Expansion**: Add new child nodes for unexplored moves
- **Simulation**: Random rollout from new node to terminal state
- **Backpropagation**: Update visit counts and values along the path back to root
