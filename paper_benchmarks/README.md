# Paper Benchmarks

Scripts and tools for generating plots and benchmarks for the **digital twin** paper.

## Overview

This package provides tools to evaluate the computational cost of the ROS2 digital twin system in **operational conditions** (with ROS2 running). The main focus is on benchmarking the `step()` function execution time using ROS2 launch.

## Structure

```
paper_benchmarks/
├── README.md                           # This file
├── launch/
│   └── benchmark_operativo.launch.py   # ROS2 launch file (NO wrapper script)
├── paper_benchmarks/
│   ├── __init__.py
│   └── benchmark_dynamics.py       # Modified dynamics node with profiling
├── scripts/
│   └── plot_results.py                # Generate plots for paper (offline)
├── urdf/                               # URDF models (copied)
│   ├── assembly_with_hand.urdf
│   └── assembly.urdf
├── config/                             # Configuration files (copied)
│   └── dynamics_params.yaml
└── results/                            # Output: CSV and plots
    ├── step_time_histogram.png
    ├── step_time_series.png
    ├── step_time_boxplot.png
    ├── step_time_vs_closure.png
    └── statistics.txt
```

## Dependencies

- ROS2 (tested on Humble)
- Pinocchio
- NumPy, SciPy
- Matplotlib, Pandas (for plotting)

## Usage

### 1. Build the workspace

```bash
cd /home/mbari/ros2_ws
colcon build --packages-select paper_benchmarks
source install/setup.bash
```

### 2. Run benchmark using ROS2 launch directly

Run a 30-second benchmark with auto-stop:

```bash
cd /home/mbari/ros2_ws
source install/setup.bash

ros2 launch paper_benchmarks benchmark_operativo.launch.py \
    profiling_output_file:=/tmp/benchmark_timing.csv \
    auto_stop_after_sec:=30
```

**What happens:**
- ROS2 launch starts the system (robot_state_publisher, dynamics, input)
- Dynamics node runs with profiling enabled
- After 30 seconds, the node auto-stops and saves CSV
- CSV is saved to `/tmp/benchmark_timing.csv`

**Note**: The `auto_stop_after_sec` parameter tells the dynamics node to stop after N seconds.

### 3. Generate plots (offline)

Generate all plots from the benchmark data:

```bash
python3 src/HES_DT_FDI/paper_benchmarks/scripts/plot_results.py \
    --input /tmp/benchmark_timing.csv \
    --output src/HES_DT_FDI/paper_benchmarks/results/
```

This generates:
- `step_time_histogram.png` - Distribution of step times
- `step_time_series.png` - Time series of step times
- `step_time_boxplot.png` - Box plot of step times
- `step_time_vs_closure.png` - Step time vs closure norm
- `statistics.txt` - Summary statistics

### 4. Custom duration

For a different duration (e.g., 60 seconds):

```bash
ros2 launch paper_benchmarks benchmark_operativo.launch.py \
    profiling_output_file:=/tmp/benchmark_timing.csv \
    auto_stop_after_sec:=60
```

## Results from Latest Run

### Statistics
- **Mean step time**: 1.80 ms
- **Std**: 0.53 ms
- **Min**: 1.09 ms
- **Median**: 1.63 ms
- **P95**: 2.83 ms
- **P99**: 3.73 ms
- **Max**: 6.79 ms
- **Total valid steps**: 14770

### Key Findings
- The step() function executes in ~1.8 ms on average (with ROS2 overhead)
- 95% of steps complete in under 2.83 ms
- Maximum observed time is 6.79 ms (likely due to solver iterations)
- The system is suitable for real-time operation at 1 kHz (dt=0.001)

## Profiling Details

The `benchmark_dynamics.py` node includes profiling code that measures:

- **Total step time**: Time for the entire `step()` method (includes ROS2 overhead)
- **Closure norm**: Residual of kinematic closure
- **NFEV**: Number of function evaluations in `solve_closure()`
- **At limit**: Whether the system is in a limit state

Data is saved to CSV with columns: `step, total_ns, total_ms, closure_norm, nfev, at_limit`

## How It Works

1. **Launch file** (`benchmark_operativo.launch.py`):
   - Starts ROS2 system (robot_state_publisher, dynamics, input)
   - Passes parameters including `auto_stop_after_sec`

2. **Dynamics node** (`benchmark_dynamics.py`):
   - Modified copy of `new_dynamics_with_hand.py`
   - Profiling enabled via `profiling_enabled` parameter
   - Auto-stop timer calls `rclpy.shutdown()` after N seconds
   - Saves CSV via `save_profile_data()` before shutdown

3. **Plot script** (`plot_results.py`):
   - Reads CSV file (offline analysis)
   - Generates plots in PNG format (300 DPI)
   - Saves statistics to text file

## Output Formats

Plots are generated in PNG format (300 DPI) suitable for inclusion in academic papers.

## Notes

- The system runs with **ROS2 overhead** (this measures real operational cost)
- For standalone (no ROS2) benchmarking, use the old approach
- The benchmark measures computational cost in realistic conditions
- Auto-stop ensures clean termination and CSV saving
