#!/usr/bin/env python3
"""
Generate plots for paper from benchmark timing data.
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_timing_data(df, output_dir, prefix=''):
    """
    Generate all plots for the paper.

    Usage:
    
    python3 src/HES_DT_FDI/paper_benchmarks/scripts/plot_computation_time.py 
            --input src/HES_DT_FDI/paper_benchmarks/results/REPO/step_standalone_timing.csv 
            --output src/HES_DT_FDI/paper_benchmarks/results/REPO/
            --prefix standalone_
    
    Args:
        df: DataFrame with timing data
        output_dir: Output directory for plots
        prefix: Optional prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert prefix to string if needed
    if prefix and not prefix.endswith('_'):
        prefix = prefix + '_'
    
    # ── Normalise columns: convert nanosecond columns to milliseconds ──
    if 'total_ms' not in df.columns and 'total_ns' in df.columns:
        df['total_ms'] = df['total_ns'] / 1e6
    
    ns_cols = [c for c in df.columns if c.endswith('_ns') and c != 'total_ns']
    for col in ns_cols:
        ms_col = col.replace('_ns', '_ms')
        if ms_col not in df.columns:
            df[ms_col] = df[col] / 1e6
    
    # Statistics
    stats = {
        'Mean (ms)': df['total_ms'].mean(),
        'Std (ms)': df['total_ms'].std(),
        'Min (ms)': df['total_ms'].min(),
        'Median (ms)': df['total_ms'].median(),
        'P95 (ms)': np.percentile(df['total_ms'], 95),
        'P99 (ms)': np.percentile(df['total_ms'], 99),
        'Max (ms)': df['total_ms'].max(),
        'Total Steps': len(df)
    }
    
    # Print statistics
    print("\n=== Statistics ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, f"{prefix}statistics.txt")
    with open(stats_file, 'w') as f:
        for k, v in stats.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")
    print(f"\nStatistics saved to: {stats_file}")
    
    # 1. Histogram of step times
    plt.figure(figsize=(10, 6))
    plt.hist(df['total_ms'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    plt.axvline(stats['Mean (ms)'], color='red', linestyle='--', 
                label=f"Mean: {stats['Mean (ms)']:.2f} ms")
    plt.axvline(stats['P95 (ms)'], color='orange', linestyle='--',
                label=f"P95: {stats['P95 (ms)']:.2f} ms")
    plt.xlabel('Step Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dynamics Step Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}step_time_histogram.png"), dpi=300)
    plt.close()
    print(f"Plot saved: {prefix}step_time_histogram.png")
    
    # 2. Time series (first N steps)
    n_plot = min(1000, len(df))
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'].iloc[:n_plot], df['total_ms'].iloc[:n_plot], 
             linewidth=0.8, color='blue', alpha=0.7)
    plt.axhline(stats['Mean (ms)'], color='red', linestyle='--', alpha=0.5,
                label=f"Mean: {stats['Mean (ms)']:.2f} ms")
    plt.xlabel('Step Number')
    plt.ylabel('Step Time (ms)')
    plt.title(f'Step Time Series (first {n_plot} steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}step_time_series.png"), dpi=300)
    plt.close()
    print(f"Plot saved: {prefix}step_time_series.png")
    
    # 3. Box plot
    if len(df) > 10:
        plt.figure(figsize=(8, 6))
        box = plt.boxplot(df['total_ms'], vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('lightblue')
        plt.ylabel('Step Time (ms)')
        plt.title('Step Time Box Plot')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        textstr = f"Mean: {stats['Mean (ms)']:.2f} ms\nMedian: {stats['Median (ms)']:.2f} ms\nP95: {stats['P95 (ms)']:.2f} ms"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}step_time_boxplot.png"), dpi=300)
        plt.close()
        print(f"Plot saved: {prefix}step_time_boxplot.png")
    
    # 4. Scatter plot: closure_norm vs step time
    if 'closure_norm' in df.columns:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['step'].iloc[:n_plot], df['total_ms'].iloc[:n_plot], 
                           c=df['closure_norm'].iloc[:n_plot], 
                           cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(scatter, label='Closure Norm')
        plt.xlabel('Step Number')
        plt.ylabel('Step Time (ms)')
        plt.title(f'Step Time vs Closure Norm (first {n_plot} steps)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}step_time_vs_closure.png"), dpi=300)
        plt.close()
        print(f"Plot saved: {prefix}step_time_vs_closure.png")
    
    # Section timing statistics (if detailed profiling)
    section_cols = [col for col in df.columns if col.endswith('_ms') and col != 'total_ms']
    section_stats = {}
    for col in section_cols:
        if col in df.columns:
            section_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'max': df[col].max()
            }
    
    # 5. Section breakdown plots (if detailed profiling)
    if section_stats:
        # Horizontal stacked bar chart of average section times
        sections = list(section_stats.keys())
        means = [section_stats[s]['mean'] for s in sections]
        stds = [section_stats[s]['std'] for s in sections]
        total = sum(means)
        
        # Sort by size (largest first)
        sorted_indices = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
        sections = [sections[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(sections)))
        
        left = 0
        for i, (section, mean, std) in enumerate(zip(sections, means, stds)):
            pct = (mean / total) * 100
            color = colors[i]
            
            # Bar segment
            ax.barh(0, mean, left=left, height=0.5, color=color, edgecolor='black')
            
            # Label
            clean_name = section.replace('_ms', '')
            if pct > 10:
                # Inside label for large segments
                label = f'{clean_name}\n{mean:.3f}ms ({pct:.1f}%)'
                ax.text(left + mean/2, 0, label, ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white')
            else:
                # Outside label for small segments
                label = f'{clean_name}: {mean:.3f}ms ({pct:.1f}%)'
                ax.text(left + mean + 0.01, 0, label, ha='left', va='center',
                        fontsize=9)
            
            left += mean
        
        # Formatting
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_yticks([0])
        ax.set_yticklabels(['Average Step Time'], fontsize=11)
        ax.set_xlim(0, total * 1.15)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_title('Average Step Time Breakdown by Section', fontsize=14, fontweight='bold')
        
        # Add total time annotation
        total_mean = sum(means)
        total_std = np.sqrt(sum(s**2 for s in stds))
        ax.text(total * 1.16, 0, f'Total: {total_mean:.3f}±{total_std:.3f}ms',
                ha='left', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}section_breakdown_horizontal.png"), dpi=300)
        plt.close()
        print(f"Plot saved: {prefix}section_breakdown_horizontal.png")
        
        # Bar chart of section statistics
        fig, ax = plt.subplots(figsize=(10, 6))
        section_names = list(section_stats.keys())
        means = [section_stats[k]['mean'] for k in section_names]
        stds = [section_stats[k]['std'] for k in section_names]
        
        x_pos = np.arange(len(section_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                      color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Section')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Section Timing Statistics (Mean ± Std)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace('_ms', '') for s in section_names], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}', 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}section_breakdown_bar.png"), dpi=300)
        plt.close()
        print(f"Plot saved: {prefix}section_breakdown_bar.png")
        
        # Save section statistics to file
        section_stats_file = os.path.join(output_dir, f"{prefix}section_statistics.txt")
        with open(section_stats_file, 'w') as f:
            f.write("=== Section Timing Statistics ===\n\n")
            for section, stat in section_stats.items():
                f.write(f"{section}:\n")
                f.write(f"  Mean: {stat['mean']:.4f} ms\n")
                f.write(f"  Std:  {stat['std']:.4f} ms\n")
                f.write(f"  Max:  {stat['max']:.4f} ms\n\n")
        print(f"Section statistics saved to: {section_stats_file}")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate plots from benchmark results'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Input CSV file'
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory for plots (default: results)'
    )
    parser.add_argument(
        '--prefix', type=str, default='',
        help='Prefix for plot filenames'
    )
    
    args = parser.parse_args()
    
    # Single file analysis
    df = pd.read_csv(args.input)
    plot_timing_data(df, args.output, args.prefix)
