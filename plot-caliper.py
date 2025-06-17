#!/usr/bin/env python3
"""
Plot memory usage from Caliper memory profiling data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import re

def parse_caliper_memory_data(filename):
    """Parse Caliper memory data from cali-query output"""
    
    # Read the file and extract the data section
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find the data section (after the header)
    lines = content.strip().split('\n')
    
    # Find where the actual data starts (after header line)
    data_start = 0
    for i, line in enumerate(lines):
        if 'mem.alloc' in line and 'min#alloc.total_size' in line:
            data_start = i + 1
            break
    
    # Parse the data
    memory_data = []
    current_phase = None
    phase_order = []
    
    for line in lines[data_start:]:
        if not line.strip():
            continue
            
        # Split line by whitespace, but be careful with multiple spaces
        parts = re.split(r'\s+', line.strip())
        
        if len(parts) < 10:
            continue
            
        # Extract memory operation type (malloc, calloc, realloc, free)
        mem_op = parts[0] if parts[0] else None
        
        # Look for phase information in the line
        phase_match = None
        for part in parts:
            if any(phase in part for phase in ['initialization', 'computation', 'cleanup', 'finalization']):
                phase_match = part
                break
        
        if phase_match and phase_match not in phase_order:
            phase_order.append(phase_match)
            current_phase = phase_match
        
        # Extract numeric values
        try:
            if mem_op in ['malloc', 'calloc', 'realloc']:
                min_size = float(parts[1]) if parts[1] != '' else 0
                max_size = float(parts[2]) if parts[2] != '' else 0
                sum_size = float(parts[3]) if parts[3] != '' else 0
                avg_size = float(parts[4]) if parts[4] != '' else 0
                count = int(parts[9]) if len(parts) > 9 and parts[9] != '' else 0
                
                memory_data.append({
                    'phase': current_phase or 'unknown',
                    'operation': mem_op,
                    'min_size': max(0, min_size),  # Handle negative values (frees)
                    'max_size': max(0, max_size),
                    'sum_size': max(0, sum_size),
                    'avg_size': max(0, avg_size),
                    'count': count,
                    'net_allocation': sum_size  # Positive for alloc, negative for free
                })
        except (ValueError, IndexError):
            continue
    
    return pd.DataFrame(memory_data), phase_order

def plot_memory_usage(df, phase_order):
    """Create multiple plots showing memory usage patterns"""
    
    # Filter out rows with no meaningful data
    df_clean = df[(df['sum_size'] > 0) | (df['net_allocation'] != 0)].copy()
    
    if df_clean.empty:
        print("No memory data found to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Memory Usage Analysis from Caliper', fontsize=16)
    
    # Plot 1: Total memory allocated per phase
    ax1 = axes[0, 0]
    phase_totals = df_clean.groupby('phase')['sum_size'].sum()
    
    # Reorder according to phase_order
    ordered_phases = [p for p in phase_order if p in phase_totals.index]
    if ordered_phases:
        phase_totals = phase_totals.reindex(ordered_phases)
    
    bars1 = ax1.bar(range(len(phase_totals)), phase_totals.values / (1024*1024), 
                    color='skyblue', alpha=0.7)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Total Memory Allocated (MB)')
    ax1.set_title('Memory Allocation by Phase')
    ax1.set_xticks(range(len(phase_totals)))
    ax1.set_xticklabels([p.replace('initialization.', 'init.').replace('finalization.', 'final.') 
                         for p in phase_totals.index], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars1, phase_totals.values):
        height_mb = value / (1024*1024)
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(phase_totals.values)/(1024*1024)/50,
                f'{height_mb:.2f}MB', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Memory operations breakdown
    ax2 = axes[0, 1]
    op_totals = df_clean.groupby('operation')['sum_size'].sum()
    colors = {'malloc': 'red', 'calloc': 'green', 'realloc': 'orange', 'free': 'purple'}
    
    wedges, texts, autotexts = ax2.pie(op_totals.values, labels=op_totals.index, 
                                      autopct='%1.1f%%', startangle=90,
                                      colors=[colors.get(op, 'gray') for op in op_totals.index])
    ax2.set_title('Memory Operations Breakdown')
    
    # Plot 3: Allocation counts per phase
    ax3 = axes[1, 0]
    count_by_phase = df_clean.groupby('phase')['count'].sum()
    if ordered_phases:
        count_by_phase = count_by_phase.reindex(ordered_phases)
    
    bars3 = ax3.bar(range(len(count_by_phase)), count_by_phase.values, 
                    color='lightcoral', alpha=0.7)
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Number of Allocations')
    ax3.set_title('Allocation Count by Phase')
    ax3.set_xticks(range(len(count_by_phase)))
    ax3.set_xticklabels([p.replace('initialization.', 'init.').replace('finalization.', 'final.') 
                         for p in count_by_phase.index], rotation=45, ha='right')
    
    # Plot 4: Average allocation size per phase (in KB for readability)
    ax4 = axes[1, 1]
    avg_by_phase = df_clean.groupby('phase')['avg_size'].mean()
    if ordered_phases:
        avg_by_phase = avg_by_phase.reindex(ordered_phases)
    
    bars4 = ax4.bar(range(len(avg_by_phase)), avg_by_phase.values / 1024, 
                    color='gold', alpha=0.7)
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Average Allocation Size (KB)')
    ax4.set_title('Average Allocation Size by Phase')
    ax4.set_xticks(range(len(avg_by_phase)))
    ax4.set_xticklabels([p.replace('initialization.', 'init.').replace('finalization.', 'final.') 
                         for p in avg_by_phase.index], rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def print_memory_summary(df):
    """Print a summary of memory usage"""
    print("\n=== Memory Usage Summary ===")
    
    # Total memory allocated
    total_allocated = df[df['sum_size'] > 0]['sum_size'].sum()
    print(f"Total memory allocated: {total_allocated/(1024*1024):.2f} MB")
    
    # By phase
    print("\nMemory allocation by phase:")
    phase_summary = df.groupby('phase').agg({
        'sum_size': 'sum',
        'count': 'sum',
        'avg_size': 'mean'
    }).round(2)
    
    for phase, row in phase_summary.iterrows():
        if row['sum_size'] > 0:
            print(f"  {phase}: {row['sum_size']/(1024*1024):.2f} MB, {row['count']} allocations, "
                  f"avg {row['avg_size']/1024:.1f} KB")
    
    # Largest allocations
    print(f"\nLargest single allocation: {df['max_size'].max()/(1024*1024):.2f} MB")
    print(f"Most allocations in one phase: {df.groupby('phase')['count'].sum().max()}")

# Main execution
if __name__ == "__main__":
    import sys
    
    # Handle command line arguments
    if len(sys.argv) < 2:
        print("Usage: python caliper_memory_plotter.py <memory_data_file> [output_plot.png]")
        print("\nExample:")
        print("  python caliper_memory_plotter.py memory_data.txt")
        print("  python caliper_memory_plotter.py memory_data.txt my_plot.png")
        sys.exit(1)
    
    filename = sys.argv[1]
    save_plot = len(sys.argv) > 2
    plot_filename = sys.argv[2] if save_plot else None
    
    try:
        df, phase_order = parse_caliper_memory_data(filename)
        
        if not df.empty:
            print(f"Processing memory data from: {filename}")
            print_memory_summary(df)
            fig = plot_memory_usage(df, phase_order)
            
            if fig:
                if save_plot:
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    print(f"\nPlot saved to: {plot_filename}")
                else:
                    plt.show()
        else:
            print("No memory data found in the file")
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please check the filename.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)
