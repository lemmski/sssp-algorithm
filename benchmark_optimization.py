"""
Performance benchmark for Cython-optimized vs Python FastSSSP implementation.

This script compares the performance of:
1. Pure Python implementation
2. Cython-optimized tight loops
3. Traditional Dijkstra's algorithm

Usage:
    python benchmark_optimization.py
"""

import time
import random
import numpy as np
from typing import List
from sssp_algorithm import Graph, FastSSSP, dijkstra_sssp


def generate_test_graph(num_vertices: int, edge_probability: float = 0.3) -> Graph:
    """Generate a random test graph."""
    graph = Graph(num_vertices)

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):  # Only add each edge once
            if random.random() < edge_probability:
                weight = random.uniform(0.1, 10.0)
                graph.add_edge(i, j, weight)

    return graph


def benchmark_algorithm(name: str, algorithm_func, graph: Graph, source: int, runs: int = 3) -> float:
    """Benchmark an algorithm function."""
    times = []

    for _ in range(runs):
        start_time = time.perf_counter()
        distances, predecessors = algorithm_func(source)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    print(f"{name:20} | {avg_time:.6f}s | {len(distances)} vertices, {graph.num_edges} edges")
    return avg_time


def run_benchmarks():
    """Run comprehensive performance benchmarks."""
    print("="*80)
    print("FastSSSP Algorithm Performance Benchmark")
    print("="*80)
    print("Comparing Python vs Cython-optimized implementations")
    print("="*80)

    # Test different graph sizes
    test_cases = [
        (50, 0.2, "Small (50 vertices)"),
        (100, 0.15, "Medium (100 vertices)"),
        (200, 0.1, "Large (200 vertices)"),
        (500, 0.05, "XL (500 vertices)")
    ]

    all_results = []

    for num_vertices, edge_prob, description in test_cases:
        print(f"\n{description}")
        print("-" * 50)

        # Generate test graph
        graph = generate_test_graph(num_vertices, edge_prob)
        source = random.randint(0, num_vertices - 1)

        print(f"Graph: {num_vertices} vertices, {graph.num_edges} edges")
        print(f"Source: {source}")
        print()

        # Create algorithm instance
        algorithm = FastSSSP(graph)

        # Benchmark Python implementation
        python_time = benchmark_algorithm(
            "Python FastSSSP",
            algorithm.compute_sssp,
            graph, source
        )

        # Benchmark optimized implementation
        try:
            optimized_time = benchmark_algorithm(
                "Cython FastSSSP",
                algorithm.compute_sssp_optimized,
                graph, source
            )

            speedup = python_time / optimized_time if optimized_time > 0 else float('inf')
            print(f"Speedup: {speedup:.2f}x")

        except ImportError:
            print("Cython version: Not available (need to build with: python setup.py build_ext --inplace)")
            optimized_time = python_time  # For comparison

        # Benchmark traditional Dijkstra
        dijkstra_time = benchmark_algorithm(
            "Traditional Dijkstra",
            lambda s: dijkstra_sssp(graph, s),
            graph, source
        )

        # Store results
        all_results.append({
            'size': num_vertices,
            'python_time': python_time,
            'optimized_time': optimized_time,
            'dijkstra_time': dijkstra_time
        })

        print()

    # Summary
    print("="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    print(f"{'Graph Size':<12} {'Python':<12} {'Optimized':<12} {'Dijkstra':<12} {'Speedup':<12}")
    print("-" * 80)

    for result in all_results:
        size = result['size']
        python_t = result['python_time']
        opt_t = result['optimized_time']
        dijkstra_t = result['dijkstra_time']

        speedup = python_t / opt_t if opt_t > 0 else float('inf')
        speedup_str = f"{speedup:.1f}x" if speedup != float('inf') else "N/A"

        print(f"{size:<12} {python_t:<12.6f} {opt_t:<12.6f} {dijkstra_t:<12.6f} {speedup_str:<12}")

    print("\n" + "="*80)
    print("BUILDING CYTHON EXTENSIONS")
    print("="*80)
    print("To enable Cython optimizations:")
    print("1. Install Cython: pip install cython")
    print("2. Build extensions: python setup.py build_ext --inplace")
    print("3. Re-run this benchmark")
    print("\nExpected performance gains:")
    print("- Edge relaxation loops: 5-10x faster")
    print("- Overall algorithm: 2-5x faster")
    print("- Memory usage: 10-20% less")
    print("="*80)


def demonstrate_edge_relaxation_optimization():
    """Demonstrate the specific optimization of edge relaxation loops."""
    print("\n" + "="*60)
    print("EDGE RELAXATION OPTIMIZATION DEMO")
    print("="*60)

    # Create a simple graph for demonstration
    graph = Graph(10)
    for i in range(9):
        graph.add_edge(i, i+1, 1.0)
        if i < 8:
            graph.add_edge(i, i+2, 1.5)

    print("Simple chain graph with cross edges")
    print(f"Vertices: {graph.num_vertices}, Edges: {graph.num_edges}")

    algorithm = FastSSSP(graph)

    # Time Python version
    start = time.perf_counter()
    for _ in range(1000):
        distances, _ = algorithm.compute_sssp(0)
    python_time = time.perf_counter() - start

    # Time optimized version (if available)
    try:
        start = time.perf_counter()
        for _ in range(1000):
            distances, _ = algorithm.compute_sssp_optimized(0)
        optimized_time = time.perf_counter() - start

        speedup = python_time / optimized_time
        print(f"Python time:    {python_time:.4f}s")
        print(f"Optimized time: {optimized_time:.4f}s")
        print(f"Speedup:        {speedup:.2f}x")

    except ImportError:
        print("Optimized version not available")
        print(f"Python time:    {python_time:.4f}s")


if __name__ == "__main__":
    run_benchmarks()
    demonstrate_edge_relaxation_optimization()
