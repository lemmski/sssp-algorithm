"""
Performance validation and correctness verification for the FastSSSP algorithm.

This script demonstrates:
1. Correctness verification against Dijkstra's algorithm
2. Performance characteristics on different graph types
3. Edge case handling
4. Algorithm behavior validation
"""

import time
import random
from typing import List, Tuple
from sssp_algorithm import Graph, FastSSSP, dijkstra_sssp


def generate_random_graph(num_vertices: int, edge_probability: float = 0.3,
                         max_weight: float = 10.0) -> Graph:
    """Generate a random graph for testing."""
    graph = Graph(num_vertices)

    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j and random.random() < edge_probability:
                weight = random.uniform(0.1, max_weight)
                graph.add_edge(i, j, weight)

    return graph


def validate_correctness(num_tests: int = 10, max_vertices: int = 50):
    """Validate correctness by comparing with Dijkstra's algorithm."""
    print(f"Running correctness validation with {num_tests} random graphs...")

    all_correct = True
    for test in range(num_tests):
        num_vertices = random.randint(5, max_vertices)
        graph = generate_random_graph(num_vertices)

        if graph.num_edges == 0:
            continue  # Skip empty graphs

        source = random.randint(0, num_vertices - 1)

        # Run both algorithms
        fast_algorithm = FastSSSP(graph)
        fast_distances, _ = fast_algorithm.compute_sssp(source)
        dijkstra_distances, _ = dijkstra_sssp(graph, source)

        # Compare results
        test_correct = True
        for vertex in range(num_vertices):
            fast_dist = fast_distances[vertex]
            dijkstra_dist = dijkstra_distances[vertex]

            # Handle infinite distances
            if fast_dist == float('inf') and dijkstra_dist == float('inf'):
                continue
            elif fast_dist == float('inf') or dijkstra_dist == float('inf'):
                test_correct = False
                break
            elif abs(fast_dist - dijkstra_dist) > 1e-10:
                test_correct = False
                break

        if test_correct:
            print(f"✓ Test {test + 1}: PASSED ({num_vertices} vertices, {graph.num_edges} edges)")
        else:
            print(f"✗ Test {test + 1}: FAILED ({num_vertices} vertices, {graph.num_edges} edges)")
            print(f"  First mismatch at vertex {vertex}: Fast={fast_distances[vertex]}, Dijkstra={dijkstra_distances[vertex]}")
            all_correct = False

    print(f"\nCorrectness validation: {'PASSED' if all_correct else 'FAILED'}")
    return all_correct


def performance_comparison(max_vertices: int = 100, step: int = 10):
    """Compare performance of FastSSSP vs Dijkstra's algorithm."""
    print(f"\nRunning performance comparison (up to {max_vertices} vertices)...")

    results = []

    for n in range(step, max_vertices + 1, step):
        # Generate a test graph
        graph = generate_random_graph(n, edge_probability=0.2)
        source = 0

        # Time FastSSSP
        start_time = time.time()
        fast_distances, _ = FastSSSP(graph).compute_sssp(source)
        fast_time = time.time() - start_time

        # Time Dijkstra
        start_time = time.time()
        dijkstra_distances, _ = dijkstra_sssp(graph, source)
        dijkstra_time = time.time() - start_time

        results.append((n, fast_time, dijkstra_time))

        # Verify correctness
        max_diff = max(abs(fast_distances[v] - dijkstra_distances[v])
                      for v in graph.vertices
                      if fast_distances[v] != float('inf') and dijkstra_distances[v] != float('inf'))

        ratio = fast_time / dijkstra_time if dijkstra_time > 0 else float('inf')
        print(f"n={n:3d}: FastSSSP={fast_time:.4f}s, Dijkstra={dijkstra_time:.4f}s, "
              f"Ratio={ratio:.2f}x, MaxDiff={max_diff:.2e}")

    return results


def test_edge_cases():
    """Test various edge cases."""
    print("\nTesting edge cases...")

    # Test 1: Single vertex graph
    single_graph = Graph(1)
    algorithm = FastSSSP(single_graph)
    distances, predecessors = algorithm.compute_sssp(0)
    assert distances[0] == 0.0
    assert predecessors[0] is None
    print("✓ Single vertex graph")

    # Test 2: Two vertices with edge
    two_graph = Graph(2)
    two_graph.add_edge(0, 1, 5.0)
    distances, _ = FastSSSP(two_graph).compute_sssp(0)
    assert distances[0] == 0.0
    assert distances[1] == 5.0
    print("✓ Two vertices with edge")

    # Test 3: Disconnected graph
    disc_graph = Graph(3)
    disc_graph.add_edge(0, 1, 1.0)  # Component 1: 0-1
    # Vertex 2 is isolated
    distances, _ = FastSSSP(disc_graph).compute_sssp(0)
    assert distances[0] == 0.0
    assert distances[1] == 1.0
    assert distances[2] == float('inf')
    print("✓ Disconnected graph")

    # Test 4: Zero weight edges
    zero_graph = Graph(3)
    zero_graph.add_edge(0, 1, 0.0)
    zero_graph.add_edge(1, 2, 1.0)
    distances, _ = FastSSSP(zero_graph).compute_sssp(0)
    assert distances[0] == 0.0
    assert distances[1] == 0.0  # Zero weight edge
    assert distances[2] == 1.0
    print("✓ Zero weight edges")

    print("All edge cases passed!")


def main():
    """Main validation function."""
    print("=" * 60)
    print("FastSSSP Algorithm Validation")
    print("=" * 60)

    # Set random seed for reproducibility
    random.seed(42)

    # Test edge cases
    test_edge_cases()

    # Validate correctness
    correctness_ok = validate_correctness(num_tests=20, max_vertices=30)

    # Performance comparison
    if correctness_ok:
        performance_results = performance_comparison(max_vertices=50, step=10)

        print(f"\nPerformance Summary:")
        print(f"{'Vertices':<10} {'FastSSSP':<10} {'Dijkstra':<10} {'Speedup':<10}")
        print("-" * 50)
        for n, fast_time, dijkstra_time in performance_results:
            speedup = dijkstra_time / fast_time if fast_time > 0 else float('inf')
            speedup_str = f"{speedup:.2f}x" if speedup != float('inf') else "N/A"
            print(f"{n:<10} {fast_time:<10.4f} {dijkstra_time:<10.4f} {speedup_str:<10}")

    print("\n" + "=" * 60)
    print("Validation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
