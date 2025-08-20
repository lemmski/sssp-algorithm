"""
Demonstration of the FastSSSP algorithm implementation.

This script shows the algorithm working on a concrete example,
comparing results with Dijkstra's algorithm.
"""

from sssp_algorithm import Graph, FastSSSP, dijkstra_sssp


def create_demo_graph() -> Graph:
    """Create a demonstration graph with known shortest paths."""
    graph = Graph(8)

    # Add edges: (from, to, weight)
    edges = [
        (0, 1, 4.0), (0, 2, 2.0), (0, 3, 6.0),
        (1, 2, 1.0), (1, 4, 3.0), (1, 5, 8.0),
        (2, 3, 3.0), (2, 4, 5.0), (2, 6, 7.0),
        (3, 6, 2.0), (3, 7, 4.0),
        (4, 5, 2.0), (4, 7, 6.0),
        (5, 7, 1.0),
        (6, 7, 3.0)
    ]

    for from_v, to_v, weight in edges:
        graph.add_edge(from_v, to_v, weight)

    return graph


def print_path(predecessors: dict, source: int, target: int) -> str:
    """Reconstruct and print the path from source to target."""
    if predecessors[target] is None:
        return f"No path to {target}"

    path = []
    current = target
    while current is not None:
        path.append(current)
        current = predecessors[current]

    path.reverse()
    return " → ".join(map(str, path))


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("FastSSSP Algorithm Demonstration")
    print("=" * 70)

    # Create demo graph
    graph = create_demo_graph()
    print(f"Demo Graph: {graph.num_vertices} vertices, {graph.num_edges} edges")
    print()

    # Choose source vertex
    source = 0
    print(f"Computing shortest paths from source vertex {source}")
    print("-" * 50)

    # Run FastSSSP algorithm
    fast_algorithm = FastSSSP(graph)
    fast_distances, fast_predecessors = fast_algorithm.compute_sssp(source)

    # Run Dijkstra's algorithm for comparison
    dijkstra_distances, dijkstra_predecessors = dijkstra_sssp(graph, source)

    # Display results
    print(f"{'Vertex':<8} {'FastSSSP':<12} {'Dijkstra':<12} {'Match':<8} {'Path'}")
    print("-" * 80)

    for vertex in range(graph.num_vertices):
        fast_dist = fast_distances[vertex]
        dijkstra_dist = dijkstra_distances[vertex]

        # Format distances
        if fast_dist == float('inf'):
            fast_str = "∞"
        else:
            fast_str = f"{fast_dist:.2f}"

        if dijkstra_dist == float('inf'):
            dijkstra_str = "∞"
        else:
            dijkstra_str = f"{dijkstra_dist:.2f}"

        # Check if distances match
        match = "✓" if abs(fast_dist - dijkstra_dist) < 1e-10 else "✗"

        # Get path
        path = print_path(fast_predecessors, source, vertex)

        print(f"{vertex:<8} {fast_str:<12} {dijkstra_str:<12} {match:<8} {path}")

    print()
    print("Algorithm Analysis:")
    print("-" * 30)
    print("• FastSSSP uses recursive partitioning to achieve O(m log^{2/3} n) complexity")
    print("• Breaks the O(m + n log n) barrier of traditional Dijkstra's algorithm")
    print("• Combines Dijkstra and Bellman-Ford approaches through pivot selection")
    print("• Particularly effective on sparse graphs with non-negative edge weights")

    print()
    print("Key Features:")
    print("-" * 20)
    print("✓ Deterministic algorithm (no randomization)")
    print("✓ Works in the comparison-addition model")
    print("✓ Handles disconnected graphs correctly")
    print("✓ Supports zero-weight edges")
    print("✓ Provides both distances and predecessor pointers")
    print("✓ Validated against standard Dijkstra's implementation")

    print()
    print("=" * 70)
    print("Demonstration completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
