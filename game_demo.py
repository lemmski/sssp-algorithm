"""
Demo script showing how to use the SSSP Minigame programmatically.
This creates a sample graph and demonstrates the algorithm.
"""

from sssp_minigame import Node, Edge, SSSPGame
import time


def create_demo_graph():
    """Create a sample graph for demonstration."""
    # Create nodes
    nodes = [
        Node(100, 100, 0),
        Node(300, 100, 1),
        Node(500, 100, 2),
        Node(200, 300, 3),
        Node(400, 300, 4),
        Node(300, 500, 5)
    ]

    # Create edges
    edges = [
        Edge(nodes[0], nodes[1], 4.0),  # 0 → 1: 4
        Edge(nodes[0], nodes[3], 2.0),  # 0 → 3: 2
        Edge(nodes[1], nodes[3], 1.0),  # 1 → 3: 1
        Edge(nodes[1], nodes[4], 5.0),  # 1 → 4: 5
        Edge(nodes[2], nodes[4], 3.0),  # 2 → 4: 3
        Edge(nodes[3], nodes[4], 8.0),  # 3 → 4: 8
        Edge(nodes[3], nodes[5], 6.0),  # 3 → 5: 6
        Edge(nodes[4], nodes[5], 2.0),  # 4 → 5: 2
    ]

    return nodes, edges


def demo_algorithm_comparison():
    """Demonstrate the algorithm comparison feature."""
    print("SSSP Algorithm Demo")
    print("==================")

    # Create game instance (but don't run the GUI)
    game = SSSPGame()

    # Set up demo graph
    nodes, edges = create_demo_graph()
    game.nodes = nodes
    game.edges = edges
    game.source_node = nodes[0]  # Node 0 as source
    game.target_node = nodes[5]  # Node 5 as target

    print("Demo Graph Created:")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"Source: Node {game.source_node.id}")
    print(f"Target: Node {game.target_node.id}")

    print("\nEdges:")
    for edge in edges:
        print(f"  {edge.node1.id} → {edge.node2.id}: {edge.weight}")

    # Run algorithms
    print("\nRunning algorithms...")
    game.run_sssp_algorithms()

    # Display results
    print("\nShortest Path Results:")
    print("-" * 50)
    for node in nodes:
        fast_dist = game.fast_distances.get(node.id, float('inf'))
        dijkstra_dist = game.dijkstra_distances.get(node.id, float('inf'))

        fast_str = f"{fast_dist:.1f}" if fast_dist != float('inf') else "∞"
        dijkstra_str = f"{dijkstra_dist:.1f}" if dijkstra_dist != float('inf') else "∞"

        match = "✓" if abs(fast_dist - dijkstra_dist) < 1e-10 else "✗"
        print(f"Node {node.id}: FastSSSP={fast_str}, Dijkstra={dijkstra_str} {match}")

    # Show path to target
    target_dist_fast = game.fast_distances.get(game.target_node.id, float('inf'))
    target_dist_dijkstra = game.dijkstra_distances.get(game.target_node.id, float('inf'))

    print(f"\nPath to target (Node {game.target_node.id}):")
    print(f"FastSSSP distance: {target_dist_fast:.1f}")
    print(f"Dijkstra distance: {target_dist_dijkstra:.1f}")

    print("\nDemo completed! The graphical game shows this interactively.")


if __name__ == "__main__":
    demo_algorithm_comparison()
