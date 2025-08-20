"""
Text-based SSSP Algorithm Game

A console-based game that demonstrates the FastSSSP algorithm through
interactive graph building and algorithm comparison.

Perfect for terminals and environments without GUI support.
"""

import random
import time
from typing import List, Dict
from sssp_algorithm import Graph, FastSSSP, dijkstra_sssp


class TextGame:
    """Text-based SSSP algorithm game."""

    def __init__(self):
        self.graph = Graph(0)
        self.node_positions = {}  # node_id -> (x, y) for visualization
        self.source_node = None
        self.game_running = True

    def print_header(self):
        """Print the game header."""
        print("\n" + "="*60)
        print("           SSSP ALGORITHM TEXT GAME")
        print("="*60)
        print("Discover shortest paths with the new FastSSSP algorithm!")
        print("Compare it with traditional Dijkstra's algorithm.")
        print("="*60)

    def print_menu(self):
        """Print the main menu."""
        print("\nWhat would you like to do?")
        print("1. Add a node")
        print("2. Add an edge between nodes")
        print("3. Set source node")
        print("4. Run SSSP algorithms")
        print("5. View current graph")
        print("6. Generate random graph")
        print("7. Clear graph")
        print("8. Quit")
        print("\nEnter your choice (1-8): ", end="")

    def add_node(self):
        """Add a new node to the graph."""
        if self.graph.num_vertices >= 20:
            print("Maximum nodes reached (20). Clear some nodes first.")
            return

        # Generate random position for visualization
        x = random.randint(0, 50)
        y = random.randint(0, 20)

        node_id = self.graph.num_vertices
        self.graph = Graph(self.graph.num_vertices + 1)  # Recreate with new size
        self.node_positions[node_id] = (x, y)

        print(f"Added node {node_id} at position ({x}, {y})")

    def add_edge(self):
        """Add an edge between two nodes."""
        if self.graph.num_vertices < 2:
            print("Need at least 2 nodes to add an edge.")
            return

        self.print_graph()
        print("\nEnter edge details:")

        try:
            node1 = int(input("From node: "))
            node2 = int(input("To node: "))
            weight = float(input("Weight: "))

            if node1 < 0 or node1 >= self.graph.num_vertices or node2 < 0 or node2 >= self.graph.num_vertices:
                print("Invalid node IDs.")
                return

            if weight <= 0:
                print("Weight must be positive.")
                return

            self.graph.add_edge(node1, node2, weight)
            print(f"Added edge {node1} → {node2} with weight {weight}")

        except ValueError:
            print("Invalid input. Please enter numbers.")

    def set_source(self):
        """Set the source node for SSSP."""
        if self.graph.num_vertices == 0:
            print("No nodes available.")
            return

        self.print_graph()
        try:
            source = int(input("Enter source node ID: "))
            if 0 <= source < self.graph.num_vertices:
                self.source_node = source
                print(f"Set source node to {source}")
            else:
                print("Invalid node ID.")
        except ValueError:
            print("Invalid input.")

    def run_algorithms(self):
        """Run both SSSP algorithms and compare results."""
        if self.graph.num_vertices == 0:
            print("No graph to analyze.")
            return

        if self.source_node is None:
            print("Please set a source node first.")
            return

        print(f"\nRunning SSSP algorithms from source node {self.source_node}...")
        print("Please wait...\n")

        # Run FastSSSP
        start_time = time.time()
        fast_algorithm = FastSSSP(self.graph)
        fast_distances, fast_predecessors = fast_algorithm.compute_sssp(self.source_node)
        fast_time = time.time() - start_time

        # Run Dijkstra
        start_time = time.time()
        dijkstra_distances, dijkstra_predecessors = dijkstra_sssp(self.graph, self.source_node)
        dijkstra_time = time.time() - start_time

        # Display results
        print("RESULTS:")
        print("-" * 60)
        print(f"{'Node':<5} {'FastSSSP':<12} {'Dijkstra':<12} {'Match'}")
        print("-" * 60)

        matches = 0
        total_nodes = self.graph.num_vertices

        for node in range(total_nodes):
            fast_dist = fast_distances[node]
            dijkstra_dist = dijkstra_distances[node]

            # Format distances
            if fast_dist == float('inf'):
                fast_str = "∞"
            else:
                fast_str = f"{fast_dist:.2f}"

            if dijkstra_dist == float('inf'):
                dijkstra_str = "∞"
            else:
                dijkstra_str = f"{dijkstra_dist:.2f}"

            # Check match
            match = "✓" if abs(fast_dist - dijkstra_dist) < 1e-10 else "✗"
            if match == "✓":
                matches += 1

            print(f"{node:<5} {fast_str:<12} {dijkstra_str:<12} {match}")

        print("-" * 60)
        speedup = dijkstra_time / fast_time if fast_time > 0 else float('inf')
        print(f"Performance:")
        print(f"  FastSSSP:  {fast_time:.6f} seconds")
        print(f"  Dijkstra:  {dijkstra_time:.6f} seconds")
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"  Accuracy:  {matches}/{total_nodes} nodes correct")

        if matches == total_nodes:
            print("\n🎉 Perfect! FastSSSP found the same shortest paths as Dijkstra!")
        else:
            print("\n⚠️  Some distances differ - there might be an implementation issue.")

        # Show some example paths
        self.show_example_paths(fast_predecessors, dijkstra_predecessors)

    def show_example_paths(self, fast_preds, dijkstra_preds):
        """Show example paths from source to a few nodes."""
        if self.graph.num_vertices <= 1:
            return

        print(f"\nExample Paths from source {self.source_node}:")
        print("-" * 40)

        # Show paths to first few nodes
        for target in range(min(3, self.graph.num_vertices)):
            if target == self.source_node:
                continue

            fast_path = self.reconstruct_path(fast_preds, self.source_node, target)
            dijkstra_path = self.reconstruct_path(dijkstra_preds, self.source_node, target)

            print(f"To node {target}:")
            print(f"  FastSSSP:  {' → '.join(map(str, fast_path))}")
            print(f"  Dijkstra:  {' → '.join(map(str, dijkstra_path))}")

    def reconstruct_path(self, predecessors, source, target):
        """Reconstruct path from source to target using predecessor pointers."""
        if predecessors[target] is None:
            return [target] if target == source else []

        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]

        if path[-1] != source:
            return []  # No path found

        path.reverse()
        return path

    def print_graph(self):
        """Print the current graph structure."""
        print(f"\nCurrent Graph (Source: {self.source_node}):")
        print("-" * 40)
        print(f"Nodes: {self.graph.num_vertices}")
        print(f"Edges: {self.graph.num_edges}")

        if self.graph.num_vertices > 0:
            print("\nNode positions:")
            for node_id in range(self.graph.num_vertices):
                if node_id in self.node_positions:
                    x, y = self.node_positions[node_id]
                    marker = " [SOURCE]" if node_id == self.source_node else ""
                    print(f"  {node_id}: ({x}, {y}){marker}")

            print("\nEdges:")
            if self.graph.num_edges > 0:
                for node1 in range(self.graph.num_vertices):
                    for edge in self.graph.adjacency_list[node1]:
                        print(f"  {node1} → {edge.to}: {edge.weight}")
            else:
                print("  (no edges)")

    def generate_random_graph(self):
        """Generate a random graph for testing."""
        try:
            num_nodes = int(input("Number of nodes (5-15): "))
            if num_nodes < 5 or num_nodes > 15:
                print("Please enter a number between 5 and 15.")
                return

            edge_probability = 0.3  # 30% chance of edge between any two nodes

            # Create graph
            self.graph = Graph(num_nodes)
            self.node_positions = {}

            # Add edges randomly
            for i in range(num_nodes):
                self.node_positions[i] = (random.randint(0, 50), random.randint(0, 20))

                for j in range(i + 1, num_nodes):  # Only add each edge once
                    if random.random() < edge_probability:
                        weight = round(random.uniform(1.0, 10.0), 1)
                        self.graph.add_edge(i, j, weight)

            # Set random source
            self.source_node = random.randint(0, num_nodes - 1)

            print(f"Generated random graph with {num_nodes} nodes and {self.graph.num_edges} edges")
            print(f"Source node set to {self.source_node}")

        except ValueError:
            print("Invalid input.")

    def clear_graph(self):
        """Clear all nodes and edges."""
        self.graph = Graph(0)
        self.node_positions = {}
        self.source_node = None
        print("Graph cleared.")

    def play(self):
        """Main game loop."""
        self.print_header()

        while self.game_running:
            self.print_menu()
            try:
                choice = input().strip()

                if choice == '1':
                    self.add_node()
                elif choice == '2':
                    self.add_edge()
                elif choice == '3':
                    self.set_source()
                elif choice == '4':
                    self.run_algorithms()
                elif choice == '5':
                    self.print_graph()
                elif choice == '6':
                    self.generate_random_graph()
                elif choice == '7':
                    self.clear_graph()
                elif choice == '8':
                    print("Thanks for playing!")
                    self.game_running = False
                else:
                    print("Invalid choice. Please enter 1-8.")

            except KeyboardInterrupt:
                print("\nGame interrupted. Thanks for playing!")
                self.game_running = False
            except Exception as e:
                print(f"An error occurred: {e}")


def main():
    """Main function to run the text-based game."""
    print("Loading SSSP Algorithm Text Game...")
    game = TextGame()
    game.play()


if __name__ == "__main__":
    main()
