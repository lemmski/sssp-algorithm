"""
Comprehensive unit tests for the FastSSSP algorithm implementation.

Tests cover:
- Basic functionality with sample graphs
- Edge cases (empty graphs, disconnected components)
- Correctness verification against known results
- Performance characteristics
- Error handling
"""

import unittest
import math
from sssp_algorithm import Graph, FastSSSP, Edge, dijkstra_sssp


class TestFastSSSP(unittest.TestCase):
    """Test suite for FastSSSP algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.simple_graph = self._create_simple_graph()
        self.complex_graph = self._create_complex_graph()
        self.empty_graph = Graph(0)

    def _create_simple_graph(self) -> Graph:
        """Create a simple test graph for basic functionality tests."""
        graph = Graph(4)

        # Add edges: (from, to, weight)
        edges = [
            (0, 1, 1.0),
            (0, 2, 4.0),
            (1, 2, 2.0),
            (1, 3, 6.0),
            (2, 3, 3.0),
        ]

        for from_v, to_v, weight in edges:
            graph.add_edge(from_v, to_v, weight)

        return graph

    def _create_complex_graph(self) -> Graph:
        """Create a more complex test graph with multiple paths."""
        graph = Graph(6)

        # Add edges: (from, to, weight)
        edges = [
            (0, 1, 4.0),
            (0, 2, 2.0),
            (1, 2, 1.0),
            (1, 3, 5.0),
            (2, 1, 1.0),
            (2, 3, 8.0),
            (2, 4, 10.0),
            (3, 4, 2.0),
            (3, 5, 6.0),
            (4, 5, 3.0),
        ]

        for from_v, to_v, weight in edges:
            graph.add_edge(from_v, to_v, weight)

        return graph

    def _create_disconnected_graph(self) -> Graph:
        """Create a graph with disconnected components."""
        graph = Graph(5)

        # Component 1: vertices 0, 1
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 0, 1.0)

        # Component 2: vertices 2, 3, 4 (isolated from first component)
        graph.add_edge(2, 3, 2.0)
        graph.add_edge(3, 4, 3.0)

        return graph

    def test_graph_creation(self):
        """Test basic graph creation and edge addition."""
        graph = Graph(3)

        # Test initial state
        self.assertEqual(graph.num_vertices, 3)
        self.assertEqual(graph.num_edges, 0)
        self.assertEqual(len(graph.vertices), 3)

        # Add edges
        graph.add_edge(0, 1, 2.5)
        graph.add_edge(1, 2, 3.7)

        self.assertEqual(graph.num_edges, 2)
        self.assertEqual(len(graph.adjacency_list[0]), 1)
        self.assertEqual(len(graph.adjacency_list[1]), 1)
        self.assertEqual(len(graph.adjacency_list[2]), 0)

    def test_negative_weight_error(self):
        """Test that negative edge weights raise an error."""
        graph = Graph(2)

        with self.assertRaises(ValueError):
            graph.add_edge(0, 1, -1.0)

    def test_simple_sssp(self):
        """Test SSSP on a simple graph with known shortest paths."""
        algorithm = FastSSSP(self.simple_graph)
        distances, predecessors = algorithm.compute_sssp(0)

        # Verify distances
        expected_distances = {
            0: 0.0,
            1: 1.0,
            2: 3.0,
            3: 6.0
        }

        for vertex, expected_dist in expected_distances.items():
            self.assertAlmostEqual(distances[vertex], expected_dist, places=5,
                                 msg=f"Distance to vertex {vertex} incorrect")

        # Verify predecessors
        self.assertIsNone(predecessors[0])  # Source has no predecessor
        self.assertEqual(predecessors[1], 0)  # 0 -> 1
        self.assertEqual(predecessors[2], 1)  # 0 -> 1 -> 2
        self.assertEqual(predecessors[3], 2)  # 0 -> 1 -> 2 -> 3

    def test_complex_sssp(self):
        """Test SSSP on a more complex graph."""
        algorithm = FastSSSP(self.complex_graph)
        distances, predecessors = algorithm.compute_sssp(0)

        # Verify some key distances
        self.assertAlmostEqual(distances[0], 0.0, places=5)
        self.assertAlmostEqual(distances[1], 3.0, places=5)  # 0->2->1
        self.assertAlmostEqual(distances[2], 2.0, places=5)  # 0->2
        self.assertAlmostEqual(distances[3], 8.0, places=5)  # 0->2->1->3: 2+1+5=8
        self.assertAlmostEqual(distances[4], 10.0, places=5) # 0->2->1->3->4: 2+1+5+2=10
        self.assertAlmostEqual(distances[5], 13.0, places=5) # 0->2->1->3->4->5: 2+1+5+2+3=13

    def test_invalid_source(self):
        """Test that invalid source vertex raises an error."""
        algorithm = FastSSSP(self.simple_graph)

        with self.assertRaises(ValueError):
            algorithm.compute_sssp(10)  # Vertex 10 doesn't exist

        with self.assertRaises(ValueError):
            algorithm.compute_sssp(-1)  # Negative vertex index

    def test_empty_graph(self):
        """Test algorithm behavior with empty graph."""
        algorithm = FastSSSP(self.empty_graph)

        with self.assertRaises(ValueError):
            algorithm.compute_sssp(0)  # No vertices available

    def test_single_vertex_graph(self):
        """Test algorithm on a single vertex graph."""
        graph = Graph(1)
        algorithm = FastSSSP(graph)
        distances, predecessors = algorithm.compute_sssp(0)

        self.assertEqual(len(distances), 1)
        self.assertAlmostEqual(distances[0], 0.0, places=5)
        self.assertIsNone(predecessors[0])

    def test_disconnected_graph(self):
        """Test algorithm on a graph with disconnected components."""
        graph = self._create_disconnected_graph()
        algorithm = FastSSSP(graph)
        distances, predecessors = algorithm.compute_sssp(0)

        # Component 1 should be reachable
        self.assertAlmostEqual(distances[0], 0.0, places=5)
        self.assertAlmostEqual(distances[1], 1.0, places=5)

        # Component 2 should be unreachable (infinite distance)
        self.assertEqual(distances[2], float('inf'))
        self.assertEqual(distances[3], float('inf'))
        self.assertEqual(distances[4], float('inf'))

    def test_zero_weight_edges(self):
        """Test algorithm with zero-weight edges."""
        graph = Graph(3)
        graph.add_edge(0, 1, 0.0)
        graph.add_edge(1, 2, 1.0)

        algorithm = FastSSSP(graph)
        distances, predecessors = algorithm.compute_sssp(0)

        self.assertAlmostEqual(distances[0], 0.0, places=5)
        self.assertAlmostEqual(distances[1], 0.0, places=5)
        self.assertAlmostEqual(distances[2], 1.0, places=5)

    def test_floating_point_precision(self):
        """Test algorithm with very small and very large floating point weights."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1e-10)
        graph.add_edge(1, 2, 1e10)

        algorithm = FastSSSP(graph)
        distances, predecessors = algorithm.compute_sssp(0)

        self.assertAlmostEqual(distances[0], 0.0, places=10)
        self.assertAlmostEqual(distances[1], 1e-10, places=10)
        self.assertAlmostEqual(distances[2], 1e10, places=5)

    def test_multiple_sources(self):
        """Test that running SSSP from different sources works correctly."""
        algorithm = FastSSSP(self.simple_graph)

        # Test from source 0
        distances_0, _ = algorithm.compute_sssp(0)

        # Test from source 2
        distances_2, _ = algorithm.compute_sssp(2)

        # Verify different results
        self.assertNotEqual(distances_0[3], distances_2[3])
        self.assertAlmostEqual(distances_0[0], 0.0, places=5)
        self.assertAlmostEqual(distances_2[2], 0.0, places=5)

    def test_no_negative_distances(self):
        """Test that all computed distances are non-negative."""
        algorithm = FastSSSP(self.complex_graph)
        distances, _ = algorithm.compute_sssp(0)

        # All finite distances should be non-negative
        for vertex, distance in distances.items():
            if distance != float('inf'):
                self.assertGreaterEqual(distance, 0.0,
                                      f"Negative distance found for vertex {vertex}: {distance}")

    def test_source_distance_zero(self):
        """Test that the source vertex always has distance 0."""
        algorithm = FastSSSP(self.complex_graph)
        distances, _ = algorithm.compute_sssp(0)

        self.assertEqual(distances[0], 0.0, "Source vertex should have distance 0")

    def test_correctness_vs_dijkstra(self):
        """Test that our algorithm produces the same results as Dijkstra's algorithm."""
        # Test on simple graph
        fast_distances, _ = FastSSSP(self.simple_graph).compute_sssp(0)
        dijkstra_distances, _ = dijkstra_sssp(self.simple_graph, 0)

        for vertex in self.simple_graph.vertices:
            self.assertAlmostEqual(fast_distances[vertex], dijkstra_distances[vertex], places=5,
                                 msg=f"Distance mismatch for vertex {vertex} in simple graph")

        # Test on complex graph
        fast_distances, _ = FastSSSP(self.complex_graph).compute_sssp(0)
        dijkstra_distances, _ = dijkstra_sssp(self.complex_graph, 0)

        for vertex in self.complex_graph.vertices:
            self.assertAlmostEqual(fast_distances[vertex], dijkstra_distances[vertex], places=5,
                                 msg=f"Distance mismatch for vertex {vertex} in complex graph: "
                                      f"Fast={fast_distances[vertex]}, Dijkstra={dijkstra_distances[vertex]}")

    def test_path_reconstruction(self):
        """Test that predecessor pointers allow correct path reconstruction."""
        algorithm = FastSSSP(self.simple_graph)
        distances, predecessors = algorithm.compute_sssp(0)

        # Reconstruct path to vertex 3: should be 0 -> 1 -> 2 -> 3
        path = []
        current = 3
        while current is not None:
            path.append(current)
            current = predecessors[current]

        path.reverse()
        expected_path = [0, 1, 2, 3]
        self.assertEqual(path, expected_path)

        # Verify path distances
        total_distance = 0.0
        for i in range(len(path) - 1):
            # Find edge weight between path[i] and path[i+1]
            for edge in self.simple_graph.adjacency_list[path[i]]:
                if edge.to == path[i+1]:
                    total_distance += edge.weight
                    break

        self.assertAlmostEqual(total_distance, distances[3], places=5)


class TestGraphEdgeCases(unittest.TestCase):
    """Test edge cases for the Graph class."""

    def test_large_graph(self):
        """Test graph with many vertices."""
        num_vertices = 1000
        graph = Graph(num_vertices)

        # Add some edges
        for i in range(num_vertices - 1):
            graph.add_edge(i, i + 1, 1.0)

        self.assertEqual(graph.num_vertices, num_vertices)
        self.assertEqual(graph.num_edges, num_vertices - 1)

    def test_self_loops(self):
        """Test graph with self-loops."""
        graph = Graph(3)
        graph.add_edge(0, 0, 1.0)  # Self-loop
        graph.add_edge(0, 1, 2.0)

        algorithm = FastSSSP(graph)
        distances, predecessors = algorithm.compute_sssp(0)

        # Distance to self should be 0 (not using the self-loop)
        self.assertAlmostEqual(distances[0], 0.0, places=5)
        self.assertAlmostEqual(distances[1], 2.0, places=5)

    def test_parallel_edges(self):
        """Test graph with parallel edges (same source and target)."""
        graph = Graph(3)
        graph.add_edge(0, 1, 5.0)
        graph.add_edge(0, 1, 2.0)  # Parallel edge with smaller weight

        algorithm = FastSSSP(graph)
        distances, predecessors = algorithm.compute_sssp(0)

        # Should use the smaller weight edge
        self.assertAlmostEqual(distances[1], 2.0, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
