"""
Cython-optimized implementation of FastSSSP algorithm tight loops.

This module contains C-accelerated versions of the most performance-critical
parts of the FastSSSP algorithm, including:
- Edge relaxation loops
- Distance update operations
- Graph traversal functions

Performance improvements:
- C-level loops instead of Python iteration
- Typed memoryviews for efficient data access
- Reduced function call overhead
- Optimized memory allocation
"""

import cython
from libc.stdlib cimport malloc, free
from libc.math cimport INFINITY
import numpy as np
cimport numpy as np

# Define types for better performance
ctypedef np.float64_t FLOAT
ctypedef np.int32_t INT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def relax_edges_optimized(np.ndarray[FLOAT, ndim=1] distances,
                         np.ndarray[INT, ndim=2] edges,
                         np.ndarray[FLOAT, ndim=1] weights):
    """
    Cython-optimized edge relaxation for SSSP.

    Args:
        distances: Array of vertex distances
        edges: Array of (from, to) edge pairs
        weights: Array of edge weights

    Returns:
        Number of relaxations performed
    """
    cdef:
        int num_edges = edges.shape[0]
        int i, u, v
        FLOAT new_dist, weight
        int relaxations = 0

    # Main relaxation loop - this is the tight loop we want to optimize
    for i in range(num_edges):
        u = edges[i, 0]
        v = edges[i, 1]
        weight = weights[i]

        # Skip if source distance is infinite
        if distances[u] == INFINITY:
            continue

        new_dist = distances[u] + weight

        # Relax edge if better path found
        if new_dist < distances[v]:
            distances[v] = new_dist
            relaxations += 1

    return relaxations

@cython.boundscheck(False)
@cython.wraparound(False)
def bellman_ford_optimized(np.ndarray[FLOAT, ndim=1] distances,
                          np.ndarray[INT, ndim=2] edges,
                          np.ndarray[FLOAT, ndim=1] weights,
                          int max_iterations):
    """
    Cython-optimized Bellman-Ford algorithm.

    Args:
        distances: Array of vertex distances
        edges: Array of (from, to) edge pairs
        weights: Array of edge weights
        max_iterations: Maximum number of iterations

    Returns:
        Number of iterations performed
    """
    cdef:
        int num_edges = edges.shape[0]
        int num_vertices = distances.shape[0]
        int iteration, i, u, v
        FLOAT new_dist, weight
        int updated

    for iteration in range(max_iterations):
        updated = 0

        # Relax all edges
        for i in range(num_edges):
            u = edges[i, 0]
            v = edges[i, 1]
            weight = weights[i]

            if distances[u] == INFINITY:
                continue

            new_dist = distances[u] + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                updated = 1

        # If no updates, we're done
        if updated == 0:
            return iteration + 1

    return max_iterations

@cython.boundscheck(False)
@cython.wraparound(False)
def dijkstra_optimized(np.ndarray[FLOAT, ndim=1] distances,
                      np.ndarray[INT, ndim=2] edges,
                      np.ndarray[FLOAT, ndim=1] weights,
                      int source):
    """
    Cython-optimized Dijkstra's algorithm using binary heap.

    Args:
        distances: Array of vertex distances (modified in-place)
        edges: Array of (from, to) edge pairs
        weights: Array of edge weights
        source: Source vertex index

    Returns:
        Number of vertices processed
    """
    cdef:
        int num_vertices = distances.shape[0]
        int num_edges = edges.shape[0]
        int i, u, v, current_vertex
        FLOAT current_dist, new_dist, weight
        np.ndarray[FLOAT, ndim=1] heap_distances = np.full(num_vertices, INFINITY, dtype=np.float64)
        np.ndarray[INT, ndim=1] heap_vertices = np.full(num_vertices, -1, dtype=np.int32)
        np.ndarray[INT, ndim=1] heap_positions = np.full(num_vertices, -1, dtype=np.int32)
        int heap_size = 0

    # Initialize distances and heap
    distances[:] = INFINITY
    distances[source] = 0.0

    # Add source to heap
    heap_distances[0] = 0.0
    heap_vertices[0] = source
    heap_positions[source] = 0
    heap_size = 1

    while heap_size > 0:
        # Extract minimum distance vertex
        current_dist = heap_distances[0]
        current_vertex = heap_vertices[0]

        # Remove from heap
        heap_size -= 1
        if heap_size > 0:
            heap_distances[0] = heap_distances[heap_size]
            heap_vertices[0] = heap_vertices[heap_size]
            heap_positions[heap_vertices[0]] = 0
            _heapify_down(heap_distances, heap_vertices, heap_positions, heap_size, 0)

        # Skip if we have a better path already
        if current_dist > distances[current_vertex]:
            continue

        # Relax all outgoing edges
        for i in range(num_edges):
            u = edges[i, 0]
            if u != current_vertex:
                continue

            v = edges[i, 1]
            weight = weights[i]
            new_dist = current_dist + weight

            if new_dist < distances[v]:
                distances[v] = new_dist

                # Update heap
                if heap_positions[v] == -1:
                    # Insert new vertex
                    heap_distances[heap_size] = new_dist
                    heap_vertices[heap_size] = v
                    heap_positions[v] = heap_size
                    heap_size += 1
                    _heapify_up(heap_distances, heap_vertices, heap_positions, heap_size, heap_size - 1)
                else:
                    # Decrease key
                    pos = heap_positions[v]
                    heap_distances[pos] = new_dist
                    _heapify_up(heap_distances, heap_vertices, heap_positions, heap_size, pos)

    return num_vertices - np.sum(np.isinf(distances))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _heapify_up(np.ndarray[FLOAT, ndim=1] distances,
                     np.ndarray[INT, ndim=1] vertices,
                     np.ndarray[INT, ndim=1] positions,
                     int heap_size, int index):
    """Heapify up operation for binary heap."""
    cdef:
        int parent
        FLOAT temp_dist
        INT temp_vertex

    while index > 0:
        parent = (index - 1) // 2
        if distances[index] >= distances[parent]:
            break

        # Swap
        temp_dist = distances[index]
        temp_vertex = vertices[index]
        distances[index] = distances[parent]
        vertices[index] = vertices[parent]
        distances[parent] = temp_dist
        vertices[parent] = temp_vertex

        positions[vertices[index]] = index
        positions[vertices[parent]] = parent

        index = parent

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _heapify_down(np.ndarray[FLOAT, ndim=1] distances,
                       np.ndarray[INT, ndim=1] vertices,
                       np.ndarray[INT, ndim=1] positions,
                       int heap_size, int index):
    """Heapify down operation for binary heap."""
    cdef:
        int left, right, smallest
        FLOAT temp_dist
        INT temp_vertex

    while True:
        left = 2 * index + 1
        right = 2 * index + 2
        smallest = index

        if left < heap_size and distances[left] < distances[smallest]:
            smallest = left
        if right < heap_size and distances[right] < distances[smallest]:
            smallest = right

        if smallest == index:
            break

        # Swap
        temp_dist = distances[index]
        temp_vertex = vertices[index]
        distances[index] = distances[smallest]
        vertices[index] = vertices[smallest]
        distances[smallest] = temp_dist
        vertices[smallest] = temp_vertex

        positions[vertices[index]] = index
        positions[vertices[smallest]] = smallest

        index = smallest

def create_graph_arrays(graph):
    """
    Convert graph to numpy arrays for Cython optimization.

    Args:
        graph: Graph object with adjacency list

    Returns:
        Tuple of (edges_array, weights_array)
    """
    edges = []
    weights = []

    for u in graph.vertices:
        for edge in graph.adjacency_list[u]:
            edges.append([u, edge.to])
            weights.append(edge.weight)

    if edges:
        edges_array = np.array(edges, dtype=np.int32)
        weights_array = np.array(weights, dtype=np.float64)
    else:
        edges_array = np.empty((0, 2), dtype=np.int32)
        weights_array = np.empty(0, dtype=np.float64)

    return edges_array, weights_array
