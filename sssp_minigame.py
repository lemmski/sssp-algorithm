"""
SSSP Algorithm Minigame

A visual game that demonstrates the FastSSSP algorithm through interactive graph building
and pathfinding visualization.

Features:
- Interactive graph creation (place nodes, add edges)
- Real-time visualization of shortest path computation
- Algorithm comparison (FastSSSP vs Dijkstra)
- Performance metrics display
- Educational tooltips and explanations
"""

import pygame
import math
import random
import time
from typing import List, Tuple, Dict, Optional
from sssp_algorithm import Graph, FastSSSP, dijkstra_sssp


class Node:
    """Represents a graph node in the game."""
    def __init__(self, x: int, y: int, id: int):
        self.x = x
        self.y = y
        self.id = id
        self.radius = 20
        self.color = (100, 200, 255)  # Light blue
        self.highlight_color = (255, 255, 100)  # Yellow highlight
        self.selected = False

    def draw(self, screen: pygame.Surface, is_source: bool = False,
             is_target: bool = False, distance: float = float('inf')):
        """Draw the node on the screen."""
        # Determine color based on state
        if self.selected:
            color = self.highlight_color
        elif is_source:
            color = (100, 255, 100)  # Green for source
        elif is_target:
            color = (255, 100, 100)  # Red for target
        else:
            color = self.color

        # Draw circle
        pygame.draw.circle(screen, color, (self.x, self.y), self.radius)

        # Draw node ID
        font = pygame.font.SysFont('Arial', 14)
        id_text = font.render(str(self.id), True, (0, 0, 0))
        screen.blit(id_text, (self.x - 8, self.y - 8))

        # Draw distance if finite
        if distance != float('inf'):
            dist_text = font.render(f"{distance:.1f}", True, (0, 100, 0))
            screen.blit(dist_text, (self.x - 15, self.y + 25))

    def is_clicked(self, pos: Tuple[int, int]) -> bool:
        """Check if the node was clicked."""
        dx = pos[0] - self.x
        dy = pos[1] - self.y
        return math.sqrt(dx*dx + dy*dy) <= self.radius


class Edge:
    """Represents a weighted edge in the game."""
    def __init__(self, node1: Node, node2: Node, weight: float = 1.0):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        self.color = (150, 150, 150)  # Gray
        self.highlight_color = (255, 200, 100)  # Orange highlight
        self.highlighted = False

    def draw(self, screen: pygame.Surface):
        """Draw the edge on the screen."""
        color = self.highlight_color if self.highlighted else self.color

        # Draw line
        pygame.draw.line(screen, color, (self.node1.x, self.node1.y),
                        (self.node2.x, self.node2.y), 3)

        # Draw weight
        mid_x = (self.node1.x + self.node2.x) // 2
        mid_y = (self.node1.y + self.node2.y) // 2

        font = pygame.font.SysFont('Arial', 12)
        weight_text = font.render(f"{self.weight:.1f}", True, (0, 0, 0))
        screen.blit(weight_text, (mid_x - 10, mid_y - 10))


class SSSPGame:
    """Main game class for the SSSP algorithm minigame."""

    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("SSSP Algorithm Minigame")

        # Game state
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.selected_node: Optional[Node] = None
        self.source_node: Optional[Node] = None
        self.target_node: Optional[Node] = None

        # Algorithm state
        self.fast_distances: Dict[int, float] = {}
        self.dijkstra_distances: Dict[int, float] = {}
        self.algorithm_running = False
        self.show_comparison = False

        # UI state
        self.mode = "place_nodes"  # place_nodes, add_edges, run_algorithm
        self.node_counter = 0
        self.message = ""
        self.message_timer = 0

        # Colors
        self.BG_COLOR = (240, 240, 240)
        self.PANEL_COLOR = (220, 220, 220)
        self.TEXT_COLOR = (50, 50, 50)

        # Fonts
        self.font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)

    def create_graph_from_game_state(self) -> Graph:
        """Convert current game state to a Graph object."""
        graph = Graph(len(self.nodes))

        # Add edges
        for edge in self.edges:
            graph.add_edge(edge.node1.id, edge.node2.id, edge.weight)

        return graph

    def run_sssp_algorithms(self):
        """Run both SSSP algorithms and store results."""
        if not self.source_node:
            self.show_message("Please select a source node first!")
            return

        graph = self.create_graph_from_game_state()

        # Run FastSSSP
        start_time = time.time()
        fast_algorithm = FastSSSP(graph)
        self.fast_distances, _ = fast_algorithm.compute_sssp(self.source_node.id)
        fast_time = time.time() - start_time

        # Run Dijkstra
        start_time = time.time()
        self.dijkstra_distances, _ = dijkstra_sssp(graph, self.source_node.id)
        dijkstra_time = time.time() - start_time

        # Show comparison
        self.show_comparison = True
        speedup = dijkstra_time / fast_time if fast_time > 0 else float('inf')
        self.show_message(f"FastSSSP: {fast_time:.4f}s, Dijkstra: {dijkstra_time:.4f}s, "
                         f"Speedup: {speedup:.2f}x")

    def show_message(self, text: str, duration: int = 180):  # 3 seconds at 60 FPS
        """Display a temporary message to the user."""
        self.message = text
        self.message_timer = duration

    def draw_ui(self):
        """Draw the user interface panels."""
        # Top panel
        pygame.draw.rect(self.screen, self.PANEL_COLOR, (0, 0, self.width, 60))

        # Title
        title = self.title_font.render("SSSP Algorithm Minigame", True, self.TEXT_COLOR)
        self.screen.blit(title, (20, 15))

        # Mode indicator
        mode_text = self.font.render(f"Mode: {self.mode.replace('_', ' ').title()}", True, self.TEXT_COLOR)
        self.screen.blit(mode_text, (300, 20))

        # Instructions based on mode
        if self.mode == "place_nodes":
            instructions = "Click to place nodes. Press SPACE to switch to adding edges."
        elif self.mode == "add_edges":
            instructions = "Click nodes to select/deselect. Press SPACE to run algorithms."
        else:
            instructions = "Press R to reset, C to clear graph."

        inst_text = self.font.render(instructions, True, self.TEXT_COLOR)
        self.screen.blit(inst_text, (300, 35))

        # Bottom panel
        pygame.draw.rect(self.screen, self.PANEL_COLOR, (0, self.height - 80, self.width, 80))

        # Node/Edge count
        count_text = self.font.render(f"Nodes: {len(self.nodes)}, Edges: {len(self.edges)}",
                                    True, self.TEXT_COLOR)
        self.screen.blit(count_text, (20, self.height - 60))

        # Source/Target info
        source_text = f"Source: {self.source_node.id if self.source_node else 'None'}"
        target_text = f"Target: {self.target_node.id if self.target_node else 'None'}"
        info_text = self.font.render(f"{source_text} | {target_text}", True, self.TEXT_COLOR)
        self.screen.blit(info_text, (200, self.height - 60))

        # Message
        if self.message and self.message_timer > 0:
            msg_text = self.font.render(self.message, True, (255, 100, 100))
            self.screen.blit(msg_text, (20, self.height - 35))
            self.message_timer -= 1

        # Algorithm comparison
        if self.show_comparison and self.target_node:
            fast_dist = self.fast_distances.get(self.target_node.id, float('inf'))
            dijkstra_dist = self.dijkstra_distances.get(self.target_node.id, float('inf'))

            comp_text = self.font.render(
                f"Target Distance - FastSSSP: {fast_dist:.1f}, Dijkstra: {dijkstra_dist:.1f}",
                True, self.TEXT_COLOR)
            self.screen.blit(comp_text, (400, self.height - 60))

    def draw_graph(self):
        """Draw all nodes and edges."""
        # Draw edges first (so they appear behind nodes)
        for edge in self.edges:
            edge.draw(self.screen)

        # Draw nodes
        for node in self.nodes:
            is_source = node == self.source_node
            is_target = node == self.target_node
            distance = self.fast_distances.get(node.id, float('inf'))
            node.draw(self.screen, is_source, is_target, distance)

    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click events."""
        # Check if clicking on existing node
        clicked_node = None
        for node in self.nodes:
            if node.is_clicked(pos):
                clicked_node = node
                break

        if self.mode == "place_nodes":
            if not clicked_node:
                # Place new node
                new_node = Node(pos[0], pos[1], self.node_counter)
                self.nodes.append(new_node)
                self.node_counter += 1
            else:
                # Toggle selection
                if self.selected_node == clicked_node:
                    self.selected_node = None
                    clicked_node.selected = False
                else:
                    if self.selected_node:
                        self.selected_node.selected = False
                    self.selected_node = clicked_node
                    clicked_node.selected = True

        elif self.mode == "add_edges":
            if clicked_node:
                if not self.selected_node:
                    # First selection
                    self.selected_node = clicked_node
                    clicked_node.selected = True
                elif self.selected_node == clicked_node:
                    # Deselect
                    self.selected_node = None
                    clicked_node.selected = False
                else:
                    # Create edge between selected and clicked nodes
                    # Check if edge already exists
                    edge_exists = False
                    for edge in self.edges:
                        if ((edge.node1 == self.selected_node and edge.node2 == clicked_node) or
                            (edge.node1 == clicked_node and edge.node2 == self.selected_node)):
                            edge_exists = True
                            break

                    if not edge_exists:
                        weight = random.uniform(1.0, 10.0)
                        new_edge = Edge(self.selected_node, clicked_node, weight)
                        self.edges.append(new_edge)

                    # Reset selection
                    self.selected_node.selected = False
                    clicked_node.selected = True
                    self.selected_node = clicked_node

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                    elif event.button == 3:  # Right click - set source/target
                        for node in self.nodes:
                            if node.is_clicked(event.pos):
                                if not self.source_node:
                                    self.source_node = node
                                    self.show_message(f"Set source to node {node.id}")
                                elif not self.target_node and node != self.source_node:
                                    self.target_node = node
                                    self.show_message(f"Set target to node {node.id}")
                                elif node == self.source_node:
                                    self.source_node = None
                                    self.show_message("Cleared source node")
                                elif node == self.target_node:
                                    self.target_node = None
                                    self.show_message("Cleared target node")
                                break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Cycle through modes
                        if self.mode == "place_nodes":
                            self.mode = "add_edges"
                            self.show_message("Switched to edge creation mode")
                        elif self.mode == "add_edges":
                            self.mode = "run_algorithm"
                            self.run_sssp_algorithms()
                        else:
                            self.mode = "place_nodes"
                            self.show_message("Switched to node placement mode")
                    elif event.key == pygame.K_r:
                        # Reset algorithm results
                        self.fast_distances = {}
                        self.dijkstra_distances = {}
                        self.show_comparison = False
                        self.show_message("Reset algorithm results")
                    elif event.key == pygame.K_c:
                        # Clear everything
                        self.nodes = []
                        self.edges = []
                        self.selected_node = None
                        self.source_node = None
                        self.target_node = None
                        self.fast_distances = {}
                        self.dijkstra_distances = {}
                        self.show_comparison = False
                        self.node_counter = 0
                        self.show_message("Cleared all nodes and edges")

            # Update
            # Nothing to update in this simple version

            # Draw
            self.screen.fill(self.BG_COLOR)
            self.draw_graph()
            self.draw_ui()
            pygame.display.flip()

            # Control frame rate
            clock.tick(60)

        pygame.quit()


def main():
    """Main function to run the game."""
    print("Welcome to the SSSP Algorithm Minigame!")
    print()
    print("Instructions:")
    print("- Click to place nodes")
    print("- Right-click nodes to set source (first) and target (second)")
    print("- Press SPACE to switch between modes:")
    print("  1. Place nodes")
    print("  2. Add edges (click two nodes to connect them)")
    print("  3. Run algorithm")
    print("- Press R to reset algorithm results")
    print("- Press C to clear everything")
    print()
    print("The game will show you the shortest paths computed by both")
    print("the new FastSSSP algorithm and traditional Dijkstra's algorithm!")
    print()

    # Check if pygame is available
    try:
        game = SSSPGame()
        game.run()
    except ImportError:
        print("Error: This game requires pygame. Install it with: pip install pygame")
    except Exception as e:
        print(f"Error running game: {e}")


if __name__ == "__main__":
    main()
