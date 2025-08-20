"""
Grid-Based Pathfinding Minigame - SSSP Algorithm Demonstration

A grid-based game where you can place walls and find shortest paths,
demonstrating the FastSSSP algorithm's capabilities in pathfinding.

Features:
- Grid-based level with walls and open spaces
- Interactive wall placement and removal
- Start/end point selection
- Real-time shortest path calculation
- Algorithm performance comparison
- Visual path exploration
- Multiple difficulty levels
"""

import pygame
import random
import time
from typing import List, Tuple, Dict, Optional, Set
from sssp_algorithm import Graph, FastSSSP, dijkstra_sssp


class Cell:
    """Represents a cell in the grid."""
    def __init__(self, x: int, y: int, grid_x: int, grid_y: int, size: int):
        self.x = x  # Pixel coordinates
        self.y = y
        self.grid_x = grid_x  # Grid coordinates
        self.grid_y = grid_y
        self.size = size
        self.is_wall = False
        self.is_start = False
        self.is_end = False
        self.is_path = False
        self.distance = float('inf')
        self.visited = False

    def draw(self, screen: pygame.Surface):
        """Draw the cell on the screen."""
        # Determine color based on cell type
        if self.is_start:
            color = (0, 255, 0)  # Green for start
        elif self.is_end:
            color = (255, 0, 0)  # Red for end
        elif self.is_path:
            color = (0, 200, 255)  # Light blue for path
        elif self.is_wall:
            color = (50, 50, 50)  # Dark gray for walls
        elif self.visited:
            color = (200, 200, 255)  # Light blue for visited
        else:
            color = (255, 255, 255)  # White for empty cells

        # Draw rectangle
        pygame.draw.rect(screen, color, (self.x, self.y, self.size, self.size))

        # Draw grid lines
        pygame.draw.rect(screen, (200, 200, 200), (self.x, self.y, self.size, self.size), 1)

        # Draw distance if finite and not start/end
        if (self.distance != float('inf') and not self.is_start and not self.is_end
            and not self.is_wall and self.distance > 0):
            font = pygame.font.SysFont('Arial', 10)
            dist_text = font.render(f"{self.distance:.0f}", True, (0, 0, 0))
            screen.blit(dist_text, (self.x + 2, self.y + 2))


class GridPathfinder:
    """Main grid-based pathfinding game."""

    def __init__(self, width: int = 800, height: int = 600, grid_size: int = 20):
        pygame.init()
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cols = width // grid_size
        self.rows = height // grid_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Grid Pathfinder - SSSP Algorithm Demo")

        # Game state
        self.grid: List[List[Cell]] = []
        self.start_cell: Optional[Cell] = None
        self.end_cell: Optional[Cell] = None
        self.path: List[Cell] = []
        self.algorithm_running = False
        self.showing_path = False

        # Algorithm results
        self.fast_distances: Dict[Tuple[int, int], float] = {}
        self.dijkstra_distances: Dict[Tuple[int, int], float] = {}

        # Game mode
        self.mode = "place_walls"  # place_walls, set_points, run_algorithm
        self.mouse_held = False
        self.place_walls = True  # True for walls, False for empty

        # UI constants
        self.HEADER_HEIGHT = 50
        self.FOOTER_HEIGHT = 40
        self.PLAYABLE_HEIGHT = height - self.HEADER_HEIGHT - self.FOOTER_HEIGHT

        # Adjust grid size to fit in playable area
        self.cols = width // grid_size
        self.rows = self.PLAYABLE_HEIGHT // grid_size

        # UI
        self.font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)

        self.create_grid()

    def create_grid(self):
        """Create the grid of cells."""
        self.grid = []
        for y in range(self.rows):
            row = []
            for x in range(self.cols):
                cell_x = x * self.grid_size
                cell_y = y * self.grid_size + self.HEADER_HEIGHT  # Start below header
                cell = Cell(cell_x, cell_y, x, y, self.grid_size)
                row.append(cell)
            self.grid.append(row)

    def get_cell_at(self, pos: Tuple[int, int]) -> Optional[Cell]:
        """Get cell at pixel coordinates."""
        x, y = pos

        # Only consider clicks in the playable area
        if y < self.HEADER_HEIGHT or y >= self.HEADER_HEIGHT + self.PLAYABLE_HEIGHT:
            return None

        grid_x = x // self.grid_size
        grid_y = (y - self.HEADER_HEIGHT) // self.grid_size  # Adjust for header

        if 0 <= grid_x < self.cols and 0 <= grid_y < self.rows:
            return self.grid[grid_y][grid_x]
        return None

    def clear_pathfinding_data(self):
        """Clear all pathfinding-related data."""
        for row in self.grid:
            for cell in row:
                cell.distance = float('inf')
                cell.visited = False
                cell.is_path = False

        self.fast_distances = {}
        self.dijkstra_distances = {}
        self.path = []
        self.showing_path = False

    def create_graph_from_grid(self) -> Graph:
        """Convert grid to graph for SSSP algorithms."""
        if not self.start_cell:
            raise ValueError("No start cell set")

        num_cells = self.rows * self.cols
        graph = Graph(num_cells)

        # Add edges between adjacent non-wall cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

        for y in range(self.rows):
            for x in range(self.cols):
                cell = self.grid[y][x]
                if cell.is_wall:
                    continue

                cell_id = y * self.cols + x

                # Connect to adjacent cells
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.cols and 0 <= ny < self.rows:
                        neighbor = self.grid[ny][nx]
                        if not neighbor.is_wall:
                            neighbor_id = ny * self.cols + nx
                            # Use distance of 1 for adjacent cells, sqrt(2) for diagonal
                            weight = 1.0
                            graph.add_edge(cell_id, neighbor_id, weight)

        return graph

    def run_pathfinding_algorithms(self):
        """Run both pathfinding algorithms on the current grid."""
        if not self.start_cell or not self.end_cell:
            self.show_message("Please set both start and end points!")
            return

        if self.start_cell.is_wall or self.end_cell.is_wall:
            self.show_message("Start and end points cannot be walls!")
            return

        self.clear_pathfinding_data()
        self.algorithm_running = True

        try:
            # Convert grid to graph
            graph = self.create_graph_from_grid()
            start_id = self.start_cell.grid_y * self.cols + self.start_cell.grid_x
            end_id = self.end_cell.grid_y * self.cols + self.end_cell.grid_x

            # Run FastSSSP
            start_time = time.time()
            fast_algorithm = FastSSSP(graph)
            fast_distances, fast_predecessors = fast_algorithm.compute_sssp(start_id)
            fast_time = time.time() - start_time

            # Run Dijkstra
            start_time = time.time()
            dijkstra_distances, dijkstra_predecessors = dijkstra_sssp(graph, start_id)
            dijkstra_time = time.time() - start_time

            # Store results
            self.fast_distances = {}
            self.dijkstra_distances = {}

            for y in range(self.rows):
                for x in range(self.cols):
                    cell_id = y * self.cols + x
                    self.fast_distances[(x, y)] = fast_distances[cell_id]
                    self.dijkstra_distances[(x, y)] = dijkstra_distances[cell_id]

                    # Update cell distances
                    cell = self.grid[y][x]
                    cell.distance = fast_distances[cell_id]
                    if fast_distances[cell_id] != float('inf'):
                        cell.visited = True

            # Find and highlight the path
            self.find_and_highlight_path(fast_predecessors, start_id, end_id)

            # Calculate performance metrics
            speedup = dijkstra_time / fast_time if fast_time > 0 else float('inf')

            self.show_message(f"FastSSSP: {fast_time:.4f}s, Dijkstra: {dijkstra_time:.4f}s, "
                             f"Speedup: {speedup:.2f}x")

            self.algorithm_running = False

        except Exception as e:
            self.show_message(f"Error: {str(e)}")
            self.algorithm_running = False

    def find_and_highlight_path(self, predecessors, start_id, end_id):
        """Find and highlight the shortest path."""
        if self.fast_distances[(self.end_cell.grid_x, self.end_cell.grid_y)] == float('inf'):
            self.show_message("No path found!")
            return

        # Reconstruct path
        path = []
        current = end_id
        while current is not None:
            y = current // self.cols
            x = current % self.cols
            path.append(self.grid[y][x])
            current = predecessors[current]

        path.reverse()

        # Highlight path cells
        for cell in path:
            cell.is_path = True

        self.path = path
        self.showing_path = True

    def generate_random_maze(self):
        """Generate a random maze with walls."""
        # Clear existing walls
        for row in self.grid:
            for cell in row:
                cell.is_wall = False

        # Add some random walls (avoiding start/end if they exist)
        wall_probability = 0.25

        for y in range(self.rows):
            for x in range(self.cols):
                if random.random() < wall_probability:
                    cell = self.grid[y][x]
                    # Don't place walls on start/end points
                    if not ((self.start_cell and cell == self.start_cell) or
                           (self.end_cell and cell == self.end_cell)):
                        cell.is_wall = True

    def draw_ui(self):
        """Draw the user interface."""
        # Top panel
        pygame.draw.rect(self.screen, (220, 220, 220), (0, 0, self.width, self.HEADER_HEIGHT))

        # Title
        title = self.title_font.render("Grid Pathfinder - SSSP Algorithm Demo", True, (50, 50, 50))
        self.screen.blit(title, (20, 10))

        # Mode indicator
        mode_text = self.font.render(f"Mode: {self.mode.replace('_', ' ').title()}", True, (50, 50, 50))
        self.screen.blit(mode_text, (500, 15))

        # Instructions based on mode
        if self.mode == "place_walls":
            instructions = "Left click: toggle walls, Right click: set start/end, SPACE: change mode"
        elif self.mode == "set_points":
            instructions = "Left click: set start, Right click: set end, SPACE: run algorithms"
        else:
            instructions = "Running algorithms... Press R to reset"

        inst_text = self.font.render(instructions, True, (50, 50, 50))
        self.screen.blit(inst_text, (500, 30))

        # Bottom panel
        pygame.draw.rect(self.screen, (220, 220, 220), (0, self.height - self.FOOTER_HEIGHT, self.width, self.FOOTER_HEIGHT))

        # Status information
        start_text = f"Start: ({self.start_cell.grid_x if self.start_cell else '?'}, {self.start_cell.grid_y if self.start_cell else '?'})"
        end_text = f"End: ({self.end_cell.grid_x if self.end_cell else '?'}, {self.end_cell.grid_y if self.end_cell else '?'})"
        path_text = f"Path length: {len(self.path)-1 if self.path else 'N/A'}"

        status_text = self.font.render(f"{start_text} | {end_text} | {path_text}", True, (50, 50, 50))
        self.screen.blit(status_text, (20, self.height - 30))

    def draw_grid(self):
        """Draw all grid cells."""
        for row in self.grid:
            for cell in row:
                cell.draw(self.screen)

    def show_message(self, text: str):
        """Display a message (for now just print to console)."""
        print(f"[Game] {text}")

    def handle_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse clicks."""
        cell = self.get_cell_at(pos)
        if not cell:
            return

        if self.mode == "place_walls":
            if button == 1:  # Left click - toggle wall
                if not cell.is_start and not cell.is_end:
                    cell.is_wall = not cell.is_wall
                    self.clear_pathfinding_data()
            elif button == 3:  # Right click - set start/end
                if not cell.is_wall:
                    if not self.start_cell:
                        self.start_cell = cell
                        cell.is_start = True
                    elif not self.end_cell and cell != self.start_cell:
                        self.end_cell = cell
                        cell.is_end = True
                    elif cell == self.start_cell:
                        self.start_cell = None
                        cell.is_start = False
                    elif cell == self.end_cell:
                        self.end_cell = None
                        cell.is_end = False

        elif self.mode == "set_points":
            if button == 1 and not cell.is_wall:  # Left click - set start
                if self.start_cell:
                    self.start_cell.is_start = False
                self.start_cell = cell
                cell.is_start = True
                self.clear_pathfinding_data()
            elif button == 3 and not cell.is_wall:  # Right click - set end
                if self.end_cell:
                    self.end_cell.is_end = False
                self.end_cell = cell
                cell.is_end = True
                self.clear_pathfinding_data()

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
                    if event.button in [1, 3]:  # Left or right click
                        self.handle_click(event.pos, event.button)
                elif event.type == pygame.MOUSEMOTION:
                    if self.mouse_held and self.mode == "place_walls":
                        cell = self.get_cell_at(event.pos)
                        if cell and not cell.is_start and not cell.is_end:
                            cell.is_wall = self.place_walls
                            self.clear_pathfinding_data()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Cycle through modes
                        if self.mode == "place_walls":
                            self.mode = "set_points"
                            self.show_message("Switched to point placement mode")
                        elif self.mode == "set_points":
                            self.mode = "run_algorithm"
                            self.run_pathfinding_algorithms()
                        else:
                            self.mode = "place_walls"
                            self.show_message("Switched to wall placement mode")
                    elif event.key == pygame.K_r:
                        # Reset
                        self.clear_pathfinding_data()
                        self.show_message("Reset pathfinding data")
                    elif event.key == pygame.K_c:
                        # Clear all
                        for row in self.grid:
                            for cell in row:
                                cell.is_wall = False
                                cell.is_start = False
                                cell.is_end = False
                        self.start_cell = None
                        self.end_cell = None
                        self.clear_pathfinding_data()
                        self.show_message("Cleared all walls and points")
                    elif event.key == pygame.K_g:
                        # Generate random maze
                        self.generate_random_maze()
                        self.clear_pathfinding_data()
                        self.show_message("Generated random maze")
                    elif event.key == pygame.K_m:
                        # Toggle wall placement mode
                        self.place_walls = not self.place_walls
                        mode = "walls" if self.place_walls else "empty"
                        self.show_message(f"Mouse now places {mode}")

            # Update
            # Nothing to update in this simple version

            # Draw
            self.screen.fill((240, 240, 240))
            self.draw_grid()
            self.draw_ui()
            pygame.display.flip()

            # Control frame rate
            clock.tick(60)

        pygame.quit()


def print_instructions():
    """Print game instructions."""
    print("\n" + "="*60)
    print("           GRID PATHFINDER - SSSP DEMO")
    print("="*60)
    print("Navigate mazes and discover shortest paths!")
    print("See your FastSSSP algorithm in action on a grid.")
    print("="*60)
    print()
    print("INSTRUCTIONS:")
    print("• Left Click: Place/remove walls or set points")
    print("• Right Click: Set start/end points")
    print("• SPACE: Switch between modes:")
    print("  1. Place walls")
    print("  2. Set start/end points")
    print("  3. Run pathfinding algorithms")
    print("• R: Reset pathfinding results")
    print("• C: Clear all walls and points")
    print("• G: Generate random maze")
    print("• M: Toggle wall/empty placement mode")
    print()
    print("The game will show you:")
    print("• How FastSSSP finds optimal paths")
    print("• Comparison with Dijkstra's algorithm")
    print("• Performance differences")
    print("• Visual path exploration")
    print()


def main():
    """Main function to run the grid pathfinder game."""
    print_instructions()

    try:
        game = GridPathfinder()
        game.run()
    except Exception as e:
        print(f"Error running game: {e}")
        print("Make sure pygame is installed: pip install pygame")


if __name__ == "__main__":
    main()
