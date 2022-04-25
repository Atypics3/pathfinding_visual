## Using A* (A-star) pathfinding algorithm (https://en.wikipedia.org/wiki/A*_search_algorithm):
## -------------------------------
# is a modification of Dijkstra's algo., optimized for a single destination
# finds paths to one location, or the closest to several locations
# prioritizes paths that appear to be closer to the end point (destination)
# maintains a tree of paths orignating at the start node and extending those paths 
# one edge at a time until the end (condition) is reached.
#
# specifically, f(n) = g(n) + h(n), where:
# - n is the next node on the path 
# - g(n) is the cost of the path from the starting node to n
# - h(n) is a heuristics ftn. that estimates the cost of the cheapest path from n to the end (or goal)
#
# The algorithm ends when the path it chooses to extend is a path from start to end 
# or if there are no paths avaiable to be extended
# Guaranteed to return a least-cost path from start to goal
# could also use breadth first search or dijkstra's algo.
#
# As an example, when searching for the shortest route on a map, 
# h(x) might represent the straight-line distance to the goal, 
# since that is physically the smallest possible distance between any two points. 
# 
# If the heuristic h satisfies the additional condition h(x) ≤ d(x, y) + h(y) for every edge (x, y)
# of the graph (where d denotes the length of that edge), then h is called monotone, or consistent. 
# With a consistent heuristic, A* is guaranteed to find an optimal path without processing
# any node more than once and A* is equivalent to running Dijkstra's algorithm with the 
# reduced cost d'(x, y) = d(x, y) + h(y) − h(x). 
## -------------------------------
import pygame
import math
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption('A* Path Finding Algorithm')

## colors:
RED = (255, 0, 0)
BLUE = (0, 255, 0)
GREEN = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE= (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width                # x-coord
        self.y = col * width                # y-coord
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
    
    # gets the position
    def get_pos(self):
        return self.row, self.col # returns (x-coord, y-coord)
    
    # if the node has been looked at already (red) or not (white)
    def is_closed(self):
        return self.color == RED

    # if the node is in a open set (green) or not (white)
    def is_open(self):
        return self.color == GREEN

    # if the node is blocked by a barrier node (black) or not (white)
    def is_barrier(self):
        return self.color == BLACK

    #  if the node is in a start node (orange) or not (white)
    def is_start(self):
        return self.color == ORANGE

    # if the node is in a end node (purple) or not (white)
    def is_end(self):
        return self.color == TURQUOISE


    # if the node needs to be reset
    def reset(self):
        self.color = WHITE

    # makes the node closed (or red)
    def make_closed(self):
        self.color = RED
    
    # makes the node open (or green)
    def make_open(self):
        self.color = GREEN

    # makes the node a barrier node (or black)
    def make_barrier(self):
        self.color = BLACK
    
    # makes the node a start node (or orange)
    def make_start(self):
        self.color = ORANGE
    
    # makes the node a end node (or turquoise)
    def make_end(self):
        self.color = TURQUOISE

    # makes the node a path node (or purple)
    def make_path(self):
        self.color = PURPLE

    # enables drawing in the window
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
    
    # checks down, up, right, and left and appends a extra row, if needed and not a barrier node
    def update_neighbors(self, grid):
        self.neighbors = []
        # down
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        # up
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        # right
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        # left
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    # for comparison of two nodes (less than)
    def __lt__(self, other):
        return False


def h(p1, p2):
    x1, y1 = p1                             # point 1 = (x1, y1)
    x2, y2 = p2                             # point 2 = (x2, y2)
    return abs(x1 - x2) + abs(y1 - y2)      # gets the distance between the two points


# makes the current node a part of the path and continously draws said path as long as the current node is in came_from
def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

# A*
def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()              
    open_set.put((0, count, start))         # puts the value of the f score, count, and start into the open set
    came_from = {}                          # keeps track of where we came from

    # g score, keeps track of current shortest distance between one node to another
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0

    # f score, want to see how far away the start node is to the end node
    f_score = {node: float("inf") for row in grid for node in row}
    # gets the heuristics distance from h score
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]         # gets minimum element of open_set
        open_set_hash.remove(current)       # removes said element from the hash

        if current == end:                  # if found, then the shortest path has been found
            reconstruct_path(came_from, end , draw)
            return True
        
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            # if a better way to a neighbor has been found, update current path
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score

                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())

                if neighbor not in open_set_hash:
                    count+= 1
                    # puts in the new neighbor into the path
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()  # marks the neighbor as open
        
        draw()

        # if current node isn't the start node, then close it
        if current != start:
            current.make_closed()
    
    return False



# generates a grid that has drawable nodes inside it
def make_grid(rows, width):
    grid=[]
    gap = width // rows                     # gets the gap between the width and the row

    for i in range(rows):
        grid.append([])
        # populate the row with nodes
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    
    return grid


# draws the horizontal and vertical lines between the gaps in the grid
def draw_grid(win, rows, width):
    gap = width // rows                     # gets the gap between the width and the row
    
    # draws the horizontal lines for the grid
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        # draws the vertical lines for the grid
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


# draws the grid
def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for node in row:
            node.draw(win)
    
    # updates whatever is drawn onto the display window
    draw_grid(win, rows, width)
    pygame.display.update()


# helper function to get the position of a clicked node
def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y,x = pos

    row = y // gap
    col = x // gap

    return row, col


def main(win, width):
    ROWS = 50 
    grid = make_grid(ROWS, width)

    start = None                            # start node is chosen by user
    end = None                              # end node is chosen by user
    run = True                              

    while run:
        draw(win, grid, ROWS, width)

        # checking each event detected in window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # left mouse button
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]

                if not start and node != end:
                    start = node
                    start.make_start()
                
                elif not end and node != start:
                    end = node
                    end.make_end()

                elif node != end and node != start:
                    node.make_barrier()

            # right mouse button
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                node.reset()
                
                if node == start:
                    start = None
                
                elif node == end:
                    end = None

            # down arrow key
            if event.type == pygame.KEYDOWN:
                # SPACE key = starts the program
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    
                    # calls algorithm() which takes in draw(), represented by lambda
                    algorithm(lambda: draw(win,grid, ROWS, width), grid, start, end)
    
                #  C key = clears the screen
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()

main(WIN, WIDTH)