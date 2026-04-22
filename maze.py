from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import random

def maze_path_length(maze: list[list[int]], start: tuple = (0, 0), end: tuple | None = None) -> int:
    if end is None:
        end = (len(maze) - 1, len(maze[0]) - 1)
    startx, starty = start
    endx, endy = end
    if maze[startx][starty] == 1 or maze[endx][endy] == 1:
        return -1
    r, c = len(maze), len(maze[0])
    visited = [[False for _ in range(c)] for _ in range(r)]
    visited[startx][starty] = True
    queue = deque()
    queue.append((startx, starty, 0))
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while queue:
        x, y, dist = queue.popleft()
        if (x, y) == (endx, endy):
            return dist
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < r and 0 <= ny < c and not visited[nx][ny] and maze[nx][ny] == 0:
                visited[nx][ny] = True
                queue.append((nx, ny, dist + 1))
    return -1

def visualize_maze(maze: list, text: bool = False, start: tuple = (0, 0), end: tuple | None = None):
    if end is None:
        end = (len(maze) - 1, len(maze[0]) - 1)
    if text:
        for row in maze:
            print("".join(['#' if cell == 1 else '.' for cell in row]))
    else:
        maze = np.array(maze)
        fig, ax = plt.subplots()
        ax.imshow(maze, cmap='gray_r', origin='upper')
        nrows, ncols = maze.shape
        ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_aspect('equal')
        for (r, c) in [start, end]:
            ax.plot(c, r, 'rx', markersize=14, markeredgewidth=3)

        plt.show()

def generate_maze(n: int = 20, m: int = 20) -> list[list[int]]:
    maze = np.ones((n, m), dtype=int)
    visited = np.zeros((n, m), dtype=bool)

    def neighbors(r, c):
        dirs = [(-2,0), (2,0), (0,-2), (0,2)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < m and not visited[nr, nc]:
                yield nr, nc, dr, dc

    def carve(r, c):
        visited[r, c] = True
        maze[r, c] = 0
        for nr, nc, dr, dc in neighbors(r, c):
            if not visited[nr, nc]:
                maze[r + dr//2, c + dc//2] = 0
                carve(nr, nc)

    carve(0, 0)
    maze[0, 0] = 0
    maze[-1, -1] = 0
    return maze.tolist()

def _bfs_path(maze, start, end):
    n, m = len(maze), len(maze[0])
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            path = []
            while True:
                path.append((r, c))
                if prev[(r, c)] is not None:
                    r, c = prev[(r, c)]
                else:
                    return path[::-1]
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < m and maze[nr][nc] == 0 and (nr, nc) not in prev:
                prev[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return None

def perturb_maze(maze: list[list[int]], open_attempts=30):
    maze = [row[:] for row in maze]
    n, m = len(maze), len(maze[0])
    start, end = (0,0), (n-1,m-1)

    path = _bfs_path(maze, start, end)
    if not path or len(path) < 3:
        return maze

    old_len = len(path)

    # Block one random cell on the path (not start or end)
    block = random.choice(path[1:-1])
    maze[block[0]][block[1]] = 1

    # Try opening random walls until solvable again
    for _ in range(open_attempts):
        wall_cells = [(r,c) for r in range(n) for c in range(m) if maze[r][c] == 1 and (r,c) not in [block, start, end]]
        if not wall_cells:
            break
        r, c = random.choice(wall_cells)
        maze[r][c] = 0
        new_path = _bfs_path(maze, start, end)
        if new_path:
            return maze
        else:
            maze[r][c] = 1

    return maze


if __name__ == "__main__":
    m = generate_maze(19, 19)
    visualize_maze(m, text=False)
    visualize_maze(perturb_maze(m), text=False)