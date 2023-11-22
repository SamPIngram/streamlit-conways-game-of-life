import numpy as np
import time
import matplotlib.pyplot as plt
import math
from numba import njit, prange

def randomGrid(N):
    """ returns a grid of NxN random values"""
    return np.random.choice(vals, N*N, p=[0.2, 0.8]).reshape(N,N)

def update_grid_orig(img, grid, N):

    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):

            # compute 8-neighbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulation takes place on a toroidal surface
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] + 
                        grid[(i-1)%N, j] + grid[(i+1)%N, j] + 
                        grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                        grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/255)
            
            # apply Conway's rules
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON
    
    # update data
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img

@njit
def update_grid_numba(grid, N):

    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):

            # compute 8-neighbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulation takes place on a toroidal surface
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] + 
                        grid[(i-1)%N, j] + grid[(i+1)%N, j] + 
                        grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                        grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/255)
            
            # apply Conway's rules
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON
    
    # update data
    return newGrid

@njit(parallel=True)
def update_grid_numba_p(grid, N):

    newGrid = grid.copy()
    for i in prange(N):
        for j in range(N):

            # compute 8-neighbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulation takes place on a toroidal surface
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] + 
                        grid[(i-1)%N, j] + grid[(i+1)%N, j] + 
                        grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                        grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])/255)
            
            # apply Conway's rules
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON
    
    # update data
    return newGrid


if __name__ == '__main__':
    runs = 100
    N = 4000

    ON = 255
    OFF = 0
    vals = [ON, OFF]

    grid = randomGrid(N)
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')

    # time_start = time.perf_counter()
    # for i in range(runs):
    #     update_grid_orig(img, grid, N)
    # time_end = time.perf_counter()
    # print(f"Original: {time_end - time_start:0.4f} seconds")

    time_start = time.perf_counter()
    for i in range(runs):
        newGrid = update_grid_numba(grid, N)
        img.set_data(newGrid)
        grid[:] = newGrid[:]
    time_end = time.perf_counter()
    print(f"Numba: {time_end - time_start:0.4f} seconds")

    time_start = time.perf_counter()
    for i in range(runs):
        newGrid = update_grid_numba_p(grid, N)
        img.set_data(newGrid)
        grid[:] = newGrid[:]
    time_end = time.perf_counter()
    print(f"Numba Parallel: {time_end - time_start:0.4f} seconds")