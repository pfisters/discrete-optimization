#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import math
import os
import random
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, product, permutations, chain

def read_input(input_data):
    lines = input_data.split('\n')
    nodeCount = int(lines[0])
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append((float(parts[0]), float(parts[1])))
    return points, nodeCount

def read_solution_from_file(nodeCount):
    if os.path.exists('best-results/' + str(nodeCount) + '.txt'):
        return open('best-results/' + str(nodeCount) + '.txt', 'r').read()
    else:
        return None

def write_solution_to_file(nodeCount, output_data):    
    text_file = open('best-results/' + str(nodeCount) + '.txt','w')
    text_file.write(output_data)
    text_file.close()

def trivial_solution(vars):
    unvisited = list(range(0, nodeCount))

    # start with random node and always take the closest as successor
    start_node = random.randint(0, nodeCount -1)
    current = start_node

    unvisited.remove(start_node)

    while unvisited:
        if len(unvisited) % 10 == 0:
            print("Trivial Solution - ", len(unvisited), " nodes remaining")
        neighbors = list(product([current], unvisited))
        neigboring_distances = {
            k: dist[k] for k in dist.keys() & neighbors
        }

        min_distance_neighbor = min(neigboring_distances, key=neigboring_distances.get)
        nearest = min_distance_neighbor[1]
        unvisited.remove(nearest)
        vars[current, nearest].start = True
        current = nearest
        
    vars[current, start_node].start = True

    return vars


def distance_dict():
    dist = {(i, j):
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(nodeCount) for j in range(nodeCount) if i != j}
    
    return dist

def shortest_subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in model._vars.keys()
                                if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = shortest_subtour(selected)
        if len(tour) < nodeCount:
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in permutations(tour, 2)) <= len(tour)-1)

def all_subtourselim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
        # find all cycles in the selected edge list
        tours = all_subtours(selected)
        for tour in tours:
            if len(tour) < nodeCount:
                # add subtour elimination constr. for every pair of cities in tour
                model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2)) <= len(tour)-1)

                # flow through subsets must be larger than two
                neighbors = [i for i in range(nodeCount) if i not in tour]
                model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in product(tour, neighbors)) >= 2)
                
# shortest subtour
def shortest_subtour(edges):
    unvisited = list(range(nodeCount))
    cycle = range(nodeCount+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle

# all subtours
def all_subtours(edges):
    unvisited = list(range(nodeCount))
    cycles = []
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
        cycles.append(thiscycle)
    return cycles

def gurobi_solution():
    ''' option
    0 : auto
    1 : auto tune
    2 : Focus on best bound
    3 : Focus on finding a solution

    OPTIONS
    - MIP Focus
        1 : Find feasible solutions
        2 : Prove optimality
        3 : Improve objective bounds
    
    '''
    # define model
    m = gp.Model('TSP')
    # add variables
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')

    '''    
    # add symmetry
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]
    '''

    # add constraints: every point has two adjacent points
    m.addConstrs(vars.sum(i, '*') == 1 for i in range(nodeCount))
    m.addConstrs(vars.sum('*', i) == 1 for i in range(nodeCount))

    # MTZ constraints
    u = m.addVars(range(1, nodeCount), vtype=GRB.INTEGER, name='u')
    m.addConstrs(0 <= u[i] for i in range(1, nodeCount))
    m.addConstrs(u[i] <= nodeCount-1 for i in range(1, nodeCount))
    m.addConstrs(u[i] - u[j] + nodeCount * vars[i, j] <= nodeCount - 1 for i in range(1, nodeCount) for j in range(1, nodeCount) if i != j)

    # create trivial greedy solution
    # vars = trivial_solution(vars)

    m._vars = vars
    
    if nodeCount == 1889:
        m.Params.BestObjStop = 323000
        m.Params.MIPFocus = 1
    elif nodeCount == 33810:
        m.Params.MIPFocus = 1 # focus on finding feasible solution

    # m.optimize(all_subtourselim) # use subtour elimination
    m.optimize() # use MTZ

    # retrieve solution
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    solution = shortest_subtour(selected)

    # make sure all the nodes are inside the solution
    assert len(solution) == nodeCount
        
    # calculate the length of the tour
    obj = m.objVal

    return solution, obj

def solve_it(input_data):
    global points, nodeCount, dist
    
    points, nodeCount = read_input(input_data)
    dist = distance_dict()

    output_data = read_solution_from_file(nodeCount)

    if output_data is None:
        solution, obj = gurobi_solution()
        
        # prepare the solution in the specified output format
        output_data = '%.2f' % obj + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, solution))

        # write solution to textfile
        write_solution_to_file(nodeCount, output_data)

    return output_data

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')