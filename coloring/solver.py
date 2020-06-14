#!/usr/bin/python
# -*- coding: utf-8 -*-
import networkx as nx

def equitable_color(edges, node_count):
    # define graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # coloring
    max_degree = max(G, key=G.degree)
    coloring = nx.equitable_color(G, max_degree + 1)

    # transform to output format
    solution = [0]*node_count
    for key, value in coloring.items():
        solution[key] = value
    
    # specify output format
    colors = max(coloring.values()) + 1
    output_data = str(colors) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    
    return output_data

def greedy_color(edges, node_count):
    # define graph
    G = nx.Graph()
    G.add_edges_from(edges)

    print("[Greedy Color] Maximum Degree: ", max(map(lambda x: x[1], G.degree())))

    # coloring
    coloring = nx.greedy_color(G, strategy='DSATUR')

    # transform to output format
    solution = [0]*node_count
    for key, value in coloring.items():
        solution[key] = value
    
    # specify output format
    colors = max(coloring.values()) + 1
    output_data = str(colors) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    
    return output_data

def simple_assignment(node_count):
    solution = range(0, node_count)
    
    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # return simple_assignment(node_count)
    return greedy_color(edges, node_count)
    # return equitable_color(edges, node_count)

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

