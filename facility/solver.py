#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import os
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, product, permutations, chain

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def read_input(input_data):
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    return facilities, customers

def read_solution_from_file(facility_count, customer_count):
    if os.path.exists('best-results/' + str(facility_count) + '_' + str(customer_count) + '.txt'):
        return open('best-results/' + str(facility_count) + '_' + str(customer_count) + '.txt', 'r').read()
    else:
        return None

def write_solution_to_file(facility_count, customer_count, output_data):    
    text_file = open('best-results/' + str(facility_count) + '_' + str(customer_count) + '.txt','w')
    text_file.write(output_data)
    text_file.close()

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def distance_dict():
    dist = {
        (i.index, j.index): length(i.location, j.location) 
        for i, j in product(customers, facilities)
    }
    return dist

def trivial_solution():
    solution = [-1]*len(customers)
    capacity_remaining = [f.capacity for f in facilities]

    facility_index = 0
    for customer in customers:
        if capacity_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
        else:
            facility_index += 1
            assert capacity_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand

    used = [0]*len(facilities)
    for facility_index in solution:
        used[facility_index] = 1

    # calculate the cost of the solution
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    return solution, obj

def gurobi_solution(option):
    # define model
    m = gp.Model('facility_location')

    num_facilities = len(facilities)
    num_customers = len(customers)
    demands = list(map(lambda customer: customer.demand, customers))
    capacities = list(map(lambda facility: facility.capacity, facilities))
    setup_costs = list(map(lambda facility: facility.setup_cost, facilities))
    cartesian_prod = list(product(range(num_customers), range(num_facilities)))

    # create variables
    select = m.addVars(num_facilities, vtype=GRB.BINARY, name='select')
    assign = m.addVars(cartesian_prod, vtype=GRB.BINARY, name='assign')

    # add constraints
    ## every customer is served by exactly one facilits
    m.addConstrs((
        gp.quicksum(assign[(c,f)] for f in range(num_facilities)) == 1 for c in range(num_customers)), 
        name='Assignment')
    ## only setup facilities can be used
    m.addConstrs((
        assign[(c,f)] <= select[f] for c,f in cartesian_prod), 
        name='Setup2ship')
    ## capacity constraints are met
    m.addConstrs((
        gp.quicksum(assign[(c,f)]*demands[c] for c in range(num_customers)) <= capacities[f] for f in range(num_facilities)), 
        name='Capacity')
    # set objective
    m.setObjective(select.prod(setup_costs) + assign.prod(dist), GRB.MINIMIZE)
    
    # execute options
    if option == 1:
        m.Params.tuneResults = 1
        # tune the model
        m.tune()
        if m.tuneResultCount > 0:
            m.getTuneResult(0)
    elif option == 2:
        m.Params.MIPFocus = 3 # focus on improving bounds
    elif option == 3:
        m.Params.MIPFocus = 1 # focus on finding solutions
    
    # set stopping criteria

    # optimize objective
    m.optimize()

    # extract solution
    obj = m.objVal
    solution = [-1]*num_customers

    for customer, facility in assign.keys():
        if (abs(assign[customer, facility].x) > 1e-6):
            solution[customer] = facility

    return solution, obj


def solve_it(input_data):
    global facilities, customers, dist

    # parse the input
    facilities, customers = read_input(input_data)

    # calculate the distances
    dist = distance_dict()

    # read solution if exists
    output_data = read_solution_from_file(len(facilities), len(customers))

    if output_data is None:
        # solution, obj = trivial_solution()
        solution, obj = gurobi_solution(2)
        
        # prepare the solution in the specified output format
        output_data = '%.2f' % obj + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, solution))

        # write solution to file
        write_solution_to_file(len(facilities), len(customers), output_data)

    return output_data

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

