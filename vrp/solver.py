#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import os
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, product, permutations, chain
from collections import namedtuple

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def distance_dict():
    dist = {
        (i.index, j.index): length(i, j) for i, j in product(customers, customers) if i.index != j.index
    }
    return dist

def read_input(input_data):
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0]

    return customers, vehicle_capacity, vehicle_count, customer_count, depot

def read_solution_from_file():
    if os.path.exists('best-results/' + str(customer_count) + '.txt'):
        return open('best-results/' + str(customer_count) + '.txt', 'r').read()
    else:
        return None

def write_solution_to_file(output_data):    
    text_file = open('best-results/' + str(customer_count) + '.txt','w')
    text_file.write(output_data)
    text_file.close()

def trivial_solution():
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []
    
    remaining_customers = set(customers)
    remaining_customers.remove(depot)
    
    for v in range(0, vehicle_count):
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand*customer_count + customer.index)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    used.add(customer)
            remaining_customers -= used

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(depot,vehicle_tour[0])
            for i in range(0, len(vehicle_tour)-1):
                obj += length(vehicle_tour[i],vehicle_tour[i+1])
            obj += length(vehicle_tour[-1],depot)

    return vehicle_tours, obj

def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in model._vars.keys()
                                if vals[i, j] > 0.5)
        # find subtours which are not visiting the deopot
        tours = subtours_not_connected_to_depot(selected)
        for tour in tours:
            if len(tour) < customer_count:
                # add subtour elimination constr. for every pair of cities in tour
                model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in permutations(tour, 2)) <= len(tour)-1)
        
        # enforce capacity constraints
        depot_tours = get_tours(selected, False)
        for tour in depot_tours:
            tour_demands = sum(map(lambda node: demands[node], tour))
            if tour_demands > vehicle_capacity:
                min_vehicles_needed = math.ceil(tour_demands/vehicle_capacity)
                model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in permutations(tour, 2)) <= len(tour) - min_vehicles_needed)

def get_tours(selected, customer_data=True):
    vehicle_tours = [[] for i in range(vehicle_count)]
    for i, tup in enumerate(selected.select(0, '*')):
        neighbor = tup[1]
        while neighbor:
            if customer_data:
                vehicle_tours[i].append(customers[neighbor])
            else:
                vehicle_tours[i].append(neighbor)
            neighbor = selected.select(neighbor, '*')[0][1]
    return vehicle_tours

# Given a tuplelist of edges, find all subtours not containing depot
def subtours_not_connected_to_depot(edges):
    unvisited = list(range(1, customer_count))
    # First, remove all nodes connected to depot
    depot_connected = [j for i, j in edges.select(0, '*')]
    while depot_connected:
        current = depot_connected.pop()
        unvisited.remove(current)
        neighbors = [j for i, j in edges.select(current, '*')
                     if j in unvisited and j != 0]
        depot_connected += neighbors

    # find all subtours not connected to depot   
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

def gurobi_solution(trivial_solution = None):
    # define model
    m = gp.Model('VRP')

    # create variables
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')

    # inbound and outbound flow is always 1 for all customers
    m.addConstrs(vars.sum(i, '*') == 1 for i in range(1, customer_count))
    m.addConstrs(vars.sum('*', i) == 1 for i in range(1, customer_count))
    
    # depot has inbound and outbound flow smaller or equal to the number of vehicles
    m.addConstr(vars.sum(0, '*') == vehicle_count)
    m.addConstr(vars.sum('*', 0) == vehicle_count)
    
    # add MTZ constraints
    if True:
        u = m.addVars(range(1, customer_count), vtype=GRB.CONTINUOUS, name='u')
        m.addConstrs(0 <= u[i] for i in range(1, customer_count))
        m.addConstrs(u[i] <= vehicle_capacity - demands[i] for i in range(1, customer_count))
        tuples = []
        for i, j in [(i, j) for i, j in product(range(1, customer_count), range(1, customer_count)) if i != j]:
            if demands[i] + demands[j] <= vehicle_capacity:
                tuples.append((i, j))
        m.addConstrs(u[j]-u[i] >= demands[j]-vehicle_capacity*(1-vars[i,j]) for i,j in tuples)

    # if trivial solution is provided, use it as starting point        
    if trivial_solution is not None:
        # set all the variables to zero
        for i, j in dist.keys():
            vars[i, j].start = 0

        # set initial solution
        for tour in trivial_solution:
            if len(tour) > 0:
                first_visit = tour[0].index
                vars[depot.index, first_visit].start = 1
                for i in range(len(tour)-1):
                    vars[tour[i].index, tour[i+1].index].start = 1
                last_visit = tour[-1].index
                vars[last_visit, depot.index].start = 1

    # optimize model
    m._vars = vars

    if customer_count == 51:
        # m.Params.MIPFocus = 1 # focus on finding solution
        m.Params.MIPFocus = 1 # focus on improving bounds
        m.Params.BestObjStop = 540
        # m.Params.Cuts = 3 # aggressively looking for cuts
    elif customer_count == 101:
        m.Params.MIPFocus = 1
        m.Params.BestObjStop = 830
        # m.Params.Cuts = 3
    elif customer_count == 200:
        m.Params.MIPFocus = 1
        m.Params.BestObjStop = 3719
    elif customer_count == 421:
        m.Params.MIPFocus = 1
        m.Params.BestObjStop = 2000
    else:
        m.Params.MIPFocus = 1 # focus on finding feasible solutions
    
    # using subtour elimination
    if True:
        m.optimize()
    else: # use only MTZ constraints
        m.Params.LazyConstraints = 1
        m.optimize(subtourelim)

    # get solution
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    # get objective value
    obj = m.objVal
    
    # get tours
    vehicle_tours = get_tours(selected)

    return vehicle_tours, obj

def solve_it(input_data):
    global customers, vehicle_capacity, vehicle_count
    global customer_count, depot, dist, demands

    # parse the input
    customers, vehicle_capacity, vehicle_count, customer_count, depot = read_input(input_data)
    demands = list(map(lambda customer: customer.demand, customers))
    
    # calculate distances
    dist = distance_dict()

    # read solution if exists
    output_data = read_solution_from_file()

    if output_data is None:
        # get solution
        # vehicle_tours, _ = trivial_solution()
        vehicle_tours, obj = gurobi_solution()

        # prepare the solution in the specified output format
        output_data = '%.2f' % obj + ' ' + str(0) + '\n'
        for v in range(0, vehicle_count):
            output_data += str(depot.index) + ' ' + ' '.join([str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'
        
        # write solution to file
        write_solution_to_file(output_data)

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

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

