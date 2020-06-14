#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import deque
from recordclass import recordclass

import numpy as np
import heapq

Item = recordclass('Item', 'index value weight')
Node = recordclass('Node', 'level value weight items')
PQNode = recordclass('PQNode', 'level value weight bound items')

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    
    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]
    
    def empty(self):
        return len(self._queue) == 0
    
    def lenght(self):
        return len(self._queue)


def greedy_density(items, capacity):
    # sort tuples according to their value density
    items_sorted = sorted(items, key = lambda x: x.value / x.weight, reverse=True)
    
    # takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items_sorted)

    for item in items_sorted:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def greedy_density_recursive(items, capacity):
    taken = [0]*len(items)
    n = len(items) - 1

    value = knap_sack_recurr_greedy(capacity, n, items, taken)
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def knap_sack_recurr_greedy(remain_cap, n, items, taken):
    val = items[n].value
    wtg = items[n].weight
    idx = items[n].index
    # initial condition
    if n == 0 or remain_cap == 0:
        return 0
    # weight of next item is higher than the remaining capacity
    if wtg > remain_cap:
        return knap_sack_recurr_greedy(remain_cap, n-1, items, taken)
    else:
        # take it
        took = taken
        took[idx] = 1
        take = val + knap_sack_recurr_greedy(remain_cap - wtg, n-1, items, took)
        # don't take it
        not_take = knap_sack_recurr_greedy(remain_cap, n-1, items, taken)
        return max(take, not_take)

def knap_sack_dynamic_programming(items, capacity):
    n = len(items)
    table = np.pad(np.zeros((capacity, n)), (0,1), 'constant')

    # build up K
    for j in range(1, n + 1):
        val = items[j-1].value
        wgt = items[j-1].weight

        for i in range(1, capacity + 1):
            if wgt > i:
                table[i,j] = table[i, j-1]
            else:
                table[i,j] = max(table[i,j-1], table[i - wgt, j-1] + val)
    
    # retrieve maximal value
    value = int(table[capacity,n])

    # define items taken
    taken = [0]*len(items)
    remain_cap = capacity
    
    for i in range(n, 0, -1):
        if table[remain_cap, i] != table[remain_cap, i-1]:
            taken[i-1] = 1
            remain_cap -= items[i-1].weight
        else:
            taken[i-1] = 0
    
    # create output
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def branch_and_bound_breadth_first(items, capacity):
    # sort for value weight density
    items = sorted(items, key=lambda x: x.value / x.weight, reverse=True)
    item_count = len(items)

    v = Node(level = -1, value = 0, weight = 0, items = [])
    Q = deque([])
    Q.append(v)

    maximum = 0
    best_items = []

    while(len(Q) != 0):
        # dequeue a node
        v = Q[0]
        Q.popleft()
        u = Node(level = None, weight = None, value = None, items = [])
        u.level = v.level + 1
        u.weight = v.weight + items[u.level].weight
        u.value = v.value + items[u.level].value
        u.items = list(v.items)       
        u.items.append(items[u.level].index)

        if (u.weight <= capacity and u.value > maximum):
            maximum = u.value
            best_items = u.items
        
        bound_u = bound(u, capacity, item_count, items)

        if (bound_u > maximum):
            Q.append(u)
        
        u = Node(level = None, weight = None, value = None, items = [])
        u.level = v.level + 1
        u.weight = v.weight
        u.value = v.value
        u.items = list(v.items)
      
        bound_u = bound(u, capacity, item_count, items)

        if (bound_u > maximum):
            Q.append(u)

    taken = [0]*len(items)
    for item in best_items:
        taken[item] = 1
    
    output_data = str(maximum) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def branch_and_bound_best_first(items, capacity):
    # sort for value weight density
    items = sorted(items, key=lambda x: x.value / x.weight, reverse=True)
    item_count = len(items)
    
    # create queue
    v = PQNode(level = -1, value = 0, weight = 0, bound = 0, items = [])
    v.bound = bound(v, capacity, item_count, items)
    Q = PriorityQueue()
    Q.push(v, v.bound)

    # initialize
    maximum = 0
    best_items = []

    while not Q.empty():
        v = Q.pop()
        if(v.bound > maximum):
            u = PQNode(level = None, weight = None, value = None, bound = None, items = [])
            u.level = v.level + 1
            u.weight = v.weight + items[u.level].weight
            u.value = v.value + items[u.level].value
            u.items = list(v.items)
            u.items.append(items[u.level].index)

            if(u.weight <= capacity and u.value > maximum):
                maximum = u.value
                best_items = u.items
            
            u.bound = bound(u, capacity, item_count, items)

            if (u.bound > maximum):
                Q.push(u, u.bound)
            
            u = PQNode(level = None, weight = None, value = None, bound = None, items = [])
            u.level = v.level + 1
            u.weight = v.weight
            u.value = v.value
            u.items = list(v.items)
            u.bound = bound(u, capacity, item_count, items)

            if (u.bound > maximum):
                Q.push(u, u.bound)
    
    taken = [0]*len(items)
    for item in best_items:
        taken[item] = 1
    
    # create output
    output_data = str(maximum) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def bound(u, capacity, item_count, items):
    if (u.weight >= capacity): 
        return 0
    else:
        result = u.value
        j = u.level + 1
        total_weight = u.weight

        while(j < item_count and total_weight + items[j].weight <= capacity):
            total_weight += items[j].weight
            result += items[j].value
            j += 1
        
        k = j

        if (k <= item_count - 1):
            result += (capacity - total_weight)*items[k].value / items[k].weight
        
        return result

def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    return branch_and_bound_breadth_first(items, capacity)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

