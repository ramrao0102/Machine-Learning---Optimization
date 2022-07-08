
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:43:07 2022

@author: ramra
"""

from numpy.random import randint
from numpy.random import rand

# objective function

def onemax(x):
    return -sum(x)

# tournament selection

def selection(pop, scores, k = 2):
    selection_ix = randint(len(pop))
    
    for ix in randint(0, len(pop), k):
        
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
            
    return pop[selection_ix]

# crossover two parents to create two children

def crossover(p1, p2, r_cross):
    
    c1 = p1.copy()
    c2 = p2.copy()
   
    if rand() < r_cross:
        pt = randint(1, len(p1)-2)
        
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]

    return [c1, c2]

# mutation operator

def mutation(bitstring, r_mut):
    
    for i in range(len(bitstring)):
        
        if rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]

# genetic algorithm

def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    
    pop = []
    
    for _ in range(n_pop):
        pop.append(randint(0,2, n_bits).tolist())
        
    best, best_eval = pop[0], objective(pop[0])

    for gen in range(n_iter):
        
        scores = []
        
        for c in pop:
            scores.append(objective(c))
        
                           
        for i in range(len(pop)):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                
                print('>%d,  new_best f(%s) = %.3f' % (gen, pop[i], scores[i]))
               
        selected = []
        
        for _ in range(n_pop):
            
            selected.append(selection(pop, scores))
                            
        children = []
        
        for i in range(0, n_pop, 2):
            
            p1, p2 = selected[i], selected[i+1]
                                   
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
                
        pop = children
        
    return [best, best_eval]

n_iter = 100

n_bits = 20

n_pop = 100

r_cross = 0.9

r_mut = 1.0/float(n_bits)

best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)

print("Done!")

print('f(%s) = %f' %(best, score))
        
    