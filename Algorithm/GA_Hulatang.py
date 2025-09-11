import random
import copy
import numpy as np
import time
from matplotlib import pyplot as plt
import torch
from matplotlib.patches import Circle


class Chromosome:
    """
    Class Chromosome represents one chromosome which consists of genetic code and value of
    fitness function.
    Genetic code represents potential solution to problem - the list of locations that are selected
    as medians.
    """

    def __init__(self, content, fitness):
        self.content = content
        self.fitness = fitness

    def __str__(self): return "%s f=%f" % (self.content, self.fitness)

    def __repr__(self): return "%s f=%f" % (self.content, self.fitness)


class GeneticAlgorithm:

    def __init__(self, n, m, p, cost_matrix, r, demand, costs,
                 B=None, C=None,
                 w1=0.40, w2=0.20, w3=0.25, w4=0.15,
                 iterations=200, generation_size=50, reproduction_size=20, mutation_prob=0.1):
        """
        n: number of users
        m: number of candidate facilities
        p: number of facilities to select
        cost_matrix: a distance matrix of shape [m, n]
        r: coverage radius
        demand: demand for each user, shape [n]
        costs: facility rental costs, shape [m]
        B: agglomeration benefit per candidate (length m)
        C: competition cost per candidate (length m)
        w1..w4: weights for objective terms to match Gurobi
        """
        self.time = None
        self.user_num = n
        self.fac_num = m
        self.p = p
        self.r = r
        # Weights to match Gurobi objective
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.w4 = float(w4)
        self.cost_matrix = cost_matrix
        self.demand = demand
        self.costs = costs  # 新增：设施成本
        # Agglomeration and competition components
        self.B = np.array(B if B is not None else np.zeros(m), dtype=float)
        self.C = np.array(C if C is not None else np.zeros(m), dtype=float)

        self.iterations = int(iterations)  # Maximal number of iterations
        self.current_iteration = 0
        self.generation_size = int(generation_size)  # Number of individuals in one generation
        self.reproduction_size = int(reproduction_size)  # Number of individuals for reproduction
        self.mutation_prob = float(mutation_prob)  # Mutation probability
        self.top_chromosome = None  # Chromosome that represents solution of optimization process

    def mutation(self, chromosome):
        """
        Applies mutation over chromosome with probability self.mutation_prob
        In this process, a randomly selected median is replaced with a randomly selected demand point.
        """
        mp = random.random()
        if mp < self.mutation_prob:
            i = random.randint(0, len(chromosome) - 1)
            # Candidate facilities not in the current chromosome
            available_facilities = [fac for fac in range(self.fac_num) if fac not in chromosome]
            if not available_facilities:  # If all facilities are in the chromosome, skip mutation
                return chromosome
            # Replace selected median with a randomly selected new facility
            chromosome[i] = random.choice(available_facilities)
        return chromosome

    def crossover(self, parent1, parent2):
        identical_elements = [element for element in parent1 if element in parent2]
        if len(identical_elements) == len(parent1):
            return parent1, None
        exchange_vector_for_parent1 = [element for element in parent1 if element not in identical_elements]
        exchange_vector_for_parent2 = [element for element in parent2 if element not in identical_elements]
        c = random.randint(0, min(len(exchange_vector_for_parent1), len(exchange_vector_for_parent2)) - 1)
        for i in range(c):
            exchange_vector_for_parent1[i], exchange_vector_for_parent2[i] = exchange_vector_for_parent2[i], \
                exchange_vector_for_parent1[i]
        child1 = identical_elements + exchange_vector_for_parent1
        child2 = identical_elements + exchange_vector_for_parent2
        return child1, child2

    def fitness(self, chromosome):
        """
        Calculates the fitness of a given chromosome based on the new objective function
        to match the Gurobi model:
            Maximize: w1 * covered_demand + w3 * sum(B[j]) - w4 * sum(C[j]) - w2 * sum(rent[j])
        Returns the negative value because GA minimizes fitness.

        - chromosome: A list of p selected facility indices.
        """
        # 1. Coverage term: demand covered within radius r
        selected_distances = self.cost_matrix[chromosome, :]
        coverage_matrix = (selected_distances <= self.r).astype(int)
        user_covered = (np.sum(coverage_matrix, axis=0) >= 1).astype(float)
        covered_demand = float(np.sum(self.demand * user_covered))

        # 2. Agglomeration and competition components for selected facilities
        sum_B = float(np.sum(self.B[chromosome]))
        sum_C = float(np.sum(self.C[chromosome]))

        # 3. Rent cost of selected facilities
        sum_rent = float(np.sum(self.costs[chromosome]))

        # 4. Objective (to maximize)
        objective_value = self.w1 * covered_demand + self.w3 * sum_B - self.w4 * sum_C - self.w2 * sum_rent

        # 4. Return negative value because GA minimizes fitness
        return -objective_value

    def initial_random_population(self):
        """ Creates initial population """
        init_population = []
        for _ in range(self.generation_size):
            # Randomly select p unique facilities
            rand_medians = random.sample(range(self.fac_num), self.p)
            init_population.append(rand_medians)

        init_population = [Chromosome(content, self.fitness(content)) for content in init_population]
        self.top_chromosome = min(init_population, key=lambda chromo: chromo.fitness)
        print("Initial top solution: %s" % self.top_chromosome)
        return init_population

    def selection(self, chromosomes):
        """Ranking-based selection method"""
        chromosomes.sort(key=lambda x: x.fitness)
        L = self.reproduction_size
        selected_chromosomes = []
        for i in range(self.reproduction_size):
            j = L - np.floor((-1 + np.sqrt(1 + 4 * random.uniform(0, 1) * (L ** 2 + L))) / 2)
            selected_chromosomes.append(chromosomes[int(j)])
        return selected_chromosomes

    def create_generation(self, for_reproduction):
        """ Creates new generation """
        new_generation = []
        while len(new_generation) < self.generation_size:
            parents = random.sample(for_reproduction, 2)
            child1, child2 = self.crossover(parents[0].content, parents[1].content)
            self.mutation(child1)
            new_generation.append(Chromosome(child1, self.fitness(child1)))
            if child2 is not None and len(new_generation) < self.generation_size:
                self.mutation(child2)
                new_generation.append(Chromosome(child2, self.fitness(child2)))
        return new_generation

    def optimize(self):
        start_time = time.time()
        chromosomes = self.initial_random_population()
        while self.current_iteration < self.iterations:
            for_reproduction = self.selection(chromosomes)
            chromosomes = self.create_generation(for_reproduction)
            self.current_iteration += 1
            chromosome_with_min_fitness = min(chromosomes, key=lambda chromo: chromo.fitness)
            if chromosome_with_min_fitness.fitness < self.top_chromosome.fitness:
                self.top_chromosome = chromosome_with_min_fitness
        end_time = time.time()
        self.time = end_time - start_time
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\nFinal top solution: %s" % self.top_chromosome)
        print('Time: {:0>2}:{:0>2}:{:05.4f}'.format(int(hours), int(minutes), seconds))

    def get_solution_details(self):
        """
        Returns detailed information about the best solution found
        """
        if self.top_chromosome is None:
            return None

        chromosome = self.top_chromosome.content

        # Calculate coverage
        selected_distances = self.cost_matrix[chromosome, :]
        coverage_matrix = (selected_distances <= self.r).astype(int)
        user_covered = (np.sum(coverage_matrix, axis=0) >= 1).astype(float)
        covered_demand = np.sum(self.demand * user_covered)
        total_demand = np.sum(self.demand)
        coverage_rate = covered_demand / total_demand if total_demand > 0 else 0

        # Components
        sum_rent = float(np.sum(self.costs[chromosome]))
        sum_B = float(np.sum(self.B[chromosome]))
        sum_C = float(np.sum(self.C[chromosome]))

        # Objective value (positive, as it should be maximized)
        objective_value = -float(self.top_chromosome.fitness)

        return {
            'selected_facilities': chromosome,
            'objective_value': objective_value,
            'covered_demand': covered_demand,
            'total_demand': total_demand,
            'coverage_rate': coverage_rate,
            'sum_rent': sum_rent,
            'sum_B': sum_B,
            'sum_C': sum_C,
            'w1': self.w1, 'w2': self.w2, 'w3': self.w3, 'w4': self.w4,
            'fitness': float(self.top_chromosome.fitness)
        }