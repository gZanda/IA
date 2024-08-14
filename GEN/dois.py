import random

population_size = 4
gene_length = 5
mutation_rate = 0.01
crossover_rate = 0.8
generations = 5

def random_individual(length):
    return [random.randint(0, 1) for _ in range(length)]

def binary_to_decimal(binary):
    return int("".join(map(str, binary)), 2)

def fitness(individual):
    x = binary_to_decimal(individual)
    return x ** 2

def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    return population[random.choices(range(len(population)), selection_probs)[0]]

def crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() > crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        return child1
    return parent1

def mutation(individual, mutation_rate=2):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Inversão de bit
    return individual

# População Inicial
population = [random_individual(gene_length) for _ in range(population_size)]

while generations > 0:
    fitness_values = [fitness(ind) for ind in population]
    parent1 = roulette_wheel_selection(population, fitness_values)
    parent2 = roulette_wheel_selection(population, fitness_values)
    child1 = crossover(parent1, parent2, crossover_rate)
    child2 = crossover(parent2, parent1, crossover_rate)
    child1 = mutation(child1, mutation_rate)
    child2 = mutation(child2, mutation_rate)
    population = [parent1, parent2, child1, child2]
    generations -= 1