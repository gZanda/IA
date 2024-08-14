import random

def random_individual(length):
    return [random.randint(0, 1) for _ in range(length)]

# Criação de uma população inicial de 4 indivíduos com 5 bits cada
population_size = 4
gene_length = 5
population = [random_individual(gene_length) for _ in range(population_size)]
population

def binary_to_decimal(binary):
    return int("".join(map(str, binary)), 2)

def fitness(individual):
    x = binary_to_decimal(individual)
    return x ** 2

# Avaliação da aptidão da população inicial
fitness_values = [fitness(ind) for ind in population]
fitness_values

def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    return population[random.choices(range(len(population)), selection_probs)[0]]

# Selecionando dois pais
parent1 = roulette_wheel_selection(population, fitness_values)
parent2 = roulette_wheel_selection(population, fitness_values)
parent1, parent2

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Gerando dois filhos
child1, child2 = crossover(parent1, parent2)
child1, child2

def mutation(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Inversão de bit
    return individual

# Aplicando mutação nos filhos
mutation_rate = 0.01
child1 = mutation(child1, mutation_rate)
child2 = mutation(child2, mutation_rate)
child1, child2

# Atualizando a população com os novos filhos
new_population = population[:2] + [child1, child2]  # Mantém os 2 primeiros e adiciona os novos filhos
new_population

