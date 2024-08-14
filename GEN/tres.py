import random

# Parâmetros configuráveis
population_size = 10    # Tamanho da população
mutation_rate = 0.01    # Taxa de mutação
crossover_rate = 0.7    # Taxa de crossover
generations = 4        # Número de gerações
gene_length = 5         # Número de bits para representar o número

# Função a ser otimizada
def f(x):
    return x**2 - 3*x + 4

def random_individual(length):
    return [random.randint(0, 1) for _ in range(length)]

# Criação da população inicial
population = [random_individual(gene_length) for _ in range(population_size)]

def binary_to_decimal(binary):
    return int("".join(map(str, binary)), 2)

def fitness(individual):
    x = binary_to_decimal(individual)
    return f(x)

# Avaliação da aptidão da população inicial
fitness_values = [fitness(ind) for ind in population]

def tournament_selection(population, fitness_values, tournament_size=3):
    tournament = random.sample(list(zip(population, fitness_values)), tournament_size)
    tournament_winner = max(tournament, key=lambda ind_fit: ind_fit[1])
    return tournament_winner[0]

# Seleção dos pais usando torneio
parent1 = tournament_selection(population, fitness_values)
parent2 = tournament_selection(population, fitness_values)

def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
    else:
        child1, child2 = parent1, parent2
    return child1, child2

# Gerando filhos
child1, child2 = crossover(parent1, parent2, crossover_rate)

def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Inversão de bit
    return individual

# Aplicando mutação nos filhos
child1 = mutation(child1, mutation_rate)
child2 = mutation(child2, mutation_rate)

new_population = []

# Gerando nova população
for _ in range(population_size // 2):  # Cada iteração gera 2 filhos
    parent1 = tournament_selection(population, fitness_values)
    parent2 = tournament_selection(population, fitness_values)
    child1, child2 = crossover(parent1, parent2, crossover_rate)
    child1 = mutation(child1, mutation_rate)
    child2 = mutation(child2, mutation_rate)
    new_population.extend([child1, child2])

population = new_population

for generation in range(generations):
    # Avaliação da aptidão da população
    fitness_values = [fitness(ind) for ind in population]
    
    # Geração da nova população
    new_population = []
    for _ in range(population_size // 2):
        parent1 = tournament_selection(population, fitness_values)
        parent2 = tournament_selection(population, fitness_values)
        child1, child2 = crossover(parent1, parent2, crossover_rate)
        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)
        new_population.extend([child1, child2])
    
    population = new_population
    
    # Melhor indivíduo da geração
    best_individual = max(population, key=lambda ind: fitness(ind))
    best_fitness = fitness(best_individual)
    print(f"Geração {generation + 1}: Melhor Aptidão = {best_fitness}, Melhor Indivíduo = {best_individual}")

# Melhor indivíduo após todas as gerações
best_individual = max(population, key=lambda ind: fitness(ind))
best_fitness = fitness(best_individual)
print(f"Melhor Indivíduo Final: {best_individual}, Aptidão = {best_fitness}")
