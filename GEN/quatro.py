import random

# Parâmetros configuráveis
population_size = 4
mutation_rate = 0.01
crossover_rate = 0.7
generations = 5
gene_length = 5  # Número de bits necessário para representar 21 valores

# Função a ser otimizada
def f(x):
    return x**2 - 3*x + 4

# Função para converter binário para decimal e mapeá-lo para o intervalo [-10, 10]
def binary_to_decimal(binary):
    return int("".join(map(str, binary)), 2)

def decode_individual(binary):
    decimal = binary_to_decimal(binary)
    # Mapeia [0, 31] para [-10, 10]
    x = (decimal / 31.0) * 20 - 10
    return x

# Função de aptidão baseada na transformação para o intervalo [-10, 10]
def fitness(individual):
    x = decode_individual(individual)
    return f(x)

# Criação de um indivíduo aleatório
def random_individual(length):
    return [random.randint(0, 1) for _ in range(length)]

# Criação da população inicial
population = [random_individual(gene_length) for _ in range(population_size)]
print(population)

# Função de Seleção por Torneio
def tournament_selection(population, fitness_values, tournament_size=3):
    tournament = random.sample(list(zip(population, fitness_values)), tournament_size)
    tournament_winner = max(tournament, key=lambda ind_fit: ind_fit[1])
    return tournament_winner[0]

# Função de Seleção por Torneio para o segundo melhor indivíduo
def tournament_selection_second_best(population, fitness_values, tournament_size=3):
    tournament = random.sample(list(zip(population, fitness_values)), tournament_size)
    tournament_winner = max(tournament, key=lambda ind_fit: ind_fit[1])
    tournament.remove(tournament_winner)
    tournament_second_best = max(tournament, key=lambda ind_fit: ind_fit[1])
    return tournament_second_best[0]

# Função de Cruzamento
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
    else:
        child1, child2 = parent1, parent2
    return child1, child2

# Função de Mutação
def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Inversão de bit
    return individual

# Evolução ao longo das gerações
for generation in range(generations):
    # Avaliação da aptidão da população
    fitness_values = [fitness(ind) for ind in population]
    
    # Geração da nova população
    new_population = []
    for _ in range(population_size // 2):
        parent1 = tournament_selection(population, fitness_values) # 1° do Torneio
        parent2 = tournament_selection_second_best(population, fitness_values) # 2° do Torneio
        child1, child2 = crossover(parent1, parent2, crossover_rate)
        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)
        new_population.extend([child1, child2]) # População formada por filhos
    population = new_population
    
    # Melhor indivíduo da geração
    best_individual = max(population, key=lambda ind: fitness(ind))
    best_fitness = fitness(best_individual)
    print(f"Geração {generation + 1}: Melhor Aptidão = {best_fitness}, Melhor Indivíduo = {best_individual}, Valor x = {decode_individual(best_individual)}")

# Melhor indivíduo após todas as gerações
best_individual = max(population, key=lambda ind: fitness(ind))
best_fitness = fitness(best_individual)
