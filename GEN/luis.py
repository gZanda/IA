import numpy as np

# Função objetivo a ser maximizada
def fitness(x):
    """
    Calcula o valor da função f(x) = x^2 - 3x + 4.
    """
    return x**2 - 3*x + 4

# Avalia o fitness de um indivíduo binário
def evaluate(individual, min_val, max_val):
    """
    Converte o indivíduo binário em um valor decimal no intervalo [min_val, max_val] e calcula seu fitness.
    """
    x = binary_to_decimal(individual, min_val, max_val)
    return fitness(x)

# Converte uma representação binária para decimal
def binary_to_decimal(binary_individual, min_val, max_val):
    """
    Converte um vetor binário para um número decimal no intervalo [min_val, max_val].
    """
    binary_str = ''.join(str(bit) for bit in binary_individual)  # Converte o vetor binário em uma string
    decimal_value = int(binary_str, 2)  # Converte a string binária para um número decimal
    num_bits = len(binary_individual)
    range_size = max_val - min_val
    return min_val + (decimal_value / (2**num_bits - 1)) * range_size  # Mapeia o valor decimal para o intervalo desejado

# Gera a população inicial
def initialize_population(pop_size, bit_length):
    """
    Gera uma população inicial de indivíduos binários aleatórios.
    """
    return [np.random.randint(2, size=bit_length) for _ in range(pop_size)]

# Aplica mutação em um indivíduo
def mutate(individual, mutation_rate):
    """
    Aplica mutação em um indivíduo com uma taxa de mutação dada.
    """
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]  # Inverte o bit (0 para 1 e vice-versa)
    return individual

# Aplica crossover entre dois pais para criar um filho
def crossover(parent1, parent2, crossover_rate):
    """
    Realiza o crossover entre dois pais para gerar um filho.
    """
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))  # Escolhe um ponto de crossover aleatório
        return np.concatenate((parent1[:point], parent2[point:]))  # Combina os pais para criar um filho
    return parent1  # Se o crossover não ocorre, retorna o primeiro pai como filho

# Seleção por torneio
def tournament_selection(population, fitnesses, tournament_size):
    """
    Seleciona indivíduos para a próxima geração usando seleção por torneio.
    """
    selected = []
    for _ in range(len(population)):
        indices = np.random.choice(len(population), tournament_size, replace=False)  # Seleciona indivíduos aleatórios para o torneio
        best = indices[np.argmax([fitnesses[i] for i in indices])]  # Escolhe o melhor indivíduo do torneio
        selected.append(population[best])
    return selected

# Função principal do Algoritmo Genético
def genetic_algorithm(pop_size, bit_length, num_generations, crossover_rate, mutation_rate, tournament_size, min_val, max_val):
    """
    Executa o Algoritmo Genético para maximizar a função objetivo.
    """
    # Inicializa a população
    population = initialize_population(pop_size, bit_length)
    
    # Executa o algoritmo por um número de gerações
    for generation in range(num_generations):
        # Avalia a população
        fitnesses = [evaluate(individual, min_val, max_val) for individual in population]
        
        # Imprime o melhor fitness da geração atual
        print(f"Generation {generation}: Best fitness = {max(fitnesses)}")
        
        # Seleção dos melhores indivíduos para a próxima geração
        selected_population = tournament_selection(population, fitnesses, tournament_size)
        next_population = []
        
        # Gera a próxima geração
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1 = mutate(crossover(parent1, parent2, crossover_rate), mutation_rate)
            child2 = mutate(crossover(parent2, parent1, crossover_rate), mutation_rate)
            next_population.append(child1)
            next_population.append(child2)
        
        # Atualiza a população
        population = next_population[:pop_size]
    
    # Avalia a população final
    final_fitnesses = [evaluate(individual, min_val, max_val) for individual in population]
    best_individual = population[np.argmax(final_fitnesses)]
    best_x = binary_to_decimal(best_individual, min_val, max_val)
    best_fitness = max(final_fitnesses)
    
    return best_x, best_fitness

# Parâmetros do Algoritmo Genético
pop_size = 4  # Número de indivíduos na população
bit_length = 10  # Número de bits para representar x
num_generations = 5  # Número de gerações
crossover_rate = 0.7  # Taxa de crossover
mutation_rate = 0.01  # Taxa de mutação
tournament_size = 2  # Tamanho do torneio
min_val = -10  # Valor mínimo de x
max_val = 10  # Valor máximo de x

# Executa o Algoritmo Genético
best_x, best_fitness = genetic_algorithm(pop_size, bit_length, num_generations, crossover_rate, mutation_rate, tournament_size, min_val, max_val)

# Exibe o melhor resultado encontrado
print(f"Melhor valor de x: {best_x}, Melhor fitness: {best_fitness}")