import numpy as np
import threading
import time
import matplotlib.pyplot as plt

def generate_random_vector(size):
    return np.ones(size, dtype=int)

def sequential_sum(vector):
    total = 0
    for num in vector:
        total += num
    return total

def partial_sum(vector, start, end, result, index):
    total = 0
    for i in range(start, end):
        total += vector[i]
    result[index] = total

def parallel_sum(vector, num_threads):
    length = len(vector)
    part_size = length // num_threads
    threads = []
    result = [0] * num_threads

    for i in range(num_threads):
        start = i * part_size
        end = (i + 1) * part_size if i != num_threads - 1 else length
        thread = threading.Thread(target=partial_sum, args=(vector, start, end, result, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total = sum(result)
    return total

def measure_execution_time(vector_size, num_threads):
    vector = generate_random_vector(vector_size)

    start_time = time.time()
    seq_sum = sequential_sum(vector)
    seq_time = time.time() - start_time

    start_time = time.time()
    par_sum = parallel_sum(vector, num_threads)
    par_time = time.time() - start_time

    return seq_sum, par_sum, seq_time, par_time

vector_sizes = [1000, 10000, 100000, 10000000]
thread_counts = [1, 2, 4, 8, 16]

execution_times = {size: {'seq': 0, 'par': {threads: 0 for threads in thread_counts}} for size in vector_sizes}

for vector_size in vector_sizes:
    vector = generate_random_vector(vector_size)
    
    start_time = time.time()
    seq_sum = sequential_sum(vector)
    seq_time = time.time() - start_time
    execution_times[vector_size]['seq'] = seq_time

    print(f"\nTamanho do Vetor: {vector_size}")
    print(f"Soma Sequencial: {seq_sum} | Tempo Sequencial: {seq_time:.6f} s")
    
    for num_threads in thread_counts:
        start_time = time.time()
        par_sum = parallel_sum(vector, num_threads)
        par_time = time.time() - start_time

        result_check = "CORRETO" if seq_sum == par_sum else "INCORRETO"
        
        print(f"Threads: {num_threads} | Paralelo: {par_time:.6f} s | "
              f"Soma Paralela: {par_sum} | Resultado: {result_check}")
        
        execution_times[vector_size]['par'][num_threads] = par_time

# Plotando os gráficos e salvando como arquivos de imagem
for vector_size in vector_sizes:
    plt.figure(figsize=(10, 6))
    plt.plot(thread_counts, [execution_times[vector_size]['par'][threads] for threads in thread_counts], label='Paralelo', marker='o')
    plt.axhline(y=execution_times[vector_size]['seq'], color='r', linestyle='-', label='Sequencial')
    plt.xlabel('Número de Threads')
    plt.ylabel('Tempo de Execução (s)')
    plt.title(f'Tempo de Execução vs Número de Threads (Vetor de Tamanho {vector_size})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'tempo_execucao_vetor_{vector_size}.png')  # Salva o gráfico como um arquivo PNG
    plt.close()

print("Gráficos salvos como arquivos PNG.")

    
    # return np.ones(size, dtype=int)