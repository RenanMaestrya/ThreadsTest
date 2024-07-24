import os
import numpy as np
import time
import concurrent.futures
import matplotlib.pyplot as plt

def generate_random_vector(size):
    return np.ones(size, dtype=int)

def sequential_sum(vector):
    return np.sum(vector)

def partial_sum(vector, start, end):
    return np.sum(vector[start:end])

def parallel_sum(vector, num_threads):
    length = len(vector)
    part_size = length // num_threads
    futures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * part_size
            end = (i + 1) * part_size if i != num_threads - 1 else length
            futures.append(executor.submit(partial_sum, vector, start, end))
    
    total = sum(f.result() for f in futures)
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

output_dir_base = "execution_results"
if not os.path.exists(output_dir_base):
    os.makedirs(output_dir_base)

execution_number = 1

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

    # Criando diretório específico para esta execução
    execution_dir = os.path.join(output_dir_base, f"execucao_{execution_number}")
    os.makedirs(execution_dir, exist_ok=True)

    # Plotando os gráficos e salvando como arquivos de imagem
    plt.figure(figsize=(10, 6))
    plt.plot(thread_counts, [execution_times[vector_size]['par'][threads] for threads in thread_counts], label='Paralelo', marker='o')
    plt.axhline(y=execution_times[vector_size]['seq'], color='r', linestyle='-', label='Sequencial')
    plt.xlabel('Número de Threads')
    plt.ylabel('Tempo de Execução (s)')
    plt.title(f'Tempo de Execução vs Número de Threads (Vetor de Tamanho {vector_size})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(execution_dir, f'tempo_execucao_vetor_{vector_size}.png'))  # Salva o gráfico como um arquivo PNG
    plt.close()

    execution_number += 1

print("Gráficos salvos como arquivos PNG.")
