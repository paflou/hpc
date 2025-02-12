import subprocess
import csv
from statistics import mean
import os
import re

def modify_n(n):
    # Get absolute path to cuda.cu
    cuda_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cuda.cu'))
    
    if not os.path.exists(cuda_path):
        raise FileNotFoundError(f"Could not find cuda.cu at {cuda_path}")
        
    with open(cuda_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith('#define N'):
            lines[i] = f'#define N {n}\n'
            break
    
    with open(cuda_path, 'w') as f:
        f.writelines(lines)

def compile_cuda():
    # Get directory containing cuda.cu
    cuda_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Change to that directory before compiling
    os.chdir(cuda_dir)
    
    cmd = 'nvcc cuda.cu -o cuda -lcublas'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Compilation failed: {result.stderr}\nCommand was: {cmd}\nWorking dir: {os.getcwd()}")

def run_program(times=1):
    cuda_global_times = []
    cuda_shared_times = []
    cublas_times = []
    
    program_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cuda'))
    
    if not os.path.exists(program_path):
        raise FileNotFoundError(f"Could not find cuda at {program_path}")
    
    for run in range(times):
        result = subprocess.run(program_path, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Program execution failed: " + result.stderr)
            
        output = result.stdout
        # Join lines to handle multi-line output
        output = ' '.join(output.split('\n'))
        
        # Updated patterns to match full output format; CPU logging removed
        time_patterns = {
            'global': r'Global CUDA:.*?Took\s+([\d.]+)\s+seconds',
            'shared': r'Shared CUDA:.*?Took\s+([\d.]+)\s+seconds',
            'cublas': r'cuBLAS CUDA:.*?Took\s+([\d.]+)\s+seconds'
        }
        
        # Extract times
        if match := re.search(time_patterns['global'], output):
            cuda_global_times.append(float(match.group(1)))
            print(f"Global time: {match.group(1)}")
            
        if match := re.search(time_patterns['shared'], output):
            cuda_shared_times.append(float(match.group(1)))
            print(f"Shared time: {match.group(1)}")
            
        if match := re.search(time_patterns['cublas'], output):
            cublas_times.append(float(match.group(1)))
            print(f"cuBLAS time: {match.group(1)}")
    
    return {
        'cuda_global': mean(cuda_global_times),
        'cuda_shared': mean(cuda_shared_times),
        'cublas': mean(cublas_times)
    }

def save_results(results, filename='performance_data2.csv'):
    # Note: CPU timing column removed
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Size', 'CUDA Global Time', 'CUDA Shared Time', 'cuBLAS Time'])
        for size, times in results.items():
            writer.writerow([size, times['cuda_global'], times['cuda_shared'], times['cublas']])

def main():
    matrix_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    results = {}
    
    for size in matrix_sizes:
        print(f"Testing matrix size: {size}x{size}")
        modify_n(size)
        compile_cuda()
        results[size] = run_program(times=1)
    
    save_results(results)

if __name__ == '__main__':
    main()
