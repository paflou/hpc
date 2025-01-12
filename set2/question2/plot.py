import subprocess
import csv
from statistics import mean

def modify_n(n):
    with open('cuda.cu', 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith('#define N'):
            lines[i] = f'#define N {n}\n'
            break
    
    with open('cuda.cu', 'w') as f:
        f.writelines(lines)

def compile_cuda():
    result = subprocess.run('nvcc -Xcompiler -fopenmp cuda.cu -o cuda_program -lcublas', 
                          shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("Compilation failed: " + result.stderr)

def run_program(times=3):
    cuda_times = []
    cpu_times = []
    
    for _ in range(times):
        result = subprocess.run('./cuda_program', capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'CUDA (CUBLAS)' in line:
                time = float(line.split('Took')[-1].replace('seconds', '').strip())
                cuda_times.append(time)
            elif 'CPU: First' in line:
                time = float(line.split('Took')[-1].replace('seconds', '').strip())
                cpu_times.append(time)
    
    return mean(cuda_times), mean(cpu_times)

def main():
    n_values = [32, 64, 128, 256, 512, 1024,2048,4096]
    results = []
    
    print("Starting benchmarks...")
    for n in n_values:
        print(f"Testing N={n}")
        modify_n(n)
        compile_cuda()
        cuda_time, cpu_time = run_program()
        results.append([n, cuda_time, cpu_time, cpu_time/cuda_time])
        print(f"N={n} complete: CUDA={cuda_time:.4f}s, CPU={cpu_time:.4f}s")
    
    with open('benchmark_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'CUDA_time', 'CPU_time', 'Speedup'])
        writer.writerows(results)
    
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    main()