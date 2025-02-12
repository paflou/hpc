#!/usr/bin/env python3
import subprocess
import csv
from statistics import mean
import os
import re

def modify_n(n):
    """
    Modify the openmpGPU.c source file by updating the #define N line.
    """
    # Get absolute path to openmpGPU.c (adjust path as needed)
    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'openmpGPU.c'))
    
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Could not find openmpGPU.c at {source_path}")
        
    with open(source_path, 'r') as f:
        lines = f.readlines()
    
    found = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith('#define N'):
            lines[i] = f'#define N {n}\n'
            found = True
            break
    
    if not found:
        raise Exception("Could not find a '#define N' line in the source file.")
    
    with open(source_path, 'w') as f:
        f.writelines(lines)
    print(f"Modified openmpGPU.c: Set N = {n}")

def compile_openmpGPU():
    """
    Compile the openmpGPU.c file using GCC with OpenMP support.
    """
    # Get directory containing openmpGPU.c (assumed one directory above)
    source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Change to that directory before compiling
    os.chdir(source_dir)
    
    # Compile command: adjust flags as needed (e.g., -O3 optimization)
    cmd = 'gcc openmpGPU.c -o openmpGPU -fopenmp -O3 -lm -fno-lto'
    print("Compiling with command:", cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Compilation failed:\n{result.stderr}\nCommand was: {cmd}\nWorking dir: {os.getcwd()}")
    else:
        print("Compilation succeeded.")

def run_program(times=1):
    """
    Run the compiled openmpGPU executable a number of times,
    parse its output, and return the averaged execution time.
    Expected output format from openmpGPU:
    
      openMP GPU: E[N*N-1]= ... | F[N*N-1] = ...
       Took <time_val> seconds
       
    This function only extracts the time.
    """
    openmp_gpu_times = []
    
    # Get full path to the executable (assumed to be in the same directory as the source)
    exe_path = os.path.abspath(os.path.join(os.getcwd(), 'openmpGPU'))
    
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"Could not find executable at {exe_path}")
    
    for run in range(times):
        result = subprocess.run(exe_path, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Program execution failed: " + result.stderr)
            
        output = result.stdout
        # Combine output into one string (in case of multiple lines)
        output = ' '.join(output.splitlines())
        print(f"Run {run+1} output: {output}")
        
        # Extract the time using a regex pattern
        time_pattern = r"Took\s+([-\d\.e]+)\s+seconds"
        match = re.search(time_pattern, output, re.IGNORECASE)
        if match:
            time_val = float(match.group(1))
            openmp_gpu_times.append(time_val)
            print(f"Parsed time: {time_val} seconds")
        else:
            raise Exception("Could not parse execution time from output:\n" + output)
    
    return {
        'openmp_gpu_time': mean(openmp_gpu_times)
    }

def save_results(results, filename='openmpGPU_performance_data.csv'):
    """
    Save the performance data into a CSV file.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Size', 'OpenMP GPU Time (sec)'])
        for size, times in results.items():
            writer.writerow([size, times['openmp_gpu_time']])
    print(f"Data successfully written to '{filename}'.")

def main():
    # List of matrix sizes to test. Adjust these as needed.
    matrix_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    results = {}
    
    for size in matrix_sizes:
        print(f"\nTesting matrix size: {size}x{size}")
        modify_n(size)
        compile_openmpGPU()
        results[size] = run_program(times=1)
    
    save_results(results)

if __name__ == '__main__':
    main()
