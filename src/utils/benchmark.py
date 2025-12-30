import pandas as pd
import numpy as np
import os

"""
Compares simulation server load vectors with benchmark data
using Euclidean distance and similarity percentage metrics.
Outputs a step-wise comparison CSV file.
"""

base_dir = os.path.dirname(os.path.abspath(__file__)) 
simulation_log_path = os.path.join(base_dir, "csv", "simulation_log.csv")
benchmark_path = os.path.join(base_dir, "csv", "mendeley_data.csv")
output_path = os.path.join(base_dir, "csv", "benchmark_comparison.csv")

my_data = pd.read_csv(simulation_log_path, encoding="utf-8")
benchmark = pd.read_csv(benchmark_path, encoding="utf-8")

my_servers = my_data[['Server1', 'Server2', 'Server3']]
benchmark_servers = benchmark[['X1', 'X2', 'X3']]
steps = my_data['Step']

distances = []
similarity_percentage = []
benchmark_closest = []

max_possible_distance = np.sqrt(3)

for i in range(len(my_servers)):
    sim_row = my_servers.iloc[i].values
    eucl_dist = np.sqrt(np.sum((benchmark_servers.values - sim_row) ** 2, axis=1))
    min_idx = np.argmin(eucl_dist)
    
    min_distance = eucl_dist[min_idx]
    distances.append(min_distance)
    
    similarity = (1 - min_distance / max_possible_distance) * 100
    similarity_percentage.append(similarity)
    
    benchmark_closest.append(benchmark_servers.iloc[min_idx].values)

benchmark_closest = np.array(benchmark_closest)

comparison = my_servers.copy()
comparison['Step'] = steps
comparison['Benchmark_X1'] = benchmark_closest[:,0]
comparison['Benchmark_X2'] = benchmark_closest[:,1]
comparison['Benchmark_X3'] = benchmark_closest[:,2]
comparison['Distance'] = distances
comparison['Similarity Percentage'] = similarity_percentage

comparison.to_csv(output_path, index=False)

print("Benchmark comparison completed! CSV was saved:", output_path)
print(comparison.head())
