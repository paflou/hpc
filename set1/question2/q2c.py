from mpi4py import MPI
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

X, y = make_classification(n_samples=90000, random_state=42, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

p = [{'mlp_layer1': [16, 32],
      'mlp_layer2': [16, 32],
      'mlp_layer3': [16, 32]}]

pg = list(ParameterGrid(p))

def evaluate(p):
    #print(p)
    l1 = p['mlp_layer1']
    l2 = p['mlp_layer2']
    l3 = p['mlp_layer3']
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_pred, y_test)
    #print(ac)
    return p, ac

def master():
    num_workers = size - 1 

    start_time = time.time()

    for i, p in enumerate(pg):
        worker = (i % num_workers) + 1  # Round-robin assignment
        comm.send(p, dest=worker)
        print(f"Master: Sent task {p} to worker {worker}.")

    for worker in range(1, size):
        comm.send(None, dest=worker)

    results = []
    for _ in enumerate(pg):
        result = comm.recv(source=MPI.ANY_SOURCE)
        results.append(result)
        print(f"Master: Received result {result}.")

    end_time = time.time()
    for r in results:
        print(r)

    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")


def worker():
    while True:
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
        if task is None:  # Stop signal
            print(f"Worker {rank}: Received stop signal. Exiting.")
            break
        print(f"Worker {rank}: Received task {task}.")
        result = evaluate(task)
        comm.send(result, dest=0)
        print(f"Worker {rank}: Sent result {result}.")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if __name__ == "__main__":
    if rank == 0:
        master()
    else:
        worker()
