from mpi4py import MPI # type: ignore
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.model_selection import ParameterGrid # type: ignore
from sklearn.datasets import make_classification # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import time

X, y = make_classification(n_samples=10000, random_state=42, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

p = [{'mlp_layer1': [16, 32],
      'mlp_layer2': [16, 32],
      'mlp_layer3': [16, 32]}]

pg = list(ParameterGrid(p))

def evaluate(p):
    print(p)
    l1 = p['mlp_layer1']
    l2 = p['mlp_layer2']
    l3 = p['mlp_layer3']
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_pred, y_test)
    print(ac)
    return p, ac

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:  # Master process
    start_time = time.time()  # Start timing

    task_queue = pg[:]
    num_workers = size - 1
    results = []

    # Send initial tasks to workers
    for worker in range(1, num_workers + 1):
        if task_queue:
            task = task_queue.pop(0)
            comm.send(task, dest=worker)

    # Receive results and assign new tasks
    while len(results) < len(pg):
        result = comm.recv(source=MPI.ANY_SOURCE)
        results.append(result)

        if task_queue:
            task = task_queue.pop(0)
            comm.send(task, dest=result['worker'])
        else:
            comm.send(None, dest=result['worker'])  # Signal worker to stop

    # Stop timing
    end_time = time.time()

    # Display results and execution time
    print("Results:", results)
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")

else:  # Worker processes
    while True:
        task = comm.recv(source=0)
        if task is None:  # Termination signal
            break
        result = task_evaluation(task)
        comm.send({'worker': rank, 'result': result}, dest=0)
