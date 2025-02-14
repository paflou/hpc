from mpi4py.futures import MPICommExecutor
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=10000, random_state=42, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


p = [{'mlp_layer1': [16, 32],
      'mlp_layer2': [16, 32],
      'mlp_layer3': [16, 32]}]

pg = ParameterGrid(p)


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

def main():
    start = time.time()
    with MPICommExecutor() as executor:
        if executor is None:
            return
        # Map the evaluation function over the parameter grid
        results = list(executor.map(evaluate, pg))
    
    end = time.time()
    
    for res in results:
        print(res)
    
    print(f"Total Execution Time: {end - start:.2f} seconds")

if __name__ == '__main__':
    main()
