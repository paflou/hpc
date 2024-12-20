from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.model_selection import ParameterGrid # type: ignore
from sklearn.datasets import make_classification # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from multiprocessing import Pool # type: ignore
import time

# Data preparation
X, y = make_classification(n_samples=10000, random_state=42, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define parameter grid
params = [{'mlp_layer1': [16, 32], 'mlp_layer2': [16, 32], 'mlp_layer3': [16, 32]}]
pg = list(ParameterGrid(params))

# Function to evaluate a parameter combination
def evaluate(params):
    l1, l2, l3 = params['mlp_layer1'], params['mlp_layer2'], params['mlp_layer3']
    model = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    return params, accuracy

# Parallel processing
if __name__ == "__main__":
    start = time.time()
    with Pool(8) as pool:
        results = pool.map(evaluate, pg)
    end = time.time()
    
    # Display results
    for params, accuracy in results:
        print(f"Params: {params}, Accuracy: {accuracy}")
    print(f"Execution Time: {end - start} seconds")
