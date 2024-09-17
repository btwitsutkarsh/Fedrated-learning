pip install numpy scikit-learn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simulate multiple clients by splitting the training data
num_clients = 5
client_data = np.array_split(X_train, num_clients)
client_labels = np.array_split(y_train, num_clients)

# Initialize global model
global_model = SGDClassifier(loss='log_loss', random_state=42)
# Perform a partial fit to initialize the model
global_model.partial_fit(X_train[:1], y_train[:1], classes=np.unique(y))

# Federated Learning
num_rounds = 10
for round in range(num_rounds):
    print(f"Round {round + 1}/{num_rounds}")
    
    # Train on each client's data
    client_models = []
    for i in range(num_clients):
        client_model = SGDClassifier(loss='log_loss', random_state=42)
        client_model.partial_fit(client_data[i], client_labels[i], classes=np.unique(y))
        client_models.append(client_model)
    
    # Aggregate client models (simple averaging of coefficients and intercepts)
    coeffs = []
    intercepts = []
    for model in client_models:
        coeffs.append(model.coef_)
        intercepts.append(model.intercept_)
    
    global_model.coef_ = np.mean(coeffs, axis=0)
    global_model.intercept_ = np.mean(intercepts, axis=0)

    # Evaluate global model
    accuracy = accuracy_score(y_test, global_model.predict(X_test))
    print(f"Global model accuracy: {accuracy:.4f}")

print("Federated Learning completed.")
