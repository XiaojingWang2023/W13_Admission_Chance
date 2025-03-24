from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import MinMaxScaler


# Function to train the model
def train_NNmodel(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Neural Network model
    MLP = MLPClassifier(hidden_layer_sizes=(3,3), batch_size=50, max_iter=200, random_state=123)
    
    # Train the model
    MLP.fit(X_train_scaled, y_train)
    
    # Save the trained model
    with open('models/NNmodel.pkl', 'wb') as f:
        pickle.dump(MLP, f)

    return MLP, X_test_scaled, y_test
