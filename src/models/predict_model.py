# Import accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score

# # Function to predict and evaluate
def evaluate_model(model, X_test_scaled, y_test):
    # make predictions on train
    y_pred = model.predict(X_test_scaled)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    # return accuracy, confusion_mat

    return accuracy, confusion_mat