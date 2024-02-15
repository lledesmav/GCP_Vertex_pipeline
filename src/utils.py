from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Preprocessing function
def preprocess_data(data):
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Converting to DataFrames
    train_df = pd.DataFrame(X_train, columns=X.columns)
    train_df['target'] = y_train.values
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df['target'] = y_test.values

    return train_df, test_df, scaler

# Model training function
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Model evaluation function
def evaluate_model(model, test_df):
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
def get_metrics(y_true, y_hat):
    log = {}
    
    # For multiclass classification, you can use 'macro', 'micro', or 'weighted'
    average_method = 'macro'

    log['precision'] = precision_score(y_true, y_hat, average=average_method)
    log['recall'] = recall_score(y_true, y_hat, average=average_method)
    log['f1'] = f1_score(y_true, y_hat, average=average_method)

    matrix = confusion_matrix(y_true, y_hat)
    return log, matrix
