
import pandas as pd

def load_data(data_path):
    
    # Import the data from 'Admission.csv'
    df = pd.read_csv(data_path)


    # Drop 'Serial_No' variable from the data
    df = df.drop('Serial_No', axis=1)

    return df
