import pandas as pd


def create_dummy_vars(df):
    
    # convert target variable 'Admit_Chance' to a binary class
    df['Admit_Chance'] = (df['Admit_Chance'] >= 0.8).astype(int)
    
    # create dummy categorical variables 'University_Rating' and 'Research'
    df = pd.get_dummies(df, columns=['University_Rating', 'Research'], dtype='int')
    
    # store the processed dataset in data/processed
    df.to_csv('data/processed/Processed_Admission.csv', index=None)

    # Separate the features and target variable 
    X = df.drop('Admit_Chance', axis=1)
    y = df['Admit_Chance']

    return X, y