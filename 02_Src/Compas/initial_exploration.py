import pandas as pd

import matplotlib.pyplot as plt

import missingno as msno

import os
from IPython.display import display




def read_file(file_path):
    try:
        file, extension = os.path.splitext(file_path.lower())

        if extension == '.csv':
            df = pd.read_csv(file_path)
        elif extension == '.parquet':
            df = pd.read_parquet(file_path)
        elif extension == '.xlsx' or extension == 'xls':
            df =pd.read_excel(file_path)
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    

def initial_exploration(df):
    print('Rows and columns:')
    display(df.shape)

    print('First five rows:')
    display(df.head(5))
    print("\n")

    print('Last five rows:')
    display(df.tail(5))
    print("\n")

    print('Random 5 sample:')
    display(df.sample(5))
    print("\n")

    print('Dataset columns:')
    display(df.columns)

    print('Column data types:')
    display(df.dtypes)

    print('Number of columns per data type:')
    display(df.dtypes.value_counts())

    print('Detailed info:\n')
    display(df.describe())

    print('Unique values count:')
    display(df.nunique())

    print('Unique values:\n')
    df_unique_values = pd.DataFrame(df.apply(lambda x: x.unique()))
    display(df_unique_values)

    print('Duplicates per column:')
    df_duplicated = pd.DataFrame(df.apply(lambda x: x.duplicated()).sum().reset_index().rename(columns = {'index': 'Col', 0: 'Duplicates'}))
    df_duplicated = df_duplicated.sort_values(by = 'Duplicates', ascending = False).reset_index(drop=True)
    display(df_duplicated)

    print('Null values per column:')
    display(df.isnull().sum().reset_index().rename(columns = {'index': 'Col', 0: 'Null values'}))
    
    print('Null % per column:')
    display(df.isnull().mean().mul(100).round(2).sort_values(ascending = False).reset_index().rename(columns = {'index': 'Col', 0: 'Pct'}))

    print("## Null values: Visualization")
    msno.bar(df, figsize = (6, 3), fontsize= 9)
    plt.show()
    print('-' * 100)

    print("## Null values: Pattern visualization")
    msno.matrix(df, figsize = (6, 3), fontsize= 9, sparkline = False)
    plt.show()
    print('-' * 100)
