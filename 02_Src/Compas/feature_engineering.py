import pandas as pd

from sklearn.preprocessing import OneHotEncoder



def one_hot_encoding(df, column, drop_val):
    encoder = OneHotEncoder(
    drop=[drop_val], 
    sparse_output=False
    )

    encoded = encoder.fit_transform(df[[column]])

    df_encoded = pd.DataFrame(
    encoded,
    columns = encoder.get_feature_names_out([column])
    )

    return df_encoded