import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(df, column, drop_val):
    """
    Apply One-Hot encoding to a categorical variable in a DataFrame.

    This function transforms a categorical column into binary dummy variables
    using OneHotEncoder from sklearn. One of the categories is removed
    (drop) to avoid multicollinearity in statistical models.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame that contains the variable to be encoded.

    column : str
        Name of the categorical column to be transformed.

    drop_val : str
        Category that will be removed during encoding (reference)

    """
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