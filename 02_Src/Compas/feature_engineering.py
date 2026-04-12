import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(df, column, drop_val):
    """
    Aplica codificación One-Hot a una variable categórica de un DataFrame.

    Esta función transforma una columna categórica en variables dummy binarias
    utilizando OneHotEncoder de sklearn. Se elimina una de las categorías
    (drop) para evitar multicolinealidad en modelos estadísticos.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada que contiene la variable a codificar.

    column : str
        Nombre de la columna categórica que se desea transformar.

    drop_val : str
        Categoría que se eliminará durante la codificación (referencia),
        con el fin de evitar la trampa de las variables dummy.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con las nuevas variables dummy generadas a partir de la
        columna original.

    Notas
    -----
    - Utiliza sklearn.preprocessing.OneHotEncoder.
    - El resultado no incluye la columna original.
    - El parámetro `sparse_output=False` asegura que la salida sea un array denso.
    - Es útil como paso previo al entrenamiento de modelos de machine learning.

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