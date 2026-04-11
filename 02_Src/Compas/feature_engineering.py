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


def compute_weight(df, row):
    P_A = df["race"].value_counts(normalize=True)
    P_Y = df["two_year_recid"].value_counts(normalize=True)
    P_AY = df.groupby(["race","two_year_recid"]).size() / len(df)
    return (P_A[row["race"]] * P_Y[row["two_year_recid"]]) / P_AY[row["race"], row["two_year_recid"]]