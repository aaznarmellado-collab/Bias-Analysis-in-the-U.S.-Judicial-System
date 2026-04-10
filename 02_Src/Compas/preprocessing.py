import pandas as pd


from rapidfuzz import process, fuzz


import re


def convert_to_datetime(df):
    date_regex = r'^(?:(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}))(?:[\sT]+(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?)?$'

    for col in df.columns:
        if df[col].dtype.name == 'object':
            mask = df[col].dropna().str.match(date_regex)
            if mask.all():
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)

    return df


def normalize_df(df):
    df.columns = df.columns.str.lower()
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(
        lambda col: col.str.lower().str.strip()
    )
    return df


def check_duplicate_column(df, column1, column2):
    return df[df[column1]!=df[column2]]



def fuzzy_replace_safe(df, col, row, threshold=85):
    
    posibles = df[df["dob"] == row["dob"]]
    
    if posibles.empty:
        return row[col]
    
    result = process.extractOne(
        row[col],
        posibles[col].tolist(),
        scorer=fuzz.token_sort_ratio
    )
    
    if result is None:
        return row[col]
    
    match, score = result[0], result[1]
    
    return match if score >= threshold else row[col]


def clean_invisible_chars(text):
    if pd.isna(text):
        return text
    
    # reemplazar non-breaking space por espacio normal
    text = text.replace('\u00A0', ' ')
    
    # eliminar zero-width
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # quitar espacios extra
    text = " ".join(text.split())
    
    return text


def convert_to_datetime(df):
    date_regex = r'^(?:(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}))(?:[\sT]+(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?)?$'

    for col in df.columns:
        if df[col].dtype.name == 'object':
            mask = df[col].dropna().str.match(date_regex)
            if mask.all():
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)

    return df


def convert_categorical(df):
    age_order = ['less than 25', '25-45', '46-65', 'greater than 65']
    score_order = ['low', 'medium', 'high']
    charge_degree_order = ["misdemeanor", "felony"]


    df["age_cat"] = pd.Categorical(df["age_cat"], categories= age_order, ordered=True)
    df["score_text"] = pd.Categorical(df["score_text"], categories= score_order, ordered=True)
    df["c_charge_degree"] = pd.Categorical(df["c_charge_degree"], categories= charge_degree_order, ordered=True)
    

    return df