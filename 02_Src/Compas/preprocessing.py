import pandas as pd

from rapidfuzz import process, fuzz

import re

def convert_to_datetime(df):
    """
    Automatically converts columns formatted as dates to the datetime type.

    The function identifies object-type columns containing values formatted
    as dates using a regular expression and converts them to the
    pandas datetime type.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the date columns converted.

    Notes
    -----
    - Detects multiple date formats (YYYY-MM-DD, DD-MM-YYYY, etc.).
    - Uses automatic format inference.
    - Non-convertible values are transformed to NaT.

    """
    date_regex = r'^(?:(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}))(?:[\sT]+(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?)?$'

    for col in df.columns:
        if df[col].dtype.name == 'object':
            mask = df[col].dropna().str.match(date_regex)
            if mask.all():
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)

    return df


def normalize_df(df):
    """
    Normalizes column names and text values in a DataFrame.

    Converts:
    - Column names to lowercase.
    - String column values to lowercase and removes extra spaces.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Normalized DataFrame.

    """
    df.columns = df.columns.str.lower()
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(
        lambda col: col.str.lower().str.strip()
    )
    return df


def check_duplicate_column(df, column1, column2):
    """
    Compares two columns and returns the rows where their values differ.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    column1 : str
        Name of the first column.

    column2 : str
        Name of the second column.

    Returns
    -------
    pandas.DataFrame
        A subset of the DataFrame where the values in both columns do not match.

    """
    return df[df[column1]!=df[column2]]



def fuzzy_replace_safe(df, col, row, threshold=85):
    """
    Performs fuzzy matching on values in a column.

    Searches for similar matches within a filtered subset of data
    (by date of birth ‘dob’) and replaces the value if the similarity
    exceeds a defined threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Reference DataFrame.

    col : str
        Name of the column to evaluate.

    row : pandas.Series
        Current row to process.

    threshold : int, optional (default=85)
        Minimum similarity threshold to accept the replacement.

    Returns
    -------
    str
        Original value or replaced value if a valid match is found.

    Notes
    -----
    - Uses rapidfuzz to measure similarity between strings.
    - Reduces errors caused by text inconsistencies.

    """
    
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


def remove_invisible_chars(text):
    """
    Removes invisible characters and unnecessary spaces from text.

    Performs:
    - Replacement of non-breaking spaces.
    - Removal of invisible characters (zero-width characters).
    - Normalization of spaces.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Cleaned text.

    """
    if pd.isna(text):
        return text
    
    # reemplazar non-breaking space por espacio normal
    text = text.replace('\u00A0', ' ')
    
    # eliminar zero-width
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # quitar espacios extra
    text = " ".join(text.split())
    
    return text


def convert_categorical(df):
    """
    Converts categorical variables into ordered categorical variables.

    Defines a specific order for certain variables and transforms them into
    an ordinal categorical type.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ordered categorical variables.

    Notes
    -----
    - Transformed variables:
        * age_cat
        * score_text
        * c_charge_degree

    """
    age_order = ['less than 25', '25-45', '46-65', 'greater than 65']
    score_order = ['low', 'medium', 'high']
    charge_degree_order = ["misdemeanor", "felony"]


    df["age_cat"] = pd.Categorical(df["age_cat"], categories= age_order, ordered=True)
    df["score_text"] = pd.Categorical(df["score_text"], categories= score_order, ordered=True)
    df["c_charge_degree"] = pd.Categorical(df["c_charge_degree"], categories= charge_degree_order, ordered=True)
    

    return df


def classify_charge(x):
    """
    Classifies a type of crime into multiple binary categories.

    Analyzes the text of a crime description and generates binary variables
    indicating the presence of different types of crime.

    Categories:
    - Violent
    - Drugs
    - Property
    - Fraud
    - Weapons
    - Trafficking
    - Sexual
    - Public order
    - Justice-related
    - No charges

    Parameters
    ----------
    x : str
        Descriptive text of the crime.

    Returns
    -------
    dict
        Dictionary with binary variables (0/1) for each category.

    Notes
    -----
    - Based on keyword search.
    - Enables feature engineering for predictive models.

    """
    x = x.lower()
    
    return {
        # 🔴 VIOLENT
        'is_violent': int(any(k in x for k in [
            'assault','battery','robbery','murder','manslaughter',
            'kidnapping','stalking','abuse','cruelty','false imprisonment',
            'culpable negligence','child','elderly','throw','shoot',
            # 🔥 NUEVAS
            'bodily injury'
        ])),
        
        # 🟠 DRUGS
        'is_drug': int(any(k in x for k in [
            'cocaine','cannabis','heroin','drug','controlled substance',
            'traffick','deliver','paraphernalia','meth','amphetamine',
            'alprazolam','oxycodone','hydrocodone','mdma','lsd','fentanyl',
            'morphine','codeine','clonazepam','diazepam','steroid',
            'phentermine','carisoprodol','buprenorphine',
            'tetrahydrocannabinols','benzylpiperazine','pyrrolidinovalerophenone',
            'lorazepam','hydromorphone','rx',
            # 🔥 NUEVAS
            'contr subst','control substa', 'substance', 'w/intent'
        ])),
        
        # 🟡 PROPERTY
        'is_property': int(any(k in x for k in [
            'theft','burglary','burgl','stolen','shoplifting',
            'mischief','arson','damage','property'
        ])),
        
        # 🔵 FRAUD
        'is_fraud': int(any(k in x for k in [
            'fraud','credit','identity','counterfeit','forg','launder',
            'worthless','false','impersonat','insurance','check',
            'bribery','simulation','structuring',
            'id info','intellectual','counterfeit',
            # 🔥 NUEVAS
            'invalid insur','lictag','sticker','counterfeit cont'
        ])),
        
        # 🟣 WEAPONS
        'is_weapon': int(any(k in x for k in [
            'weapon','firearm','arm', 'wep'
        ])),
        
        # ⚫ TRAFFIC
        'is_traffic': int(any(k in x for k in [
            'dui','driving','license','vehicle','traffic','dl','acc',
            'motorcycle','fuel',
            'susp','revoked','cancel',
            # 🔥 NUEVAS
            'breath test','railroad','rr', 'drivers'
        ])),
        
        # ⚪ SEXUAL
        'is_sexual': int(any(k in x for k in [
            'sexual','lewd','molest','porn','voyeur',
            'sex batt', 'prostitute'
        ])),
        
        # 🟤 PUBLIC DISORDER
        'is_public_disorder': int(any(k in x for k in [
            'disorderly','trespass','prostitution','intoxication',
            'loitering','littering','panhandle','alcohol',
            'disturb','beg','school',
            'tresspass','ride','beverage','public',
            # 🔥 NUEVAS
            'intoxicated','animal','baiting', 'busn'
        ])),
        
        # ⚙️ JUSTICE / SYSTEM
        'is_justice_related': int(any(k in x for k in [
            'resist','injunct','fail','tamper','escape','violation',
            'flee','elude','obstruct','harass','witness','evidence',
            'solicit','conspiracy','attempt','contraband',
            'custody','extradition','court','leo',
            'restraining','dna','911','misuse','statement',
            'device','badge','compliance','pretrial',
            # 🔥 NUEVAS
            'delinquency','unauth','issue','phone','fac fel', 'prot'
        ])),
        
        # 🧾 NO CHARGE
        'no_charge': int('arrest case no charge' in x)
    }