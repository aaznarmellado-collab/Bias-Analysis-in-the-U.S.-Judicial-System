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


def remove_invisible_chars(text):
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


def classify_charge(x):
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