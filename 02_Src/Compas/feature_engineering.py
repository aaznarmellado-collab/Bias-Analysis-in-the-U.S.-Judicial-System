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