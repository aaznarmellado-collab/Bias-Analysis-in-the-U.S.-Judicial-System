import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy.stats import chi2_contingency
from itertools import combinations
from scipy.stats import f_oneway


from IPython.display import display, Markdown



from sklearn.metrics import confusion_matrix


import warnings
warnings.filterwarnings('ignore')




# En bar plot (plotly), animation_frame para la barra interactuable
def eda(df):
    """
    Realiza un análisis exploratorio de datos (EDA) sobre un DataFrame de pandas.

    Esta función lleva a cabo un análisis univariable y bivariable del conjunto
    de datos, incluyendo visualizaciones y estadísticas descriptivas:

    - Variables categóricas:
        * Gráficos de pastel para variables con ≤ 3 valores únicos.
        * Gráficos de barras (Plotly) para variables con > 3 valores únicos.
    - Variables numéricas:
        * Estadísticos descriptivos (media, mediana, mínimo, máximo, varianza,
          desviación estándar y moda).
        * Diagramas de caja (boxplots) para analizar distribución y outliers.
    - Análisis bivariable:
        * Boxplots de variables numéricas agrupadas por variables categóricas y sexo.
        * Matriz de correlación (método Spearman) para variables numéricas.
        * Test de chi-cuadrado entre variables categóricas/booleanas, mostrando
          tablas de contingencia cuando son estadísticamente significativas (p ≤ 0.05).

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada que contiene variables numéricas, categóricas,
        booleanas y temporales. Se espera que incluya:
        - 'person_id' (excluida del análisis numérico)
        - 'is_recid', 'is_violent_recid' (variables booleanas)
        - 'sex' (usada como hue en algunos gráficos)

    Retorna
    -------
    None
        La función no devuelve ningún valor. Muestra directamente gráficos,
        tablas y estadísticas en pantalla.

    Notas
    -----
    - Diseñada para ejecutarse en entornos interactivos como Jupyter Notebook.
    - Utiliza matplotlib, seaborn y plotly para visualización.
    - El test de chi-cuadrado se realiza con scipy.stats.chi2_contingency.
    - La correlación se calcula mediante el método Spearman.

    """
    num_var_list = df.select_dtypes(include = "number").columns.tolist()
    num_var_list.remove("person_id")
    cat_var_list = df.select_dtypes(include = ["object", "category"]).columns.tolist()
    lista_variables_temporales = df.select_dtypes(include = "datetime").columns.tolist()
    bool_var_list = df[["is_recid", "is_violent_recid"]].columns.tolist()
    chi2_list = cat_var_list + bool_var_list
    
    display(Markdown("# Univariable analysis"))

    for col in cat_var_list:
    # Sacamos valores únicos y sus conteos
        counts = df[col].value_counts(dropna=False)
        
        if df[col].nunique() <= 3:
            # Preparamos labels con nombre + conteo
            labels = [f'{valor}\n{count}' for valor, count in zip(counts.index, counts.values)]
            
            # Dibujamos el pie chart
            plt.figure()
            plt.pie(
                counts.values,
                labels=labels,
                autopct='%1.1f%%'
            )
            
            plt.title(f'{col} distribution')
            plt.show()

        else:
            total = counts.sum()

            df_count = counts.reset_index()
            df_count.columns = [col, "frequency"]
            df_count["pct"] = df_count["frequency"] / total

            fig = px.bar(
                df_count,
                x=col,
                y='frequency',
                text= df_count.apply(
                lambda r: f"{r['pct']:.1%}",
                axis=1
            ),
            )

            fig.update_layout(
            title=f'Number of people per {col}',
            yaxis_title='Number of people'
            )

            fig.show()


    # df["ethnic_code_text"].value_counts().plot(kind="bar")
    # plt.title('Distribucion de personas por raza')
    # plt.show()

    
    print(f"Mean:\n{df[num_var_list].mean()}")
    print("\n")
    print(f"Median:\n{df[num_var_list].median()}")
    print("\n")
    print(f"Minimum:\n{df[num_var_list].min()}")
    print("\n")
    print(f"Maximum:\n{df[num_var_list].max()}")
    print("\n")
    print(f"Variance:\n{df[num_var_list].var()}")
    print("\n")
    print(f"Standard deviation:\n{df[num_var_list].std()}")
    print("\n")
    print(f"Mode:\n{df[num_var_list].mode()}")

    for col in num_var_list:
        df[[col]].boxplot(figsize=(8,5))
        plt.title(f"{col} distribution")
        plt.show()
        print("\n")

    display(Markdown("# Bivariable analysis"))
    print("\n")

    for col_num in num_var_list:
        for col_cat in cat_var_list:
            if col_cat != "sex":
                sns.boxplot(
                    data= df,
                    x=col_cat,
                    y= col_num,
                    hue="sex"
                )
                plt.title(f"{col_num} distribution by gender and {col_cat}")
                plt.show()
                print("\n")


    corr_matrix = df[num_var_list].corr(method= "spearman")

    plt.figure(figsize=(25, 18))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation matrix")
    plt.show()

    

    for col1, col2 in combinations(chi2_list, 2):
        crosstab = pd.crosstab(df[col1], df[col2], margins= True)
        chi2, p, dof, expected = chi2_contingency(crosstab)
        if(p <= 0.05):
            print(f"\nCrosstab between {col1} and {col2}")
            display(crosstab)



def check_bias(df):
    """
    Evalúa posibles sesgos en las predicciones de un modelo según grupos demográficos.

    Esta función analiza el rendimiento de un modelo de clasificación calculando
    matrices de confusión para distintos subgrupos demográficos, con el objetivo
    de detectar posibles desigualdades o sesgos en las predicciones.

    Para cada variable demográfica y sus grupos, se calculan:
    - Matriz de confusión absoluta
    - Matriz de confusión normalizada respecto al total ('all')
    - Matriz de confusión normalizada respecto a las clases reales ('true')

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada que debe contener al menos las siguientes columnas:
        - 'is_recid' : variable objetivo real (binaria)
        - 'prediction' : predicciones del modelo (binarias)
        - 'race', 'age_cat', 'sex', 'maritalstatus' : variables demográficas

    Retorna
    -------
    None
        La función no devuelve ningún valor. Imprime y muestra matrices de
        confusión para cada subgrupo.

    Notas
    -----
    - Utiliza sklearn.metrics.confusion_matrix.
    - Tipos de normalización:
        * 'all': proporción respecto al total de observaciones
        * 'true': proporción respecto a la clase real
    - Útil para análisis de equidad (fairness) en modelos de clasificación.

    """
    df = df.copy()

    demographic_var_list = ['race','age_cat', 'sex', 'maritalstatus']
    for variable in demographic_var_list:
        print(f"\nVariable: {variable}")
        for group in df[variable].unique():
            
            subset = df[df[variable] == group]

            if len(subset) == 0:
                continue
            
            cm = confusion_matrix(subset['is_recid'], subset['prediction'], labels=[0,1])
            cm_df = pd.DataFrame(cm, index=['No', 'Yes'], columns=['No', 'Yes'])

            cm_all = confusion_matrix(subset['is_recid'], subset['prediction'], normalize='all', labels=[0,1])
            cm_all_df = pd.DataFrame(cm_all, index=['No', 'Yes'], columns=['No', 'Yes'])

            cm_true = confusion_matrix(subset['is_recid'], subset['prediction'], normalize = 'true', labels=[0,1])
            cm_true_df = pd.DataFrame(cm_true, index=['No', 'Yes'], columns=['No', 'Yes'])

            print(f"\nGroup: {group}")
            display(cm_df)
            print("\nNormalize All")
            display(cm_all_df)
            print("\nNormalize True")
            display(cm_true_df)
