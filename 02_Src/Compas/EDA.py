
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

    plt.figure(figsize=(10, 8))
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
    demographic_var_list = ['race','age_cat', 'sex', 'maritalstatus']
    for variable in demographic_var_list:
        print(f"\nVariable: {variable}")
        for group in df[variable].unique():
            subset = df[df[variable] == group]
            
            cm = confusion_matrix(subset['is_recid'], subset['prediction'], labels=[0,1])
            cm_df = pd.DataFrame(cm, index=['Yes', 'No'], columns=['Yes', 'No'])

            cm_all = confusion_matrix(subset['is_recid'], subset['prediction'], normalize='all', labels=[0,1])
            cm_all_df = pd.DataFrame(cm_all, index=['Yes', 'No'], columns=['Yes', 'No'])

            cm_true = confusion_matrix(subset['is_recid'], subset['prediction'], normalize = 'true', labels=[0,1])
            cm_true_df = pd.DataFrame(cm_true, index=['Yes', 'No'], columns=['Yes', 'No'])

            print(f"\nGrupo: {group}")
            display(cm_df)
            print("\nNormalize All")
            display(cm_all_df)
            print("\nNormalize True")
            display(cm_true_df)
