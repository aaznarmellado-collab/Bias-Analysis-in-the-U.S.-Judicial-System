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


# In a bar plot (Plotly), animation_frame for an interactive bar
def eda(df):
    """
    Perform an exploratory data analysis (EDA) on a pandas DataFrame.

    This function carries out a univariate and bivariate analysis of the dataset, including visualizations and descriptive statistics:

    - Categorical variables:
      * Pie charts for variables with ≤ 3 unique values.
      * Bar charts (Plotly) for variables with > 3 unique values.
    - Numerical variables:
      * Descriptive statistics (mean, median, minimum, maximum, variance, standard deviation, and mode).
      * Boxplots to analyze distribution and outliers.
    - Bivariate analysis:
      * Boxplots of numerical variables grouped by categorical variables and sex.
      * Correlation matrix (Spearman method) for numerical variables.
      * Chi-square test between categorical/boolean variables, showing contingency tables when statistically significant (p ≤ 0.05).

    Parameters
    ----------
    df : pandas.DataFrame
      Input DataFrame containing numerical, categorical, boolean, and temporal variables. It is expected to include:
      - 'person_id' (excluded from numerical analysis)
      - 'is_recid', 'is_violent_recid' (boolean variables)

    """
    num_var_list = df.select_dtypes(include = "number").columns.tolist()
    num_var_list.remove("person_id")
    cat_var_list = df.select_dtypes(include = ["object", "category"]).columns.tolist()
    lista_variables_temporales = df.select_dtypes(include = "datetime").columns.tolist()
    bool_var_list = df[["is_recid", "is_violent_recid"]].columns.tolist()
    chi2_list = cat_var_list + bool_var_list
    
    display(Markdown("# Univariable analysis"))

    for col in cat_var_list:
    # We extract unique values and their counts
        counts = df[col].value_counts(dropna=False)
        
        if df[col].nunique() <= 3:
            # We prepare labels with name + count
            labels = [f'{valor}\n{count}' for valor, count in zip(counts.index, counts.values)]
            
            # We draw the pie chart
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
    # plt.title('Distribution of people by race')
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
    Evaluate potential biases in a model’s predictions across demographic groups.

    This function analyzes the performance of a classification model by computing confusion matrices for different demographic subgroups, 
    with the goal of detecting possible inequalities or biases in the predictions.

    For each demographic variable and its groups, the following are calculated:
      - Absolute confusion matrix  
      - Confusion matrix normalized with respect to the total ('all')  
      - Confusion matrix normalized with respect to the true classes ('true')  

    Parameters
    ----------
    df : pandas.DataFrame  
      Input Data

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
