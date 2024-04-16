# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:54:10 2024

@author: jt_ur
"""

# SETUP

# Importer les libreries 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.stats import chi2_contingency
import scipy.stats as stats
import fanalysis.mca as MCA
from scipy.stats import zscore
from sklearn.impute import KNNImputer
import textwrap
from scipy.stats import kendalltau

# Créer les dataframes
def create_dataframe(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_index(axis=1, ascending=False)
    return df

# Utilisation :
df = create_dataframe(r"C:\Users\jt_ur\DA\projects\OpenAlex\Publications universitaires\works-UPEC.csv")
df2 = create_dataframe(r"C:\Users\jt_ur\DA\projects\OpenAlex\Publications universitaires\works-UPC.csv")

# Traitement des doublons
def check_duplicates(df):
    duplicates = df.duplicated()
    print(f"Nombre de lignes dupliquées : {duplicates.sum()}")

# Utilisation :
check_duplicates(df)
check_duplicates(df2)

# Traitement des Nan
def handle_nan_and_drop(df):
    nan_values_percent = (df.isna().sum() / len(df)) * 100
    cols_to_drop = nan_values_percent[nan_values_percent > 50].index
    df = df.drop(columns=cols_to_drop)
    return df

# Utilisation :
df = handle_nan_and_drop(df)
df2 = handle_nan_and_drop(df2)

# Définition des institutions
def add_institution_column(df, institution_name):
    df["Nom de l'établissement"] = institution_name
    return df

# Utilisation :
df = add_institution_column(df, "Université Paris Est Créteil")
df2 = add_institution_column(df2, "Université Paris Cité")

# Définition des variables pertinentes
def create_relevant_dataframe(df):
    relevant_columns = ["Nom de l'établissement", "id", "title", "publication_year", "language", "type", "is_oa",
                        "cited_by_count", "referenced_works_count", "cited_by_percentile_year_min",
                        "cited_by_percentile_year_max", "biblio_first_page", "biblio_last_page",
                        'sustainable_development_goals_score', 'sustainable_development_goals_display_name',
                        "primary_location_display_name", "locations_count", "primary_location_source_type",
                        "primary_location_source_host_organization_name", "author_ids", "author_names",
                        "author_orcids", "author_institution_ids", "author_institution_names",
                        "institutions_distinct_count", "primary_topic_display_name",
                        "primary_topic_subfield_display_name", "primary_topic_field_display_name",
                        "primary_topic_domain_display_name", "keywords_keyword", "countries_distinct_count"]
    return df[relevant_columns]

# Utilisation :
data = create_relevant_dataframe(df)
data2= create_relevant_dataframe(df2)

# Contrôle des Nan
def handle_nan(data):
    nan_values_percent = (data.isna().sum() / len(data)) * 100
    cols_to_drop = nan_values_percent[nan_values_percent > 50].index
    data = data.drop(columns=cols_to_drop)
    return data

# Utilisation :
data = handle_nan(data)
data2 = handle_nan(data2)

#Imputation NNK pour les variables numériques présentant entre 10% et 40% des Nan

num_var=["cited_by_count","referenced_works_count", "cited_by_percentile_year_min", "cited_by_percentile_year_max", 
         "biblio_first_page", "biblio_last_page", 'sustainable_development_goals_score',"institutions_distinct_count",
         "countries_distinct_count"]

def impute_knn(data, num_var):
    cols_to_impute = [col for col in num_var if (data[col].isnull().mean() > 0.01) & (data[col].isnull().mean() < 0.49)]
    
    if len(cols_to_impute) == 0:
        print("Aucune colonne sélectionnée pour l'imputation.")
    else:
        print(f"{len(cols_to_impute)} colonnes sélectionnées pour l'imputation : {cols_to_impute}")
    
    # Conversion des colonnes au format numérique
    for col in cols_to_impute:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Imputation avec KNN
    imputer = KNNImputer(n_neighbors=5)
    data[cols_to_impute] = imputer.fit_transform(data[cols_to_impute])
    
    return data

# Utilisation :
data = impute_knn(data, num_var)
data2 = impute_knn(data2, num_var)

#Imputation Mode pour les variables catégorielles présentant -10% de Nan et création d'une modalité "Non spécifié" si entre 10% et 49%

cat_var= ["publication_year",'primary_location_source_type',"type",
          "language", "is_oa",
          "author_names",
          'primary_location_display_name',"primary_location_source_host_organization_name",
          "primary_topic_domain_display_name","primary_topic_field_display_name",
          "primary_topic_subfield_display_name", "primary_topic_display_name", 
          "sustainable_development_goals_display_name",]

def impute_categorical(data, cat_var):
    for col in cat_var:
        data[col] = data[col].astype('category')
        missing_percentage = data[col].isnull().mean()
        if 0.1 <= missing_percentage < 0.4:
            # Créez une nouvelle modalité "Non spécifié" pour les valeurs manquantes
            data[col] = data[col].cat.add_categories("Non spécifié")
            data[col] = data[col].fillna("Non spécifié")
        elif missing_percentage < 0.1:
            # Remplacez les valeurs manquantes par le mode
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)
    
    print(f"Colonnes catégorielles imputées : {', '.join(cat_var)}")
    return data

# Utilisation :
data = impute_categorical(data, cat_var)
data2 = impute_categorical(data2, cat_var)

# Création de la variable 'number_of_pages'
def create_number_of_pages_column(df):
    df['biblio_first_page'] = pd.to_numeric(df['biblio_first_page'], errors='coerce')
    df['biblio_last_page'] = pd.to_numeric(df['biblio_last_page'], errors='coerce')
    df['number_of_pages'] = df['biblio_last_page'] - df['biblio_first_page'] + 1
    return df

# Utilisation :
data = create_number_of_pages_column(data)
data2 = create_number_of_pages_column(data2)

# Renommer les modalités de la variable 'is_oa'
def rename_is_oa(data):
    data["is_oa"] = data["is_oa"].replace({True: 'Open Access', False: 'Non Open Access'})
    return data

# Utilisation :
data = rename_is_oa(data)
data2 = rename_is_oa(data2)

# Renommer les variables

def rename_columns(data):
    labels = {
        "cited_by_count": "Citations",
        "publication_year": "Année",
        'primary_location_source_type': "Type de source",
        "type": "Type de publication",
        "language": "Langue",
        "is_oa": "Type d'accès",
        "primary_location_display_name": "Source",
        "primary_location_source_host_organization_name": "Editeur",
        "author_names": "Auteur",
        "primary_topic_domain_display_name": "Domaine disciplinaire",
        "primary_topic_field_display_name": "Champ disciplinaire",
        "primary_topic_subfield_display_name": "Sous-champ disciplinaire",
        "primary_topic_display_name": "Sujet disciplinaire",
        "sustainable_development_goals_display_name": "Objectif développement durable",
        "referenced_works_count": "Références effectuées",
        "number_of_pages": "Pages",
        "institutions_distinct_count": "Institutions associées",
        "countries_distinct_count": "Pays associés",
        "sustainable_development_goals_score": "Score objectif développement durable"
    }
    return data.rename(columns=labels)

# Utilisation :
data = rename_columns(data)
data2 = rename_columns(data2)

# Transformer la variable "Année" en chaîne de caractère
data["Année"] = data["Année"].astype(str)
data2["Année"] = data2["Année"].astype(str)

# Définir la palette 
colors = ["#041A59", "#278C8C", "#77D9A1", "#7B518C", "#7D7D7D", "#F24B4B", "#F2BA52", "#D0BAD9", "#D96B52", "#CCEEF1", "#EBAB97", "#CCB2D3"]
sns.set_palette(colors)

# ANALYSES UNIVARIEES

# Variables catégorielles

cat_var= ["Année","Type de source","Type de publication",
          "Langue","Type d'accès","Source","Editeur","Auteur",
          "Domaine disciplinaire","Champ disciplinaire","Sous-champ disciplinaire",
          "Sujet disciplinaire","Objectif développement durable"]

# Option nombre de publications
def plot_publications_by_category(dfs, cat_var):
    for var in cat_var:
        fig, axs = plt.subplots(len(dfs), 1, figsize=(11, 5.5*len(dfs)))  # Elargir le cadre des graphiques
        fig.suptitle(f'Nombre de publications par {var}', fontsize=16, weight='bold')
        for i, (df_name, df) in enumerate(dfs.items()):
            ax = axs[i]  # Utiliser un objet ax pour plus de clarté
            institution_name = df["Nom de l'établissement"].unique()[0]  # Assumer qu'il y a un seul nom d'établissement par dataframe
            if var == "Année":
                order = sorted(df[var].unique())
                title = f'{institution_name}'
                ax = sns.countplot(data=df, x=var, order=order, ax=ax)
                max_value = max(df[var].value_counts())
                ax.set_ylim(0, max_value + 0.1 * max_value)
                for p in ax.patches:
                    ax.annotate(format(p.get_height(), '.0f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center',
                                xytext=(0, 10),
                                textcoords='offset points')
                ax.set(xlabel=None, ylabel=None)  # Enlever le label de l'axe x et y
            else:
                if var == "Auteur":  # Vérifier si la variable est "Auteur"
                    df[var] = df[var].apply(lambda x: x if len(x) <= 20 else ' '.join(x.split()[:5]) + " et alia")  # Écourter avec "et alia" après le dernier mot si plus de 20 caractères
                order = df[var].value_counts().index[:10]
                title = f'Top 10 de {institution_name}'  if len(order) == 10 else f'{institution_name}'
                ax = sns.countplot(data=df, y=var, order=order, ax=ax)
                plt.yticks(rotation=0)
                max_value = max(df[var].value_counts())
                ax.set_xlim(0, max_value + 0.1 * max_value)
                for p in ax.patches:
                    ax.annotate(format(p.get_width(), '.0f'),
                                (p.get_width(), p.get_y() + p.get_height() / 2.),
                                ha='center', va='center',
                                xytext=(20, 0),
                                textcoords='offset points')
                ax.set(xlabel=None, ylabel=None)  # Enlever le label de l'axe x et y
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

# Utilisation :
dataframes = {"data1": data, "data2": data2}
cat_var = cat_var
plot_publications_by_category(dataframes, cat_var)


# Option pourcentage de publications
def plot_publications_by_category(dfs, cat_var):
    for var in cat_var:
        fig, axs = plt.subplots(len(dfs), 1, figsize=(11, 5.5*len(dfs)))  # Elargir le cadre des graphiques
        fig.suptitle(f'Pourcentage de publications par {var}', fontsize=16, weight='bold')
        
        for i, (df_name, df) in enumerate(dfs.items()):
            ax = axs[i]  # Utiliser un objet ax pour plus de clarté
            institution_name = df["Nom de l'établissement"].unique()[0]
            
            if var == "Année":
                order = sorted(df[var].unique())
                title = f'{institution_name}'
                ax = sns.countplot(data=df, x=var, order=order, ax=ax)
                plt.xticks(rotation=90)
                max_value = max(df[var].value_counts())
                ax.set_ylim(0, max_value + 0.1 * max_value)
                
                for p in ax.patches:
                    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                ax.set(xlabel=None, ylabel=None)  # Enlever le label de l'axe x et y
            else:
                if var == "Auteur":  # Vérifier si la variable est "Auteur"
                    df[var] = df[var].apply(lambda x: x if len(x) <= 20 else ' '.join(x.split()[:5]) + " et alia")  # Écourter avec "et alia" après le dernier mot si plus de 20 caractères
    
# VARIABLES NUMERIQUES

num_var= ["Citations", "Références effectuées",
          "Pages","Institutions associées",
          "Pays associés", "Score objectif développement durable"]

# Subplots de distribution des variables numériques

def plot_distributions(dataframes, num_var, institution_col, colors):
    num_rows = len(num_var)
    num_cols = len(dataframes)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
    
    for i, var in enumerate(num_var):
        for j, df in enumerate(dataframes):
            ax = axes[i, j]
            sns.boxenplot(y=df[var], color=colors[j % len(colors)], ax=ax)
            institution_name = df[institution_col].iloc[0]  # Récupère le nom de l'institution
            ax.set_title(f'Distribution de {var} ({institution_name})')
            ax.set_ylabel(var)

    plt.tight_layout()
    plt.show()

# Utilisation :
num_var = num_var
institution_col = "Nom de l'établissement" 
colors = colors

plot_distributions([data, data2], num_var, institution_col, colors)

# Boxenplot Zoom des variables numériques avec moyenne

def create_boxenplot(dataframes, variable, min_val, max_val, color, institution_col):
    num_rows = len(dataframes)
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 6), sharex=True)
    
    for i, df in enumerate(dataframes):
        filtered_data = df[(df[variable] >= min_val) & (df[variable] <= max_val)]
        sns.boxenplot(y=filtered_data[variable], color=color, ax=axes[i])
        institution_name = df[institution_col].iloc[0]  # Récupère le nom de l'institution
        axes[i].set_title(f'Distribution de {variable} ({institution_name})')
        axes[i].set_ylabel(variable)
        mean = filtered_data[variable].mean()
        axes[i].axhline(mean, color='r', linestyle='--')
        axes[i].text(0.5, mean, f'Moyenne : {mean:.2f}', color='r', ha='left', va='center')

    plt.tight_layout()
    plt.show()

# Utilisation :
variables_to_plot = [
    ("Citations", 0, 100),
    ("Références effectuées", 0, 100),
    ("Pages", 0, 100),
    ("Institutions associées", 0, 100),
    ("Pays associés", 0, 100),
    ("Score objectif développement durable", 0, 100),
]

institution_col = "Nom de l'établissement"  
colors = colors
create_boxenplot([data, data2], "Citations", 0, 100, colors[0], institution_col)
create_boxenplot([data, data2], "Références effectuées", 0, 100, colors[1], institution_col)
create_boxenplot([data, data2], "Pages", 0, 100, colors[2], institution_col)
create_boxenplot([data, data2], "Institutions associées", 0, 100, colors[3], institution_col)
create_boxenplot([data, data2], "Pays associés", 0, 100, colors[4], institution_col)
create_boxenplot([data, data2], "Score objectif développement durable", 0, 100, colors[5], institution_col)

#TRIS CROISES

# Test de correlation Pearson variables numériques
def pearson_matrix_subplot(data, num_var):
    n_dataframes = len(data)
    n_rows = 1
    n_cols = n_dataframes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_dataframes, 8))
    fig.subplots_adjust(wspace=0.5)

    for i, df in enumerate(data):
        pearson_mat = df[num_var].corr(method='pearson')
        sns.heatmap(pearson_mat.astype(float), annot=True, fmt=".2f", cmap='YlGnBu', ax=axes[i], cbar=False)
        institution_name = df['Nom de l\'établissement'].iloc[0]
        wrapped_title = textwrap.fill(f"{institution_name}", width=30)
        axes[i].set_title(wrapped_title, fontsize=12)
        axes[i].set_xticklabels([textwrap.fill(label, width=10) for label in pearson_mat.columns], rotation=45, ha='right')
        axes[i].set_yticklabels([textwrap.fill(label, width=10) for label in pearson_mat.index], rotation=0)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    fig.suptitle("Matrice de corrélation de Pearson", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Utilisation
pearson_matrix_subplot([data, data2], num_var)

# Subplot relation 'Citations' et variables numériques
def plot_citations_vs_num_var(data_list, num_var):
    n_rows = 3
    n_cols = 2

    for data in data_list:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 16))
        fig.subplots_adjust(hspace=0.5)
        institution_name = data['Nom de l\'établissement'].iloc[0]
        fig.suptitle(f"Relation entre Citations et les variables numériques - {institution_name}", fontsize=14, fontweight='bold')

        # Parcourez les variables numériques
        for i, var in enumerate(num_var):
            # Nuage de points ('scatter plot') de 'Citations' par rapport à la variable numérique
            sns.scatterplot(x=var, y='Citations', data=data, ax=axes[i//2, i%2], color=f'C{i}')
            # Droite de régression linéaire
            sns.regplot(x=var, y='Citations', data=data, ax=axes[i//2, i%2], scatter=False, color=f'C{i}')
            axes[i//2, i%2].set_title(f'Citations vs. {var}')
            axes[i//2, i%2].set_xlabel(var)
            axes[i//2, i%2].set_ylabel('Citations')

        # Affichez le tracé
        plt.tight_layout()
        plt.show()

# Utilisation
plot_citations_vs_num_var([data, data2], num_var)

# Croisement variables catégorielles
def plot_categoricals(data1, data2, cat_var):
    # Création d'un set pour stocker les paires de variables déjà tracées
    plotted_pairs = set()
    for i in range(len(cat_var)):
        for j in range(i+1, len(cat_var)):
            var1 = cat_var[i]
            var2 = cat_var[j]
            # Vérification que la paire n'a pas déjà été tracée
            if (var1, var2) not in plotted_pairs and (var2, var1) not in plotted_pairs:
                plotted_pairs.add((var1, var2))
                # Sélection des 10 modalités les plus fréquentes pour chaque variable
                top10_var1_data1 = data1[var1].value_counts().index[:10]
                top10_var2_data1 = data1[var2].value_counts().index[:10]
                top10_var1_data2 = data2[var1].value_counts().index[:10]
                top10_var2_data2 = data2[var2].value_counts().index[:10]
                # Tri des modalités de la variable "Année" (si c'est bien le nom de la variable)
                if var1 == "Année" or var2 == "Année":
                    top10_var1_data1 = sorted(top10_var1_data1)  # Tri chronologique
                    top10_var2_data1 = sorted(top10_var2_data1)  # Tri chronologique
                    top10_var1_data2 = sorted(top10_var1_data2)  # Tri chronologique
                    top10_var2_data2 = sorted(top10_var2_data2)  # Tri chronologique
                # Récupération du nom de l'établissement
                institution_name1 = data1["Nom de l'établissement"].iloc[0]
                institution_name2 = data2["Nom de l'établissement"].iloc[0]
                # Création du graphique
                fig, axs = plt.subplots(2, figsize=(10, 12))
                fig.suptitle(f'Relation entre {var1} et {var2} (top 10)', fontsize=14, weight='bold')
                sns.countplot(data=data1, x=var1, hue=var2, order=top10_var1_data1, hue_order=top10_var2_data1, ax=axs[0])
                axs[0].set_title(f'{institution_name1}', fontsize=12)
                axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                axs[0].set_xticklabels([label.get_text()[:20] + '\n' + label.get_text()[20:] if len(label.get_text()) > 20 else label.get_text() for label in axs[0].get_xticklabels()], rotation=90, ha='right')
                sns.countplot(data=data2, x=var1, hue=var2, order=top10_var1_data2, hue_order=top10_var2_data2, ax=axs[1])
                axs[0].set_xlabel('')  # Suppression du titre de l'axe x
                axs[1].set_title(f'{institution_name2}', fontsize=12)
                axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                axs[1].set_xticklabels([label.get_text()[:20] + '\n' + label.get_text()[20:] if len(label.get_text()) > 20 else label.get_text() for label in axs[1].get_xticklabels()], rotation=90, ha='right')
                axs[1].set_xlabel('')  # Suppression du titre de l'axe x
                plt.tight_layout()
                plt.subplots_adjust(top=0.92)
                plt.show()

# Utilisation 
plot_categoricals(data, data2, cat_var)

# Médiane par variable
no_filter = ["Année","Type de source","Type d'accès"]
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

def plot_median_citations(dataframes, cat_var, no_filter):
    for var in cat_var:
        fig, axs = plt.subplots(len(dataframes), figsize=(10, 6*len(dataframes)))
        fig.suptitle(f'Médiane de citations par "{var}"', fontsize=14, weight='bold')
        for i, df in enumerate(dataframes):
            institution_name = df['Nom de l\'établissement'].unique()[0]
            median_citations = df.groupby(var)['Citations'].median().reset_index()
            # Filtrer les 10 meilleures modalités si la variable n'est pas dans no_filter
            if var not in no_filter:
                median_citations = median_citations.sort_values('Citations', ascending=False).head(10)
            else:
                median_citations = median_citations.sort_values('Citations', ascending=False)
            # Écourter les modalités de la variable "Auteur" dépassant 20 caractères
            if var == 'Auteur':
                median_citations[var] = median_citations[var].apply(lambda x: x[:20] + ' et alia' if len(x) > 20 else x)
            barplot = sns.barplot(data=median_citations, x=var, y='Citations', order=median_citations[var], ax=axs[i])
            axs[i].set_xticklabels(['\n'.join(textwrap.wrap(label.get_text(), 20)) for label in axs[i].get_xticklabels()], rotation=90, ha='right')
            axs[i].set_title(f'Pour {institution_name}', fontsize=12)
            y_max = median_citations['Citations'].max()
            axs[i].set_ylim(0, y_max + 0.1 * y_max)
            for p in barplot.patches:
                barplot.annotate(format(p.get_height(), '.2f'), 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha='center', va='center', 
                                 xytext=(0, 10), textcoords='offset points')
            axs[i].set(xlabel=None, ylabel=None)  # Enlever le label de l'axe x et y
        axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.show()

# Utilisation
plot_median_citations([data, data2], cat_var, no_filter)

# Moyenne par variable
def plot_mean_citations(dataframes, cat_var, no_filter):
    for var in cat_var:
        fig, axs = plt.subplots(len(dataframes), figsize=(10, 6*len(dataframes)))
        fig.suptitle(f'Moyenne de citations par "{var}"', fontsize=14, weight='bold')
        for i, df in enumerate(dataframes):
            institution_name = df['Nom de l\'établissement'].unique()[0]
            mean_citations = df.groupby(var)['Citations'].mean().reset_index()
            # Filtrer les 10 meilleures modalités si la variable n'est pas dans no_filter
            if var not in no_filter:
                mean_citations = mean_citations.sort_values('Citations', ascending=False).head(10)
            else:
                mean_citations = mean_citations.sort_values('Citations', ascending=False)
            barplot = sns.barplot(data=mean_citations, x=var, y='Citations', order=mean_citations[var], ax=axs[i])
            axs[i].set_xticklabels(['\n'.join(textwrap.wrap(label.get_text(), 20)) for label in axs[i].get_xticklabels()], rotation=90, ha='right')
            axs[i].set_title(f'Pour {institution_name}', fontsize=12)
            y_max = mean_citations['Citations'].max()
            axs[i].set_ylim(0, y_max + 0.1 * y_max)
            for p in barplot.patches:
                barplot.annotate(format(p.get_height(), '.2f'), 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha='center', va='center', 
                                 xytext=(0, 10), textcoords='offset points')
            axs[i].set(xlabel=None, ylabel=None)  # Enlever le label de l'axe x et y
        axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.show()

# Utilisation
plot_mean_citations([data, data2], cat_var, no_filter)

# Ratio médianne de citations / nombre de publications par année
def plot_ratio(df_dict):
    # Create a single plot for all dataframes
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Ratio médian des citations par année / nombre de publications par année", fontsize=16, weight='bold')
    
    for i, (df_name, df) in enumerate(df_dict.items()):
        median_citations_year = df.groupby("Année")["Citations"].median()  # Calculer la médiane des citations
        count_publications_year = df.groupby("Année").size()  # Compter le nombre de publications par année
        ratio = np.divide(median_citations_year, count_publications_year)
        df_ratio = pd.DataFrame({'Année': ratio.index.astype(str), 'ratio': ratio.values})  # Traiter l'année comme une chaîne de caractères
        
        # Plot the line for the current dataframe with a unique color
        sns.lineplot(x='Année', y='ratio', data=df_ratio, ax=ax, label=df["Nom de l'établissement"].unique()[0])
        
        # Add horizontal lines
        for y in df_ratio['ratio']:
            ax.axhline(y=y, color='gray', linestyle='--', alpha=0.5)
        
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    ax.set(xlabel="", ylabel="")
    ax.legend(title="Établissement", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Utilisation :
dataframes = {"data1": data, "data2": data2}
plot_ratio(dataframes)

#  Ratio moyenne de citations par année / nombre de citations par année

import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

def plot_mean_citations(dataframes, cat_var, no_filter):
    for var in cat_var:
        fig, axs = plt.subplots(len(dataframes), figsize=(10, 6*len(dataframes)))
        fig.suptitle(f'Moyenne de citations par "{var}"', fontsize=14, weight='bold')
        for i, df in enumerate(dataframes):
            institution_name = df['Nom de l\'établissement'].unique()[0]
            mean_citations = df.groupby(var)['Citations'].mean().reset_index()
            # Filtrer les 10 meilleures modalités si la variable n'est pas dans no_filter
            if var not in no_filter:
                mean_citations = mean_citations.sort_values('Citations', ascending=False).head(10)
            else:
                mean_citations = mean_citations.sort_values('Citations', ascending=False)
            # Écourter les modalités de la variable "Auteur" dépassant 20 caractères
            if var == 'Auteur':
                mean_citations[var] = mean_citations[var].apply(lambda x: x[:20] + ' et alia' if len(x) > 20 else x)
            barplot = sns.barplot(data=mean_citations, x=var, y='Citations', order=mean_citations[var], ax=axs[i])
            axs[i].set_xticklabels(['\n'.join(textwrap.wrap(label.get_text(), 20)) for label in axs[i].get_xticklabels()], rotation=90, ha='right')
            axs[i].set_title(f'Pour {institution_name}', fontsize=12)
            y_max = mean_citations['Citations'].max()
            axs[i].set_ylim(0, y_max + 0.1 * y_max)
            for p in barplot.patches:
                barplot.annotate(format(p.get_height(), '.2f'), 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha='center', va='center', 
                                 xytext=(0, 10), textcoords='offset points')
            axs[i].set(xlabel=None, ylabel=None)  # Enlever le label de l'axe x et y
        axs[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.show()

# Utilisation
plot_mean_citations([data, data2], cat_var, no_filter)

# Ratio médiane citations/nombre des publications par discipline

def plot_ratio_subplot(df1, df2, column):
    # Group data for df1
    data_grouped1 = df1.groupby(column).agg({'Citations': ['count', 'median']})
    data_grouped1.columns = ['Nombre de publications', 'Médiane de citations']
    data_grouped1['Ratio'] = data_grouped1['Médiane de citations'] / data_grouped1['Nombre de publications']
    if column != 'Domaine disciplinaire':
        data_grouped1 = data_grouped1.sort_values('Ratio', ascending=False).head(10)
    data_grouped1.reset_index(inplace=True)

    # Group data for df2
    data_grouped2 = df2.groupby(column).agg({'Citations': ['count', 'median']})
    data_grouped2.columns = ['Nombre de publications', 'Médiane de citations']
    data_grouped2['Ratio'] = data_grouped2['Médiane de citations'] / data_grouped2['Nombre de publications']
    if column != 'Domaine disciplinaire':
        data_grouped2 = data_grouped2.sort_values('Ratio', ascending=False).head(10)
    data_grouped2.reset_index(inplace=True)

    # Create a subplot
    plt.figure(figsize=(12, 16))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    # Plot for df1
    sns.barplot(x=column, y='Ratio', data=data_grouped1, order=data_grouped1.sort_values('Ratio', ascending=False)[column], ax=ax1)
    institution_name1 = df1['Nom de l\'établissement'].iloc[0]
    ax1.set_title(f'{institution_name1}')
    ax1.set_xlabel(' ')
    ax1.set_ylabel(' ')
    ax1.set_xticklabels([textwrap.fill(label, 20) for label in data_grouped1[column]], rotation=0, ha='center', fontsize=8)

    # Plot for df2
    sns.barplot(x=column, y='Ratio', data=data_grouped2, order=data_grouped2.sort_values('Ratio', ascending=False)[column], ax=ax2)
    institution_name2 = df2['Nom de l\'établissement'].iloc[0]
    ax2.set_title(f'{institution_name2}')
    ax2.set_xlabel(' ')
    ax2.set_ylabel(' ')
    ax2.set_xticklabels([textwrap.fill(label, 20) for label in data_grouped2[column]], rotation=0, ha='center', fontsize=8)

    # Add a suptitle
    plt.suptitle(f'Ratio médiane des citations/nombre de publications par {column}', fontsize=14, fontweight='bold')

    # Show the subplot
    plt.tight_layout()
    plt.show()

# Utilisation
plot_ratio_subplot(data, data2, 'Domaine disciplinaire')
plot_ratio_subplot(data, data2, 'Champ disciplinaire')
plot_ratio_subplot(data, data2, 'Sous-champ disciplinaire')
plot_ratio_subplot(data, data2, 'Sujet disciplinaire')

# Ratio moyenne citations/nombre des publications par discipline

def plot_ratio_subplot(df1, df2, column):
    # Group data for df1
    data_grouped1 = df1.groupby(column).agg({'Citations': ['count', np.mean]})
    data_grouped1.columns = ['Nombre de publications', 'Moyenne de citations']
    data_grouped1['Ratio'] = data_grouped1['Moyenne de citations'] / data_grouped1['Nombre de publications']
    if column != 'Domaine disciplinaire':
        data_grouped1 = data_grouped1.sort_values('Ratio', ascending=False).head(10)
    data_grouped1.reset_index(inplace=True)

    # Group data for df2
    data_grouped2 = df2.groupby(column).agg({'Citations': ['count', np.mean]})
    data_grouped2.columns = ['Nombre de publications', 'Moyenne de citations']
    data_grouped2['Ratio'] = data_grouped2['Moyenne de citations'] / data_grouped2['Nombre de publications']
    if column != 'Domaine disciplinaire':
        data_grouped2 = data_grouped2.sort_values('Ratio', ascending=False).head(10)
    data_grouped2.reset_index(inplace=True)

    # Create a subplot
    plt.figure(figsize=(12, 16))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    # Plot for df1
    sns.barplot(x=column, y='Ratio', data=data_grouped1, order=data_grouped1.sort_values('Ratio', ascending=False)[column], ax=ax1)
    institution_name1 = df1['Nom de l\'établissement'].iloc[0]
    ax1.set_title(f'{institution_name1}')
    ax1.set_xlabel(' ')
    ax1.set_ylabel(' ')
    ax1.set_xticklabels([textwrap.fill(label, 20) for label in data_grouped1[column]], rotation=0, ha='center', fontsize=8)

    # Plot for df2
    sns.barplot(x=column, y='Ratio', data=data_grouped2, order=data_grouped2.sort_values('Ratio', ascending=False)[column], ax=ax2)
    institution_name2 = df2['Nom de l\'établissement'].iloc[0]
    ax2.set_title(f'{institution_name2}')
    ax2.set_xlabel(' ')
    ax2.set_ylabel(' ')
    ax2.set_xticklabels([textwrap.fill(label, 20) for label in data_grouped2[column]], rotation=0, ha='center', fontsize=8)

    # Add a suptitle
    plt.suptitle(f'Ration moyenne des citations/nombre de publications par {column}', fontsize=14, fontweight='bold')

    # Show the subplot
    plt.tight_layout()
    plt.show()

# Utilisation
plot_ratio_subplot(data, data2, 'Domaine disciplinaire')
plot_ratio_subplot(data, data2, 'Champ disciplinaire')
plot_ratio_subplot(data, data2, 'Sous-champ disciplinaire')
plot_ratio_subplot(data, data2, 'Sujet disciplinaire')


















