import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# GHI 2000>2022
ghi_df = pd.read_excel("ghi_2000_2022_final_dataframe.xlsx")
ghi_df

ghi_df.info()

# 129 lignes et 4 colonnes
# ghi_final.shape

# 1/ GESTION NaN - SOMME DES VALEURS MANQUANTES PAR ANNEE
# msno.matrix(ghi_df)

ghi_df.isna().sum()  # total des nan :
# 2000     17
# 2007     14
# 2014     13
# 2022     4

# 117 valeurs non manquantes
(ghi_df['2000'] == ghi_df['2000']).sum()

# différence entre valeurs totales et nan
# 2000 : 12 valeurs manquantes
# 2007 : 9 valeurs manquantes
# 2014 : 9 valeurs manquantes
# 2022 : 1 valeur manquantes
len(ghi_df) - (ghi_df['2000'] == ghi_df['2000']).sum()
len(ghi_df) - (ghi_df['2007'] == ghi_df['2007']).sum()
len(ghi_df) - (ghi_df['2014'] == ghi_df['2014']).sum()
len(ghi_df) - (ghi_df['2022'] == ghi_df['2022']).sum()

ghi_df.columns = ["Country", "GHI_2000", "GHI_2007",
                  "GHI_2014", "GHI_2022", "country_missing_data"]
ghi_df.dtypes

# convertir colonne "ghi_2022 > object" en float
ghi_df["GHI_2022"] = pd.to_numeric(ghi_df['GHI_2022'], errors='coerce')
print(ghi_df.info())

# 2/ DATA CLEANING - UNIFORMISER LES NOMS DES PAYS
ghi_df["Country"] = ghi_df["Country"].apply(
    lambda x: x.split(' (')[0])
ghi_df["Country"] = ghi_df["Country"].apply(
    lambda x: ''.join([a for a in x if not a.isnumeric()]))
ghi_df["Country"] = ghi_df["Country"].replace(
    "Türkiye", "Turkey", regex=True)
ghi_df["Country"] = ghi_df["Country"].replace(
    "Russian Federation", "Russia", regex=True)
ghi_df["Country"] = ghi_df["Country"].replace("&", "and", regex=True)
ghi_df["Country"] = ghi_df["Country"].replace(
    "Côte d'Ivoire", "Cote d'Ivoire", regex=True)
ghi_df["Country"] = ghi_df["Country"].replace(
    "Korea", "North Korea", regex=True)
ghi_df["Country"] = ghi_df["Country"].replace(
    "Rep.", "Republic", regex=True)
ghi_df["Country"] = ghi_df["Country"].replace(
    "Dem.", "Democratic", regex=True)
ghi_df["Country"] = ghi_df["Country"].replace(
    "the Congo", "Congo", regex=True)
ghi_df["Country"] = ghi_df["Country"].replace(
    "Lao PDR", "Laos", regex=True)
ghi_df["Country"] = ghi_df["Country"].replace(
    "Slovak Republicblic", "Slovakia", regex=True)
ghi_df["Country"] = ghi_df["Country"].replace(
    "Dominican Republicblic", "Dominican Republic", regex=True)

ghi_df["Country"].unique()

ghi_df.shape  # 134 lignes et 6 colonnes
ghi_df.sample(80)

# sauvegarde du nouveau dataframe
ghi_df.to_csv('GHI_2000_2022_final_dataframe.csv', index=True)

# interpolation avec méthode linéaire pour remplir les nan de manière plus réaliste
# interpole = ghi_df.interpolate(axis=0).plot()
# Baisse de l'indice de la fin en comparaison entre les courbes de 2000 et 2022.

country_missing_data = ghi_df["country_missing_data"].sum()
print(f"Il y a {country_missing_data} pays dont les scores individuels n'ont pas pu être calculés en raison du manque de données.")

# cols = ghi_df["GHI_2000", ]
# for i in range (ghi_df)
