import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import missingno as msno

import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline as pyo

import plotly.graph_objs as go

pyo.init_notebook_mode(connected=True)


#from pandas_profiling import ProfileReport

#from autoviz.Autoviz_Class import Auto


# DATA IMPORTING  - GHI 2000>2022

ghi_df_final = pd.read_excel("ghi_2000_2022_final_dataframe.xlsx")


ghi_df_final.info()


# 129 lignes et 4 colonnes

# ghi_final.shape


# 1/ GESTION NaN - SOMME DES VALEURS MANQUANTES PAR ANNEE

#msno.matrix(ghi_df_final_final)

ghi_df_final.isna().sum()  # total des nan :

# 2000     16

# 2007     13

# 2014     12

# 2022     4


# 117 valeurs non manquantes
(ghi_df_final['2000'] == ghi_df_final['2000']).sum()


# différence entre valeurs totales et nan

# 2000 : 12 valeurs manquantes

# 2007 : 9 valeurs manquantes

# 2014 : 9 valeurs manquantes

# 2022 : 1 valeur manquantes

len(ghi_df_final) - (ghi_df_final['2000'] == ghi_df_final['2000']).sum()

len(ghi_df_final) - (ghi_df_final['2007'] == ghi_df_final['2007']).sum()

len(ghi_df_final) - (ghi_df_final['2014'] == ghi_df_final['2014']).sum()

len(ghi_df_final) - (ghi_df_final['2022'] == ghi_df_final['2022']).sum()


ghi_df_final.columns = ["Country", "GHI_2000", "GHI_2007",

                  "GHI_2014", "GHI_2022", "country_missing_data"]
ghi_df_final.dtypes


# convertir colonne "ghi_2022 > object" en float
ghi_df_final["GHI_2022"] = pd.to_numeric(ghi_df_final['GHI_2022'], errors='coerce')
print(ghi_df_final.info())


# 2/ DATA CLEANING - UNIFORMISER LES NOMS DES PAYS
ghi_df_final["Country"] = ghi_df_final["Country"].apply(

    lambda x: x.split(' (')[0])
ghi_df_final["Country"] = ghi_df_final["Country"].apply(

    lambda x: ''.join([a for a in x if not a.isnumeric()]))
ghi_df_final["Country"] = ghi_df_final["Country"].replace(

    "Türkiye", "Turkey", regex=True)
ghi_df_final["Country"] = ghi_df_final["Country"].replace(

    "Russian Federation", "Russia", regex=True)
ghi_df_final["Country"] = ghi_df_final["Country"].replace("&", "and", regex=True)
ghi_df_final["Country"] = ghi_df_final["Country"].replace(

    "Côte d'Ivoire", "Cote d'Ivoire", regex=True)
ghi_df_final["Country"] = ghi_df_final["Country"].replace(

    "Korea", "North Korea", regex=True)
ghi_df_final["Country"] = ghi_df_final["Country"].replace(

    "Rep.", "Republic", regex=True)
ghi_df_final["Country"] = ghi_df_final["Country"].replace(

    "Dem.", "Democratic", regex=True)
ghi_df_final["Country"] = ghi_df_final["Country"].replace(

    "the Congo", "Congo", regex=True)
ghi_df_final["Country"] = ghi_df_final["Country"].replace(

    "Lao PDR", "Laos", regex=True)
ghi_df_final["Country"] = ghi_df_final["Country"].replace(

    "Slovak Republicblic", "Slovakia", regex=True)
ghi_df_final["Country"] = ghi_df_final["Country"].replace(

    "Dominican Republicblic", "Dominican Republic", regex=True)

ghi_df_final["Country"].unique()


# retirer les doublons
ghi_df_final = ghi_df_final.drop_duplicates()
ghi_df_final.shape # 133 LIGNES et 6 colonnes


# valeurs GHI_2022 pour Syria, Burundi, Somalia, South Sudan à rajouter car transformer en NaN par interpolation

South_Sudan = ghi_df_final.loc[ghi_df_final['Country'] == 'South Sudan', 'GHI_2022'] = 42.45

Somalia = ghi_df_final.loc[ghi_df_final['Country'] == 'Somalia', 'GHI_2022'] = 42.45

Burundi = ghi_df_final.loc[ghi_df_final['Country'] == 'Burundi', 'GHI_2022'] = 42.45
syria_data = ghi_df_final.loc[ghi_df_final['Country'] == 'Syria', 'GHI_2022'] = 42.45


ghi_df_final.sample(80)


# sauvegarde du nouveau dataframe ("cleané")
ghi_df_final.to_csv('GHI_2000_2022_final_dataframe.csv', index=True)


# *****************************************************************************************


# Baisse de l'indice de la fin en comparaison entre les courbes de 2000 et 2022.


# 3/ COUNTRY WITH MISSING OR INCOMPLETE DATA
country_missing_data = ghi_df_final["country_missing_data"].sum()

print(f"Il y a {country_missing_data} pays dont les scores individuels n'ont pas pu être calculés en raison du manque de données.")


# ----------------- ANALYSIS -----------------------
# ------------------- EDA --------------------------
# global GHI distribution
ghi_df_final.plot(axis=1)


# Hunger Index Distribution plot in 2000, 2007, 2014, 2022 

ghi_year = ['GHI_2000', 'GHI_2007', 'GHI_2014', 'GHI_2022']

fig = plt.figure(figsize=(15,5))


for i in range(len(ghi_year)):

    plt.subplot(1,4,i+1)
    plt.title(ghi_year[i])
    sns.distplot(ghi_df_final,x=ghi_df_final[ghi_year[i]])

    plt.xlim([0,75])

    plt.ylim([0,0.06])
plt.tight_layout()

plt.show()


# pillow of plots for comparision

for i in ghi_year:
    sns.distplot(ghi_df_final[i], axlabel='Hunger Index')


# Year 2000 : Countries with GHI over 50 (extreme alarming >=50) in 2000

ext_al_countries_2000 = ghi_df_final[ghi_df_final["GHI_2000"]>=50].sort_values(by="GHI_2000", ascending=True)

sns.catplot(ext_al_countries_2000, x='GHI_2000', y='Country', ci=None, kind='bar', aspect=2)

plt.title('Extremely Alarming (>=50) Hunger Index Countries in 2000', fontweight="bold", fontsize=14)


# Year 2022 : Which countries are still in an alarming scale over 35 in 2022 ?
alarm = ghi_df_final[ghi_df_final["GHI_2022"]>=35].sort_values(by="GHI_2022", ascending=True)

figma = px.bar(alarm, x='GHI_2022', y='Country', orientation='h')

figma.update_layout(title='Alarming Hunger Index Countries in 2022 (GHI>=35)', font=dict(size=14))

figma.show()


# **********************************************************************
# Calculate of GHI difference between 2000 and 2022
ghi_df_final['score_variation'] = ghi_df_final['GHI_2022'] - ghi_df_final['GHI_2000']

# scores ranked by ascending order
high_score_of_variation = ghi_df_final[['Country','score_variation']].sort_values(by="score_variation", ascending=True)

# save csv "GHI SCORES 2022 ORDERED BY SCORE"
high_score_of_variation.to_csv('GHI_scores_2022.csv', index=True)

# ---------- the 15 BEST DECREASING SCORES  ------------
best_score_of_variation = high_score_of_variation[['Country','score_variation']].sort_values(by="score_variation", ascending=True)[:15]

# ----- Visualize the GHI Variation from 2000 for extremely alarming coountries ------
trace = go.Scatterpolar(
    r = best_score_of_variation['score_variation'].tolist(),
    theta = best_score_of_variation['Country'].tolist(),
    fill = 'toself',
    hovertext = best_score_of_variation['score_variation'].tolist(),
    name = 'GHI difference 2000-2022'
)


# Create a layout for the radar chart
layout = go.Layout(
    polar = dict(
        radialaxis = dict(
            visible = True,
            range = [0, -45]
        )
    ),

    title={
        'text': "Best GHI Scores 2000-2022",
        'font': {'size': 12},
        'x':0.5,
        'y':0.9,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)
# Create the radar chart
radar = go.Figure(data=[trace], layout=layout)
radar.show()

# Comparison between the highest Hungry Index Countries and the best scores Variation
# Create two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first subplot
ext_al_countries_2000.plot(kind='barh', x='Country', y='GHI_2000', ax=ax1)
ax1.set_xlabel('GHI 2000')
ax1.set_title('Extremely Alarming Countries in 2000')

# Plot the second subplot
best_score_of_variation.plot(kind='barh', x='Country', y='score_variation', ax=ax2)
ax2.set_xlabel('GHI variation')
ax2.set_title('the best 25 GHI Scores')

# Show the plot
plt.show()

# 2/ Bar Plot for comparison btw GHI_2000 of extreme alarming countries & their Scores in 2022
# Create the bar trace for the first subplot
trace1 = go.Bar(
    x=ext_al_countries_2000['GHI_2000'],
    y=ext_al_countries_2000['Country'],
    orientation='h',
    name='Global hungry Index in 2000'
)

# Create the bar trace for the second subplot
trace2 = go.Bar(
    x=best_score_of_variation['score_variation'],
    y=best_score_of_variation['Country'],
    orientation='h',
    name='Score variation (Index)'
)

# Define the layout for the subplots
layout = go.Layout(
    title={
        'text': "Comparison between the highest Hungry Index Countries and the best scores Variation",
        'font': {'size': 11},
        'x':0.5,
        'xanchor': 'center',
    },
    height=600,
    margin=dict(l=100, r=50, b=50, t=80),
    yaxis=dict(title='Country'),
    xaxis1=dict(domain=[0, 0.45]),
    xaxis2=dict(title='GHI variation', domain=[0.55, 1])
)

# Create the subplot figure
fig = go.Figure(data=[trace1, trace2], layout=layout)

# Show the plot
fig.show()