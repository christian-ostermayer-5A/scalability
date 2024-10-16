# Conjunto de funções que, dado um histórico act e datas marcadas, 
# consegue determinar se existe uma diferença nos dados das datas marcadas com o restante da amostra e, com base
# numa interpolação, determinar qual é o impcato percentual do evento que ocorreu nas datas marcadas.
# Caso seja fornecida uma projeção, não é feita interpolação e as datas marcadas não precisam ser pontos
# isolados.



from tabulate import tabulate

def colored(texto,cor):
  if cor == 'red' or cor == 'r':
    return f'\033[1;31m{texto}\033[0;0;0m'
  elif cor == 'green' or cor == 'g':
    return f'\033[1;32m{texto}\033[0;0;0m'
  elif cor == 'yellow' or cor == 'y':
    return f'\033[1;33m{texto}\033[0;0;0m'
  elif cor == 'blue' or cor == 'b':
    return f'\033[1;34m{texto}\033[0;0;0m'
  else:
    return texto

import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta
import math
from IPython.display import clear_output
import timeit
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from gspread_dataframe import set_with_dataframe
import io
import scipy
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.spatial import distance
from scipy.stats import chi2
from scipy.interpolate import interp1d
from sklearn.ensemble import IsolationForest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from itertools import combinations
from itertools import product
import re
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.base import clone
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import HessianInversionWarning
warnings.simplefilter('ignore', ValueWarning)
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', HessianInversionWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)
np.seterr(divide='ignore', invalid='ignore')

import calendar
import unicodedata

import statistics
from statistics import mode

import itertools
from itertools import compress





def compara_feriados(df, #dataframe utilizado para as contas
                     coluna_semana, #nome da coluna de data
                     feriado, #nome da coluna de feriados
                     aberturas, #nome das aberturas quando for necessário
                     col_metrics,
                     df_comparacao_aberturas = pd.DataFrame(),
                     df_resultado_por_abertura = pd.DataFrame(),
                     df_forecast = pd.DataFrame()):
  
  #vol_weeks = ['0', '1', '2', '3', '4', '5'] #Aqui está incluído, pois é gerado exatamente assim na função de transformação de bases
  conv_weeks = col_metrics #['%__0', '%__1', '%__2', '%__3', '%__4', '%__5', '%__Volume Aberta'] #Aqui está incluído, pois é gerado exatamente assim na função de transformação de bases
  
  
  #qtd_etapas = df[col_etapas].nunique() #Criação de lista com todos as etapas únicas do funil.
  if aberturas[0] == '':
    df['concat_col'] = 'Total'
    if len(df_forecast) > 0:
      df_forecast['concat_col'] = 'Total'
  else:
    df[aberturas] = df[aberturas].astype(str)
    df['concat_col'] = df[aberturas].apply('_&_'.join, axis=1) #Separação da coluna de aberturas em colunas auxiliares
    if len(df_forecast) > 0:
      df_forecast[aberturas] = df_forecast[aberturas].astype(str)
      df_forecast['concat_col'] = df_forecast[aberturas].apply('_&_'.join, axis=1)
      
  aberturas_unicas = df['concat_col'].unique() #Criação de lista com todas as combinações de aberturas

  df.sort_values(by=[coluna_semana, 'concat_col'], ascending=[True, True]) #organiza aqui as colunas para que nas colunas auxiliares seja feito o shift correto. Entretanto, dentro do loop é feito novamente para organizar corretamente.

  # Definimos uma coluna de cluster para o caso de mistura de aberturas relevantes com irrelevantes
  df['cluster'] = ''
  # Caso tenhamos um resultado anterior, vamos separar as aberturas em clusters iniciais de aberturas boas e ruins:
  if len(df_resultado_por_abertura) > 0:
    aberturas_relevantes = df_resultado_por_abertura.loc[df_resultado_por_abertura['p-value'] < 0.05]
    if len(aberturas_relevantes) > 0:
      aberturas_relevantes['concat_col'] = aberturas_relevantes[aberturas].apply('_&_'.join, axis=1)
      aberturas_relevantes = aberturas_relevantes['concat_col'].unique()
      df['cluster'] = 'Irrelevante'
      df.loc[df['concat_col'].isin(aberturas_relevantes),'cluster'] = 'Relevante'
      df_total = df.copy()
      df_total['cluster'] = 'Total'
      df=pd.concat([df_total, df])

  aux_1_conv_weeks = [e+"_aux-1" for e in conv_weeks] #Criação de lista das colunas auxiliares
  aux_2_conv_weeks = [e+"_aux+1" for e in conv_weeks]
  col_interpoladas = [e+"_interpolada" for e in conv_weeks]
  col_delta = [e+"_delta" for e in conv_weeks]
  #conv_interpoladas = [e+"_"]

  df_resultados = df[['concat_col']]
  df_resultados['cluster'] = 0
  df_resultados['metric'] = 0
  df_resultados['p-value'] = 0
  df_resultados['impacto medio'] = 0
  df_resultados = df_resultados[['cluster','concat_col','metric', 'p-value', 'impacto medio']]
  df_resultados = df_resultados.drop(df_resultados.index)
  print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
  print(df.columns.values)
  print(df_forecast.columns.values)
  # Caso já tenhamos valores de forecast, os valores de referência interpolados serão os de forecast:
  #--------------------------------------------------------------------------------------------------------
  if len(df_forecast)>0:
    df_forecast = df_forecast.rename(columns=dict(zip(conv_weeks,col_interpoladas)))
    df = df.merge(df_forecast[[coluna_semana,'concat_col']+col_interpoladas], on=[coluna_semana,'concat_col'], how='left')
    df[col_interpoladas] = df[col_interpoladas].fillna(method='ffill')                                             

  print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
  print(df.columns.values)


  # Caso tenhamos uma comparação entre aberturas, vamos adicionar uma coluna de cluster
  #-----------------------------------------------------------------------------------------------------
  if len(df_comparacao_aberturas) > 0:
    
    # selecionamos apenas as chaves relevantes
    df_aberturas_relevantes = df_comparacao_aberturas.loc[df_comparacao_aberturas['avg p-value'] < 0.05]

    if len(df_aberturas_relevantes) > 0:
      lista_chaves_relevantes = df_aberturas_relevantes['chave'].unique()

      # para cada chave relevante, vamos identificá-la na coluna de abertura única e formar clusters na base principal
      for chave in lista_chaves_relevantes:

        if len(df[df['concat_col'].str.contains(chave+"_&", case=False, na=False)]) > 0:
          df.loc[df['concat_col'].str.contains(chave+"_&", case=False, na=False),'cluster'] = df.loc[df['concat_col'].str.contains(chave+"_&", case=False, na=False),'cluster']+"___"+chave

        elif len(df[df['concat_col'].str.contains("&_"+chave, case=False, na=False)]) > 0:
          df.loc[df['concat_col'].str.contains("&_"+chave, case=False, na=False),'cluster'] = df.loc[df['concat_col'].str.contains("&_"+chave, case=False, na=False),'cluster']+"___"+chave


  
  # Vamos agrupar a análise dos impactos por cluster de aberturas:

  lista_resultados = []
  for cluster in df['cluster'].unique():

    df_cluster = df.loc[df['cluster'] == cluster]

    cluster_df_list_normal = []
    cluster_df_list_feriado = []

    for a in aberturas_unicas:
      df_aux = df_cluster.copy()
      
      df_aux = df_aux.loc[df_aux['concat_col'] == a]
      df_aux = df_aux.sort_values(by=[coluna_semana], ascending=[True]) #organiza aqui as colunas para que nas colunas auxiliares seja feito o shift correto.
      
      # Caso NÃO tenhamos valores de forecast, PRECISAMOS fazer a interpolação:
      #--------------------------------------------------------------------------------
      if len(df_forecast) == 0:

        df_aux[aux_1_conv_weeks] = df_aux[conv_weeks].shift(periods=1, freq=None, axis=0) #shift de linhas para que na mesma linha exista o valor anterior e posterior
        df_aux[aux_2_conv_weeks] = df_aux[conv_weeks].shift(periods=-1, freq=None, axis=0)
        df_aux.drop(df_aux.head(1).index,inplace=True) # drop first x rows
        df_aux.drop(df_aux.tail(1).index,inplace=True) # drop last x rows


        for e in range (len(col_interpoladas)):

          df_aux[col_interpoladas[e]] = (df_aux[aux_1_conv_weeks[e]] + df_aux[aux_2_conv_weeks[e]])/2 #média entre o valor anterior e valor posterior

          df_aux[col_delta[e]] = (df_aux[conv_weeks[e]].astype(float).values/df_aux[col_interpoladas[e]].astype(float).values) - 1
      
      # Caso tanhamos forecast, os valores delta serão entre o forecast e o act:
      print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
      print(df_aux.columns.values)
      print(col_delta)
      print(conv_weeks)
      print(col_interpoladas)
      if len(conv_weeks) == 1:
        df_aux[col_delta[0]] = df_aux[conv_weeks[0]].astype(float).values/df_aux[col_interpoladas[0]].astype(float).values - 1
      else:
        df_aux[col_delta] = df_aux[conv_weeks].astype(float).values/df_aux[col_interpoladas].astype(float).values - 1
      #-----------------------------------------------------------------------------------


      # Separamos as amostras:
      df_aux = df_aux.fillna(0)
      df_normal = df_aux.loc[df_aux[feriado] == 0]
      df_feriado = df_aux.loc[df_aux[feriado] == 1]    

      df_normal.replace([np.inf, -np.inf], 0.0, inplace=True) #removendo os valores infinitos
      df_feriado.replace([np.inf, -np.inf], 0.0, inplace=True) #removendo os valores infinitos
      


      # Caso não estejamos rodando com clusters, vamos calcular o impacto por linha de combinação de aberturas:
      if len(df_comparacao_aberturas) == 0:
        for i in range(len(conv_weeks)): #Nesse for, fazemos o teste t e o p-valor entre os deltas dos valores originais vs os interpolados.

          t_stat, p_val = ttest_ind(df_normal[col_delta[i]], df_feriado[col_delta[i]])
          '''
          print("1-------------------------------------------------")
          print(a)
          print(conv_weeks[i])
          print(np.average(df_normal[col_delta[i]]))
          print(np.average(df_feriado[col_delta[i]]))
          print(len(df_normal[col_delta[i]].values),len(df_feriado[col_delta[i]].values))
          '''
          lista_resultado = [[cluster, a, conv_weeks[i], p_val, round(np.average(df_feriado[col_delta[i]].values),5)]]
          df_novo = pd.DataFrame(lista_resultado, columns=df_resultados.columns)
          df_resultados = pd.concat([df_resultados, df_novo], ignore_index=True)



      # caso contrário, agrupamos todas as aberturas do mesmo cluster
      else:
        cluster_df_list_normal = cluster_df_list_normal + [df_normal]
        cluster_df_list_feriado = cluster_df_list_feriado + [df_feriado]


    # Caso estejamos calculando o impactos por cluster, devemos fazer isso agora:
    if len(df_comparacao_aberturas) > 0:
      # Criamos uma grande base de deltas de todas as aberturas do mesmo cluster
      df_normal = pd.concat(cluster_df_list_normal, ignore_index=True)
      df_feriado = pd.concat(cluster_df_list_feriado, ignore_index=True)


      # Para cada cluster, calculamos os p-valores e impactos médios:
      for i in range(len(conv_weeks)): #Nesse for, fazemos o teste t e o p-valor entre os deltas dos valores originais vs os interpolados.

        t_stat, p_val = ttest_ind(df_normal[col_delta[i]], df_feriado[col_delta[i]])
        '''
        print("2-------------------------------------------------")
        print(cluster)
        print(conv_weeks[i])
        print(np.average(df_normal[col_delta[i]]))
        print(np.average(df_feriado[col_delta[i]]))
        print(len(df_normal[col_delta[i]].values),len(df_feriado[col_delta[i]].values))
        '''

        lista_resultado = [[cluster, cluster, conv_weeks[i], p_val, round(np.average(df_feriado[col_delta[i]].values),5)]]
        df_novo = pd.DataFrame(lista_resultado, columns=df_resultados.columns)
        df_resultados = pd.concat([df_resultados, df_novo], ignore_index=True)
      


      #lista_resultados = lista_resultados + [df_resultados]

  # Caso estejamos calculando o impactos por cluster, devemos formatar a base de resultados:
  if len(df_comparacao_aberturas) > 0:
    #df_resultados = pd.concat(lista_resultados, ignore_index=True)
    df_resultados[aberturas] = 'Total'

    for abertura in aberturas:
      chaves = df[abertura].unique()

      for chave in chaves:

        if len(df_resultados[df_resultados['concat_col'].str.contains("___"+chave, case=False, na=False)]) > 0:
          df_resultados.loc[df_resultados['concat_col'].str.contains("___"+chave, case=False, na=False),abertura] = chave
    

    df_resultados = df_resultados[['cluster']+aberturas+['metric', 'p-value', 'impacto medio']] #Reordena as colunas

  # Caso contrário, formatamos o resultado final por abertura:
  else:
    if aberturas[0] != '':
      df_resultados[aberturas] = df_resultados['concat_col'].str.split('_&_', expand=True) #Separação da coluna de aberturas combinas em colunas auxiliares. Basicamente retorna as colunas iniciais de abertura.
      df_resultados = df_resultados.drop('concat_col', axis=1) #Exclui a coluna 'concat_col'
      df_resultados = df_resultados[['cluster']+aberturas + ['metric', 'p-value', 'impacto medio']] #Reordena as colunas

    else:
      df_resultados = df_resultados.drop('concat_col', axis=1) #Exclui a coluna 'concat_col'
      df_resultados = df_resultados[['cluster']+['metric', 'p-value', 'impacto medio']] #Reordena as colunas


  return df_resultados



# Comparação entre as aberturas
#____________________________________________________________________________________________________________________________________

def compara_impacto_aberturas(df_resultados,
                              aberturas):

  # Filtrar somente aberturas com impacto significativo:
  df_resultados_significativos = df_resultados.loc[df_resultados['p-value'] < 0.05]

  # Vamos salvar as comparações entre aberturas numa base:
  #df_comparacao_aberturas = pd.DataFrame(columns=['abertura','chave']+col_metrics)
  list_results = [[]]

  if len(df_resultados_significativos) > 0 and aberturas != '':

    # Para cada abertura, vamos comparar os impactos em cada métrica a cada par de chaves:
    for abertura in aberturas:
      unique_keys = df_resultados_significativos[abertura].unique()
      pares_chaves = list(combinations(unique_keys, 2))

      # Comparamos os impactos entre cada par de chaves:
      for par in pares_chaves:

        list_pvalue_metric = []

        for metric in col_metrics:

          impacts_par1 = df_resultados_significativos.loc[(df_resultados_significativos[abertura] == par[0]) & (df_resultados_significativos['metric'] == metric)]['impacto medio'].values
          impacts_par2 = df_resultados_significativos.loc[(df_resultados_significativos[abertura] == par[-1]) & (df_resultados_significativos['metric'] == metric)]['impacto medio'].values

          t_stat, p_val = ttest_ind(impacts_par1,impacts_par2)
          '''
          print("_____________________________________________________________________")
          print(metric,par,p_val)
          print("-------------------------------------------")
          print(impacts_par1)
          print("-------------------------------------------")
          print(impacts_par2)
          '''
          list_aux = [abertura, par[0], metric, p_val]
          list_results = list_results+[list_aux]

          list_aux = [abertura, par[-1], metric, p_val]
          list_results = list_results+[list_aux]            
        

  # transformamos as listas de listas de resultados em um único dataframe:
  df_comparacao_aberturas = pd.DataFrame(list_results[1:],columns=['abertura','chave','metric','avg p-value'])

  # Vamos agrupar o resultado pela média dos p-valores:
  df_comparacao_aberturas = df_comparacao_aberturas.groupby(['abertura','chave','metric']).agg({'avg p-value': 'mean'}).reset_index()

  return df_comparacao_aberturas
