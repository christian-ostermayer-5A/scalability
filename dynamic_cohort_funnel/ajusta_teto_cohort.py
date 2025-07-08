#@title Def ajusta_teto_cohort

import pandas as pd

def ajusta_teto_cohort(df_cohort,nome_coluna_week_origin):
  '''
  Esta função tem como intuito limitar o teto das cohorts abertas, evitando que 
  elas passem de 100%. Definimos um dataframe auxiliar com as cohorts abertas, mergeamos
  este dataframe auxiliar com o dataframe inputado na função e verificamos quais linhas
  em cada uma das etapas do funil tem a cohort aberta maior que 100%. Para esses casos
  forçamos a cohort a ser 100% mantendo a proporção das cohorts fechadas
  que a cohort aberta tinha originalmente 
  '''

  posi_pre_inicio_dados = list(df_cohort.columns).index(nome_coluna_week_origin)

  # Separamos o df_cohort em df_cohort_1 (com as cohorts fechadas e sem a coincident) 
  # e df_cohort_coincident. Fazemos isso pois a Coincident não entra no cálculo
  # da cohort aberta. No fim daremos um append do df_out com o df_cohort_coincident
  df_cohort_1 = df_cohort[df_cohort[nome_coluna_week_origin] != 'Coincident']
  df_cohort_coincident = df_cohort[df_cohort[nome_coluna_week_origin] == 'Coincident']

  df_aux = df_cohort_1.groupby(list(df_cohort.columns[:posi_pre_inicio_dados]),as_index=False)[list(df_cohort.columns[1+posi_pre_inicio_dados:])].sum()
  
  # Renomeamos colunas para evitar duplicidade no momento do merge
  for coluna in list(df_cohort.columns[1+posi_pre_inicio_dados:]):
    df_aux = df_aux.rename(columns={coluna:f"{coluna}_aberta"})

  # Merge do dataframe original com o auxiliar
  df_out = pd.merge(df_cohort_1,df_aux,how='left',on=list(df_cohort.columns[:posi_pre_inicio_dados]))

  # Verificação da cohort aberta e correção para cada uma das etapas do funil
  for etapa in list(df_cohort.columns[(1+posi_pre_inicio_dados):]):
    df_out.loc[(df_out[f"{etapa}_aberta"] > 1),[etapa]] = df_out[etapa] * (1/df_out[f"{etapa}_aberta"])

    if len(df_out.loc[df_out[f"{etapa}_aberta"] > 1.00001]) > 0:
      df_print = df_out.loc[df_out[f"{etapa}_aberta"] > 1]
      df_print = df_print.iloc[:,1:posi_pre_inicio_dados]
      df_print = df_print.drop_duplicates()
      print(f"As seguintes aberturas da etapa {etapa} passaram de 100% em sua cohort aberta. Foi realizada a correção e as cohorts fechadas foram reduzidas na mesma proporção que tinham antes para a cohort aberta dar 100%")
      pd.options.display.max_columns = None
      pd.options.display.max_rows = None
      display(df_print)

  # Remoção das colunas auxiliares
  df_out = df_out.loc[:, ~df_out.columns.str.endswith('_aberta')]
  
  # Adição da base coincident
  df_out = df_out.append(df_cohort_coincident)
  
  return df_out
