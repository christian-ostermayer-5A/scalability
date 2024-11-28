#@title Def check_inputs_exogenas
import pandas as pd
import numpy as np
from colored import colored

def check_inputs_exogenas(df_inputs,
                          col_valores_exogenas,
                          conversoes,
                          topos,
                          max_lag,
                          max_origin,
                          nome_do_arquivo):

  mensagem = ''
  erro = 0
  nome = df_inputs.name

  # Checando o conteúdo das colunas:

  shares_permitidos = ['Share W'+str(s) for s in range(max_origin)]
  lags_permitidos = [str(l) for l in range(max_lag+1)]
  meses = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']

  colunas = ['etapa','endógena','ação','exógena','lag ou diff','slope sanity check']
  colunas_2 = ['Etapa','Endógena','Ação','Exógena','Lag ou Diff','Slope Sanity Check']
  df_inputs = df_inputs.rename(columns=dict(zip(colunas_2, colunas)))

  # selecionar aberturas e manter na ordem original
  lista_aberturas = list(set(df_inputs.columns.values) - set(colunas))
  aberturas = [x for x in df_inputs.columns.values if x in lista_aberturas]

  lista_itens_obrigatorios = [conversoes,
                              ['Volume','Cohort Aberta']+shares_permitidos,
                              ['Incluir','Check'],
                              ['ordem_semana','Tempo Numérico','Feriado','Year','Volume']+meses+col_valores_exogenas,
                              ['d']+lags_permitidos+['log'],
                              ['-1','0','1']]

  for i in range(len(colunas)):

    coluna = colunas[i]
    itens_obrigatorios = lista_itens_obrigatorios[i]
    itens = list(df_inputs[coluna].unique())
    itens = [e for e in itens if e != 'Todos']

    diferenca = list(set(itens) - set(itens_obrigatorios))

    if len(diferenca) > 0:
      mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome,'yellow')+', na coluna '+colored(str(coluna),'yellow')+', os seguintes itens estão errados: '+colored(str(diferenca),'red')+'.\nOs únicos itens permitidos nesta coluna são: '+colored(str(itens_obrigatorios),'red')+'.'
      erro += 1


  # Checando se a endógena de volume está numa etapa que contém ToF
  if erro == 0:

    inputs_etapa_volume = list(df_inputs.loc[df_inputs['endógena'] == 'Volume']['etapa'].unique())

    for e in inputs_etapa_volume:

      if e == 'Todos':
        mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome,'yellow')+', só é possível incluir séries exógenas para a série endógena de "Volume" nas etapas de funil que iniciam em um dos seguintes topos de funil: '+str(topos)+'.\nA etapa '+colored(e,'red')+' não é válida.'
        erro += 1
      elif e.split('2')[0] not in topos:
        mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome,'yellow')+', só é possível incluir séries exógenas para a série endógena de "Volume" nas etapas de funil que iniciam em um dos seguintes topos de funil: '+str(topos)+'.\nA etapa '+colored(e,'red')+' não é válida.'
        erro += 1
      else:
        mensagem = mensagem


  # Separando a base de inputs da base de sanity check:
  if len(df_inputs)>0:
    df_sanity_check = df_inputs[['exógena','slope sanity check']]
    df_sanity_check['slope sanity check'] = df_sanity_check['slope sanity check'].astype(float)
    df_sanity_check['exógena'] = df_sanity_check['exógena'].apply(lambda x: x.split('___')[0])
    df_sanity_check['exógena'] = df_sanity_check['exógena'].apply(lambda x: x.split('_t')[0])
    df_sanity_check = df_sanity_check.groupby(['exógena'],as_index=False)['slope sanity check'].sum()
    df_sanity_check.loc[df_sanity_check['slope sanity check'] > 0,['slope sanity check']] = 1
    df_sanity_check.loc[df_sanity_check['slope sanity check'] < 0,['slope sanity check']] = -1
  else:
    df_sanity_check = pd.DataFrame()

  if 'Incluir' in list(df_inputs['ação'].values):
    df_inputs = df_inputs.loc[df_inputs['ação'] == 'Incluir',aberturas+colunas[:-1]]
  else:
    df_inputs = df_inputs[aberturas+colunas[:-1]]


  df_inputs.name = nome
  return df_inputs,df_sanity_check,mensagem,erro



#__________________________________________________________________________________________________

#@title Def check_inputs_manuais

def check_inputs_manuais(df_inputs,
                          col_valores_exogenas,
                          conversoes,
                          topos,
                          max_lag,
                          max_origin,
                          nome_do_arquivo):

  mensagem = ''
  erro = 0
  nome = df_inputs.name

  # Checando o conteúdo das colunas:

  cohorts_permitidas = [str(s) for s in range(max_origin+1)]

  colunas = ['etapa','cohort','métrica']
  colunas_2 = ['Etapa','Cohort','Métrica']
  df_inputs = df_inputs.rename(columns=dict(zip(colunas_2, colunas)))

  lista_itens_obrigatorios = [conversoes,
                              ['Coincident','Aberta']+cohorts_permitidas,
                              ['Volume','Cohort']]

  for i in range(len(colunas)):


    coluna = colunas[i]
    itens_obrigatorios = lista_itens_obrigatorios[i]
    itens = list(df_inputs[coluna].unique())
    itens = [e for e in itens if e != 'Todos']


    diferenca = list(set(itens) - set(itens_obrigatorios))

    if len(diferenca) > 0:
      mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome,'yellow')+', na coluna '+colored(str(coluna),'yellow')+', os seguintes itens estão errados: '+colored(str(diferenca),'red')+'.\nOs únicos itens permitidos nesta coluna são: '+colored(str(itens_obrigatorios),'red')+'.'
      erro += 1



  # Checando se a endógena de volume está numa etapa que contém ToF
  if erro == 0:

    inputs_etapa_volume = list(df_inputs.loc[df_inputs['métrica'] == 'Volume']['etapa'].unique())

    for e in inputs_etapa_volume:

      if e == 'Todos':
        mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome,'yellow')+', só é possível aplicar inputs para a série endógena de "Volume" nas etapas de funil que iniciam em um dos seguintes topos de funil: '+str(topos)+'.\nA etapa '+colored(e,'red')+' não é válida.'
        erro += 1
      elif e.split('2')[0] not in topos:
        mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome,'yellow')+', só é possível aplicar inputs para a série endógena de "Volume" nas etapas de funil que iniciam em um dos seguintes topos de funil: '+str(topos)+'.\nA etapa '+colored(e,'red')+' não é válida.'
        erro += 1
      else:
        mensagem = mensagem


  df_inputs.name = nome
  return df_inputs,mensagem,erro

#__________________________________________________________________________________________________________________
