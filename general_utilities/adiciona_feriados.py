#@title Def adiciona_feriados
import pandas as pd
from strip_accents import strip_accents


def adiciona_feriados(df,           # DataFrame com os dados semanais onde queremos adicionar uma coluna de feriados
                      col_data,     # String do nome da coluna que contém as datas semanais no DataFrame (já deve estar no formato dateTime)
                      col_regiao,   # String do nome da coluna que contém as regiões no DataFrame
                      df_feriados): # DataFrame com os feriados

  df_feriados[['dia','mês','ano']] = df_feriados[['dia','mês','ano']].fillna(0)

  df_feriados[['dia','mês','count']] = df_feriados[['dia','mês','count']].astype(int)

  # Selecionamos apenas as colunas que importam:
  df[col_data] = pd.to_datetime(df[col_data], infer_datetime_format = True) #Convertendo datas para datetime
  out = df[[col_data,col_regiao]]
  out['aux'] = 1
  out = out.groupby([col_data,col_regiao], as_index=False)[['aux']].sum()
  out = out[[col_data,col_regiao]]

  # Vamos criar uma coluna de municípios
  out['município'] = out[col_regiao].apply(lambda x: strip_accents(x.upper()))
  out = out.loc[out['município'] != '']

  # Vamos criar uma coluna de sigla de estados
  df_feriados_de_para = df_feriados.groupby(['município','sigla estado'], as_index=False)[['count']].sum()
  out = pd.merge(out,df_feriados_de_para[['município','sigla estado']],how='left',on=['município'])

  # Vamos transformar a base semanal em diária:
  out['dias'] = out[col_data].values
  out['week_day'] = 1
  aux = out.copy()
  list_aux = [out]
  for i in range(1,7):
    aux['dias'] = aux['dias']+pd.Timedelta(1, unit='D')
    aux['week_day'] = aux['week_day'].values + 1
    list_aux = list_aux+[aux.copy()]
  out = pd.concat(list_aux)

  # Vamos criar uma coluna com o dia, outra com o mês e outra com o ano
  out['dia'] = out['dias'].apply(lambda x: x.day)
  out['mês'] = out['dias'].apply(lambda x: x.month)
  out['ano'] = out['dias'].apply(lambda x: x.year)


  # Vamos criar uma coluna com os feriados nacionais
  df_feriados_N_a = df_feriados.loc[(df_feriados['tipo'] == 'N') & (df_feriados['ano'] != '') & (df_feriados['ano'] != 0)]
  df_feriados_N_a['ano'] = df_feriados_N_a['ano'].astype(int)

  df_feriados_N_b = df_feriados.loc[(df_feriados['tipo'] == 'N') & (df_feriados['ano'] == '') & (df_feriados['ano'] != 0)]

  out = pd.merge(out,df_feriados_N_a[['dia','mês','ano','count']],how='left',on=['dia','mês','ano'])
  out = out.rename(columns={"count": "Feriado_N_a"})

  out = pd.merge(out,df_feriados_N_b[['dia','mês','count']],how='left',on=['dia','mês'])
  out = out.rename(columns={"count": "Feriado_N_b"})

  # Vamos criar uma coluna com os feriados estaduais
  df_feriados_E = df_feriados.loc[df_feriados['tipo'] == 'E']
  out = pd.merge(out,df_feriados_E[['dia','mês','sigla estado','count']],how='left',on=['dia','mês','sigla estado'])
  out = out.rename(columns={"count": "Feriado_E"})

  # Vamos criar uma coluna com os feriados municipais
  df_feriados_M = df_feriados.loc[df_feriados['tipo'] == 'M']
  out = pd.merge(out,df_feriados_M[['dia','mês','município','count']],how='left',on=['dia','mês','município'])
  out = out.rename(columns={"count": "Feriado_M"})

  # Somamos os feriados
  out = out.fillna(0)

  out['Feriado'] = out[['Feriado_N_a',	'Feriado_N_b', 'Feriado_E', 'Feriado_M']].sum(axis=1)

  # Selecionamos apenas as colunas que importam
  out = out[[col_data,col_regiao,'week_day','Feriado']]

  # Pivotamos os dias da semana
  out = pd.pivot_table(out, values='Feriado', index=[col_data,col_regiao], columns='week_day').reset_index(inplace=False)

  # Renomeamos as colunas:
  out = out.rename(columns={1: "Feriado_1",2: "Feriado_2",3: "Feriado_3",4: "Feriado_4",5: "Feriado_5",6: "Feriado_6",7: "Feriado_7"})

  # Criamos a coluna de feriado geral:
  # Nomeamos o feriado de (dummy) para que seja aplicado em relação à média dos valores da endógena afetada
  out['Feriado (dummy)'] = out[['Feriado_1',	'Feriado_2', 'Feriado_3', 'Feriado_4', 'Feriado_5', 'Feriado_6', 'Feriado_7']].sum(axis=1)

  # Não encontramos efeito significativo de feriado por dia da semana
  out = out.drop(columns=['Feriado_1',	'Feriado_2', 'Feriado_3', 'Feriado_4', 'Feriado_5', 'Feriado_6', 'Feriado_7'])

  # Adicionamos as colunas de feriados na base
  df = pd.merge(df,out,how='left',on=[col_data,col_regiao])

  return df
