#@title Def gerador_baseline
import pandas as pd
import numpy as np

def gerador_baseline(forcast_cohort,
                     inputs_substituicao_df,
                     grupos_inputs_df,
                     aberturas,
                     max_origin,
                     primeira_data_baseline,
                     sub_aberta,
                     sub_share):


  col_data = 'week_start'

  max_origin = str(max_origin)

  forcast_cohort_aux = forcast_cohort.copy()
  #print(forcast_cohort_aux.loc[(forcast_cohort_aux['city_group'] == 'Belém') & (forcast_cohort_aux['mkt_channel'] == 'CRM/Notification')])
  cb = list(forcast_cohort_aux.columns.values)

  if 'Week Origin' not in cb:
    col_week_origin = 'week origin'
  else:
    col_week_origin = 'Week Origin'

  chaves = aberturas + [col_week_origin]

  valores_aux = list(set(cb) - set([col_data]+chaves))
  valores = [v for v in cb if v in valores_aux]

  valores_x = [v+'_x' for v in valores]
  valores_y = [v+'_y' for v in valores]
  valores_diff = [v+'_diff' for v in valores]

  forcast_cohort_aux[col_data] = pd.to_datetime(forcast_cohort_aux[col_data], infer_datetime_format=True)

  agrupado = forcast_cohort_aux.loc[forcast_cohort_aux[col_data] >= primeira_data_baseline]
  agrupado = agrupado.groupby(chaves, as_index=False)[valores].sum()
  #print(agrupado.loc[agrupado['city_group'] == 'Belém'])


  agrupado_vol_coincident = agrupado.loc[agrupado[col_week_origin] != 'Coincident']
  agrupado_vol_coincident = agrupado_vol_coincident.groupby(aberturas, as_index=False)[valores].sum()

  agrupado_vol_cohort = agrupado.loc[agrupado[col_week_origin] != 'Não Convertido']

  baseline = pd.merge(agrupado_vol_cohort,agrupado_vol_coincident,how='left',on=aberturas)

  baseline[valores] = baseline[valores_x].values / baseline[valores_y].values

  baseline = baseline.drop(columns=valores_x+valores_y)




  # Vamos criar um baseline completo combinando todas as possibilidades de aberturas e Week Origins:
  #-------------------------------------------------------------------------------------------------

  '''
  # Geramos uma lista contendo listas das aberturas únicas e week origin's únicas:
  list_of_columns_list = []
  for abertura in aberturas:
    lista_aberturas = list(np.unique(agrupado[abertura].values))
    list_of_columns_list = list_of_columns_list+[lista_aberturas]
  lista_week_origin = list(range(int(max_origin)+1))
  lista_week_origin = [str(x) for x in lista_week_origin]
  lista_week_origin = lista_week_origin + ['Coincident']
  list_of_columns_list = list_of_columns_list+[lista_week_origin]

  # Com a lista de listas criadas, usamos a função 'product' para criar um DataFrame contendo
  # todas as combinações possíveis entre os ítens da lista de listas
  modelo = pd.DataFrame(list(product(*list_of_columns_list)), columns=aberturas+[col_week_origin])
  '''
  # Vamos em realidade basear o modelo nas combinações de aberturas existentes do ToF:
  modelo = agrupado.groupby(aberturas,as_index=False)[valores].sum()
  modelo = modelo[aberturas]
  lista_week_origin = list(range(int(max_origin)+1))
  lista_week_origin = [str(x) for x in lista_week_origin]
  lista_week_origin = lista_week_origin + ['Coincident']
  modelo[col_week_origin] = lista_week_origin[0]
  aux=modelo.copy()
  lista_modelo = [modelo]
  for week_origin in lista_week_origin[1:]:
    aux[col_week_origin] = week_origin
    lista_modelo = lista_modelo + [aux.copy()]
  modelo = pd.concat(lista_modelo)


  # A base final vai ser a união do modelo com a base final, com zeros onde não temos dados:
  baseline = pd.merge(modelo,baseline,how='left',on=aberturas+[col_week_origin])
  baseline = baseline.fillna(0)


  '''
  # Substituir Baseline Zerado
  baseline[col_week_origin] = baseline[col_week_origin].astype(str)
  cohort_aberta = baseline.groupby(aberturas, as_index=False)[valores].sum()

  merged = pd.merge(baseline,cohort_aberta,how='left',on=aberturas)
  merged = merged.fillna(0)



  # Removemos as aberturas que não possuem valor de nenhuma cohort aberta:
  merged['Soma'] = merged[valores_y].sum(axis=1)
  merged = merged.loc[merged['Soma'] != 0]
  merged = merged.drop(columns=['Soma'])

  for etapa in valores:
    merged.loc[(merged[etapa+'_y'] == 0) & (merged[col_week_origin] == '0'),[etapa+'_x']] = sub_aberta*sub_share
    merged.loc[(merged[etapa+'_y'] == 0) & (merged[col_week_origin] != '0'),[etapa+'_x']] = (sub_aberta*(1-sub_share))/float(max_origin)

  merged = merged.drop(columns=valores_y)
  merged = merged.rename(columns=dict(zip(valores_x,valores)))
  baseline = merged
  '''

  # Adicionaos a conversão Coincident nos casos onde só existe a max_origin:
  valores_max_origin = baseline.loc[baseline[col_week_origin] == str(max_origin),aberturas+valores]
  merged = pd.merge(baseline,valores_max_origin,how='left',on=aberturas)

  for etapa in valores:
    merged.loc[(merged[etapa+'_x'] == 0) & (merged[col_week_origin] == 'Coincident'),[etapa+'_x']] = merged.loc[(merged[etapa+'_x'] == 0) & (merged[col_week_origin] == 'Coincident'),[etapa+'_y']].values

  merged = merged.drop(columns=valores_y)
  merged = merged.rename(columns=dict(zip(valores_x,valores)))
  baseline = merged




  # Remover eventuais NaN's:
  baseline = baseline.fillna(0)

  # Nos casos onde existe baseline da última cohort mas não existe Coincident, substituir pela última:
  for val in valores:
    baseline_coincident = baseline.loc[(baseline[col_week_origin] == 'Coincident') & (baseline[val] == 0)]
    baseline_ultima_cohort = baseline.loc[(baseline[col_week_origin] == max_origin)]
    #print(baseline_coincident.loc[(baseline_coincident['city_group'] == 'Belém') & (baseline_coincident['mkt_channel'] == 'Other')])
    baseline_coincident_merged = pd.merge(baseline_coincident,baseline_ultima_cohort,how='left',on=aberturas)
    baseline_coincident_merged.loc[baseline_coincident_merged[val+'_y'] != 0][[val+'_x']] == baseline_coincident_merged.loc[baseline_coincident_merged[val+'_y'] != 0,[val+'_y']].values
    baseline_coincident_merged = baseline_coincident_merged[aberturas+[val+'_x']]
    baseline_coincident_merged[col_week_origin] = 'Coincident'
    if len(baseline_coincident_merged) > 0:
      baseline = pd.merge(baseline,baseline_coincident_merged,how='left',on=aberturas+[col_week_origin])
      baseline = baseline.fillna(0)
      baseline.loc[(baseline[col_week_origin] == 'Coincident') & (baseline[val] == 0)][[val]] = baseline.loc[(baseline[col_week_origin] == 'Coincident') & (baseline[val] == 0),[val+'_x']].values
      baseline = baseline.drop(columns=[val+'_x'])

  # Remover valores negativos:
  for val in valores:
    baseline.loc[(baseline[val] < 0) & (baseline[col_week_origin] != 'Coincident'),val] = 0.00001


  # Substituir cohorts abertas que ficaram zeradas pela médias das cohorts nas outras aberturas:
  for abertura in aberturas:

    aberturas_media = list(set(aberturas) - set([abertura]))

    baseline_sem_coincident = baseline.loc[baseline[col_week_origin] != 'Coincident']
    baseline_somente_coincident = baseline.loc[baseline[col_week_origin] == 'Coincident']

    baseline_cohort_aberta = baseline_sem_coincident.groupby(aberturas, as_index = False)[valores].sum()
    baseline_cohort_aberta_zerada = baseline_cohort_aberta.copy()

    for val in valores:
      baseline_cohort_aberta_zerada.loc[baseline_cohort_aberta_zerada[val] <= 0,val] = -1

    flag_zeradas = [x+'_flag_zerada' for x in valores]
    baseline_cohort_aberta_zerada = baseline_cohort_aberta_zerada.rename(columns=dict(zip(valores, flag_zeradas)))
    baseline_cohort_aberta = pd.merge(baseline_cohort_aberta,baseline_cohort_aberta_zerada,how='left',on=aberturas)

    baseline = pd.merge(baseline,baseline_cohort_aberta_zerada,how='left',on=aberturas)
    baseline = baseline.fillna(0)

    for val in valores:
      baseline_media = baseline.loc[baseline[val+'_flag_zerada'] != -1]
      baseline_media = baseline_media.groupby(aberturas_media+[col_week_origin], as_index=False)[[val]].mean()
      baseline_media = baseline_media.rename(columns={val:val+'_media'})
      baseline_media = baseline_media[aberturas_media+[col_week_origin,val+'_media']]

      baseline = pd.merge(baseline,baseline_media,how='left',on=aberturas_media+[col_week_origin])
      baseline = baseline.fillna(0)
      if len(baseline.loc[baseline[val+'_flag_zerada'] == -1,val]) > 0:
        #print(baseline.loc[(baseline[val+'_flag_zerada'] == -1) & (baseline['city_group'] == 'Belém')])
        baseline.loc[baseline[val+'_flag_zerada'] == -1,val] = baseline.loc[baseline[val+'_flag_zerada'] == -1,val+'_media'].values

    baseline = baseline[aberturas+[col_week_origin]+valores]


  baseline.replace([np.inf, -np.inf], 0, inplace=True)

  # Substituir cohort W0 zeradas:
  for val in valores:
    baseline.loc[(baseline[col_week_origin] == '0') & (baseline[val] == 0),val] = sub_aberta#*sub_share

  # Redistribuindo cohorts que passaram de 100% e as que ficaram zeradas:
  baseline_sem_coincident = baseline.loc[baseline[col_week_origin] != 'Coincident']
  baseline_somente_coincident = baseline.loc[baseline[col_week_origin] == 'Coincident']
  baseline_cohort_aberta = baseline_sem_coincident.groupby(aberturas, as_index = False)[valores].sum()
  baseline_share = pd.merge(baseline_sem_coincident,baseline_cohort_aberta,how='left',on=aberturas)
  shares = [x+'_share' for x in valores]
  baseline_share[shares] = baseline_share[valores_x].values / baseline_share[valores_y].values
  baseline_share = baseline_share.drop(columns=valores_x+valores_y)
  baseline_cohort_aberta_corrigida = baseline_cohort_aberta.copy()
  for val in valores:
    baseline_cohort_aberta_corrigida.loc[baseline_cohort_aberta_corrigida[val] > 1,val] = 1
    baseline_cohort_aberta_corrigida.loc[baseline_cohort_aberta_corrigida[val] == 0,val] = sub_aberta
  novo_baseline = pd.merge(baseline_share,baseline_cohort_aberta_corrigida,how='left',on=aberturas)
  novo_baseline[valores_x] = novo_baseline[shares].values * novo_baseline[valores]
  novo_baseline[valores] = novo_baseline[valores_x].values
  novo_baseline = novo_baseline.drop(columns = valores_x+shares)
  baseline = pd.concat([novo_baseline,baseline_somente_coincident])


  baseline.replace([np.inf, -np.inf], 0, inplace=True)



  # Substituir cohort abertas que passaram de 100%:
  ratios = [x+'_ratio' for x in valores]
  valores_x = [x+'_x' for x in valores]
  valores_y = [x+'_y' for x in valores]

  baseline_sem_coincident = baseline.loc[baseline[col_week_origin] != 'Coincident']
  baseline_somente_coincident = baseline.loc[baseline[col_week_origin] == 'Coincident']
  baseline_cohort_aberta = baseline_sem_coincident.groupby(aberturas, as_index = False)[valores].sum()
  baseline_cohort_aberta_corrigida = baseline_cohort_aberta.copy()

  for val in valores:
    baseline_cohort_aberta_corrigida.loc[baseline_cohort_aberta_corrigida[val] > 1,[val]] = 1

  baseline_cohort_aberta_ratio = pd.merge(baseline_cohort_aberta,baseline_cohort_aberta_corrigida,how='left',on=aberturas)
  baseline_cohort_aberta_ratio[ratios] = baseline_cohort_aberta_ratio[valores_y].values / baseline_cohort_aberta_ratio[valores_x].values
  baseline_cohort_aberta_ratio = baseline_cohort_aberta_ratio[aberturas+ratios]

  baseline_sem_coincident = pd.merge(baseline_sem_coincident,baseline_cohort_aberta_ratio,how='left',on=aberturas)
  baseline_sem_coincident[valores] = baseline_sem_coincident[valores].values * baseline_sem_coincident[ratios].values
  baseline_sem_coincident = baseline_sem_coincident[aberturas+[col_week_origin]+valores]

  baseline = pd.concat([baseline_sem_coincident,baseline_somente_coincident])

  baseline.replace([np.inf, -np.inf], 0, inplace=True)

  # Removendo etapas do funil que não possuem conversão
  etapas_sem_conversao = [x for x in valores if '2' not in x]
  baseline = baseline.drop(columns=etapas_sem_conversao)

  return baseline,inputs_substituicao_df


