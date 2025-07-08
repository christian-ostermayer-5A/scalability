#@title Def recalcula_share_diario

# Importando bibliotecas necessárias
from planning_daily_partitioning import *
from rounding_tool import *

def recalcula_share_diario(ToF_diarizado, topos_de_funil):
  '''
  Função auxiliar que fornece novos weekdayshares a partir da base diarizada já com
  os impactos dos inputs de feriados. Caso usássemos o weekdayshare normal para fazer o cálculo
  que pondera os dias das semanas incompletas (semanas que dividem plannings) teríamos
  valores semanais erroaneamente estimados, dado que negligenciaríamos a influência
  de inputs.
  '''
  aberturas_groupby = list(ToF_diarizado.columns)
  aberturas_groupby = list(set(ToF_diarizado)-set(topos_de_funil))

  aberturas_groupby.remove('dia da semana')
  aberturas_groupby.remove('data')
  aberturas_groupby.remove('mês')
  aberturas_groupby.remove('ano')

  # Soma semanal dos valores diários de ToF com inputs de feriados
  aux = ToF_diarizado.groupby(aberturas_groupby, as_index=False)[topos_de_funil].sum()
  
  for topo in topos_de_funil:
    aux = aux.rename(columns={topo: f'soma_semana_{topo}'})
  
  output = pd.merge(ToF_diarizado,aux,how='left',on=aberturas_groupby)
  
  for topo in topos_de_funil:
    output[f'share-{topo}'] = output[topo]/output[f'soma_semana_{topo}']
    output[f'share-{topo}'] = output[f'share-{topo}'].astype('float')
  
  output = output.loc[:,~output.columns.str.contains('soma_semana_', case=False)]
  output = output.drop(columns=topos_de_funil)
  output = output.drop(columns=['ano','mês','data'])
  return output


#@title pondera_semanas_incompletas

#@title pondera_semanas_incompletas

def pondera_semanas_incompletas(df_topos_merged, topos_de_funil, df_share_diario_recalculado):
  '''
  Função usada para calcular os volumes diários nos dias das semanas que dividem
  plannings que estão fora do planning calculado. Por exemplo, Q1 de 2022 começa na 
  semana do dia 27/12/2021. Nesta semana os dias 1 e 2 de Janeiro estão em Q1, mas os
  dias 27, 28, 29, 30 e 31 de Dezembro estão em Q4. Para não ficar com volumes mais
  baixos nessas semanas fazemos uma regra de 3 para encontrar os volumes dos dias 
  27, 28, 29, 30 e 31. Uma observação importante é que o mesmo cálculo é realizado
  na última semana do plannig.
   '''
  idx_pre_dados = list(df_share_diario_recalculado.columns).index(f'share-{topos_de_funil[0]}')
  aberturas_share_recalculado = list(df_share_diario_recalculado.columns[:idx_pre_dados])

  primeira_semana = df_topos_merged['semana'].min()
  ultima_semana = df_topos_merged['semana'].max()

  # Separamos o dataframe de ToF em 2. Um com a primeira e última semanas do planning
  # e outro com todas as outras semanas
  df_topos_merged_nan = df_topos_merged[df_topos_merged['semana'].isin([primeira_semana]+[ultima_semana])]
  df_topos_merged = df_topos_merged[~df_topos_merged['semana'].isin([primeira_semana]+[ultima_semana])]

  df_topos_merged_nan = pd.merge(df_topos_merged_nan,df_share_diario_recalculado,how='left', on=aberturas_share_recalculado)
  

  aberturas_share_recalculado.remove("dia da semana")
  colunas_soma = ['share-' + s for s in topos_de_funil]
  colunas_soma.extend(topos_de_funil)
  
  # Criamos uma cópia para calcular o volume total semanal teórico
  aux = df_topos_merged_nan.copy()

  for topo in topos_de_funil:
    aux = aux[aux[topo].notna()]
    
  aux = aux.groupby(aberturas_share_recalculado, as_index=False)[colunas_soma].sum()
  
  aberturas_share_recalculado.append("dia da semana")

  for topo in topos_de_funil:
    aux = aux.rename(columns={topo:f"somaparcial_{topo}"})
    aux[f'volumetotalsemana-{topo}'] = aux[f"somaparcial_{topo}"]/aux[f'share-{topo}']

  # Eliminamos colunas auxiliares
  aux = aux.loc[:,~aux.columns.str.contains('somaparcial_', case=False)]
  aux = aux.loc[:,~aux.columns.str.contains('share', case=False)]

  aberturas_share_recalculado.remove("dia da semana")
  df_topos_merged_nan = pd.merge(df_topos_merged_nan,aux,how='left',on=aberturas_share_recalculado)
  aberturas_share_recalculado.append("dia da semana")

  # Calculamos o volume teórico somente nas células 'NaN'
  for topo in topos_de_funil:
    df_topos_merged_nan.loc[(df_topos_merged_nan[topo].isna()), [topo]] = df_topos_merged_nan[f"volumetotalsemana-{topo}"]*df_topos_merged_nan[f"share-{topo}"]

  # Juntamos novamente os dois dataframes
  df_topos_merged = df_topos_merged.append(df_topos_merged_nan)
  
  # Eliminamos colunas auxiliares
  df_topos_merged = df_topos_merged.loc[:,~df_topos_merged.columns.str.contains('somaparcial_', case=False)]
  df_topos_merged = df_topos_merged.loc[:,~df_topos_merged.columns.str.contains('volumetotalsemana-', case=False)]
  df_topos_merged = df_topos_merged.loc[:,~df_topos_merged.columns.str.contains('share-', case=False)]

  for topo in topos_de_funil:
    df_topos_merged[topo] = df_topos_merged[topo].fillna(0)
  
  df_topos_merged['data'] = df_topos_merged['data'].dt.strftime('%m/%d/%Y')


  return df_topos_merged

#@title quebra_diaria_ToF_2

def quebra_diaria_ToF_2(df_ToF_semanal,     # Dataframe com os volumes semanais coincident
                      df_share_diario,      # Dataframe com os dados de share diário
                      df_feriados,          # Dataframe com a lista de feriados
                      df_impactos_feriados, # Dataframe com os impactos de feriado
                      df_share_cidades,     # Dataframe com o share das cidades dentro das regiões
                      df_ToF_mensal,        # Dataframe com os dados mensais de ToF
                      df_base_cohort,       # Dataframe com os dados de baseline cohort 
                      nome_coluna_week_origin,
                      coluna_de_semanas,
                      topos,
                      aberturas_das_bases,
                       round_output = False):    

  if len(df_ToF_mensal)>0:

    # definimos os cabeçalhos das outras bases
    cb_share_diario = list(df_share_diario.columns)
    cb_impactos_feriados = list(df_impactos_feriados.columns)
    cb_base_cohort = list(df_base_cohort.columns)

    # definimos onde começam os dados de todas as bases com base na posição da coluna "Week Origin"
    # na base cohort
    posi_share_diario = cb_share_diario.index("dia da semana")+1
    posi_impactos_feriados = cb_impactos_feriados.index("dia da semana")+1


    # Definimos os topos de funil
    topos_de_funil = topos
    aberturas = aberturas_das_bases+['building block tof']

    # Definimos novos cabeçalhos somente com o topo de funil
    n_cb_share_diario = cb_share_diario[:posi_share_diario]+topos_de_funil
    n_cb_impactos_feriados = cb_impactos_feriados[:posi_impactos_feriados]+topos_de_funil
    

    # Redefinimos as bases somente com o ToF:
    n_share_diario_df = df_share_diario[n_cb_share_diario] #np.append(np.array([n_cb_share_diario]),pd.DataFrame(share_diario[1:,:], columns = cb_share_diario)[n_cb_share_diario].to_numpy(),0)
    n_impactos_feriados_df = df_impactos_feriados[n_cb_impactos_feriados] #np.append(np.array([n_cb_impactos_feriados]),pd.DataFrame(impactos_feriados[1:,:], columns = cb_impactos_feriados)[n_cb_impactos_feriados].to_numpy(),0)

    if len(df_share_cidades) > 0:
      cb_share_cidades = list(df_share_cidades.columns)
      posi_share_cidades = cb_share_cidades.index("cidade")+1 
      n_cb_share_cidades = cb_share_cidades[:posi_share_cidades]+topos_de_funil
      n_share_cidades_df = df_share_cidades[n_cb_share_cidades] #np.append(np.array([n_cb_share_cidades]),pd.DataFrame(share_cidades[1:,:], columns = cb_share_cidades)[n_cb_share_cidades].to_numpy(),0)
    else:
      n_share_cidades_df = df_share_cidades

    # Diarizamos o ToF semanal
    ToF_diarizado = quebra_diaria(volumes_semanais = df_ToF_semanal,  # DataFrame com os volumes semanais coincident
                                  coluna_de_semanas = coluna_de_semanas,
                                  aberturas_das_bases = aberturas_das_bases,
                                  share_diario = n_share_diario_df,      # DataFrame com os dados de share diário
                                  feriados = df_feriados,          # DataFrame com a lista de feriados
                                  impactos_feriados = n_impactos_feriados_df, # DataFrame com os impactos de feriado
                                  share_cidades = n_share_cidades_df,     # DataFrame com o share das cidades dentro das regiões
                                  topo_de_funil = topos_de_funil,  
                                  base_diaria_ToF = []) 
    
    ToF_diarizado['semana'] = pd.to_datetime(ToF_diarizado['semana'], infer_datetime_format=True)
    
    # Calculamos um nova share diário baseado na diarização inicial, que aplica impactos de feriados.
    share_diario_recalculado = recalcula_share_diario(ToF_diarizado,topos_de_funil)


    # Calculamos um ToF mensal baseado no ToF diarizado
    ToF_mensal_atual = ToF_diarizado.groupby(['ano','mês']+aberturas, as_index=False)[topos_de_funil].sum()

    for topo in topos_de_funil:
      ToF_mensal_atual = ToF_mensal_atual.rename(columns={topo: f'{topo}_soma_mes'})

    # Calculamos o share diário dentro do mês, que chamaremos de "fator"
    ToF_diarizado = pd.merge(ToF_diarizado,ToF_mensal_atual,how='left',on=['ano','mês']+aberturas)
    for topo in topos_de_funil:
      ToF_diarizado[f'fator_{topo}'] = ToF_diarizado[topo]/ToF_diarizado[f'{topo}_soma_mes']
      ToF_diarizado[f'fator_{topo}'] = ToF_diarizado[f'fator_{topo}'].fillna(0)
      ToF_diarizado = ToF_diarizado.rename(columns={topo: f'{topo}_diarizado_semanal'})
    

    # Formatamos os valores das colunas de ano e mês
    #df_ToF_mensal = df_ToF_mensal.drop(columns=['data'])
    ToF_diarizado[['ano','mês']] = ToF_diarizado[['ano','mês']].astype(int)
    df_ToF_mensal[['ano','mês']] = df_ToF_mensal[['ano','mês']].astype(int)

    ToF_diarizado = pd.merge(df_ToF_mensal,ToF_diarizado,how='outer',on=['ano','mês']+aberturas)
    ToF_diarizado['data'] = pd.to_datetime(ToF_diarizado['data'], errors='coerce')
    ToF_diarizado['dia da semana'] = pd.Series(ToF_diarizado.data).dt.day_name()

    # Aplicamos o share diário mensal na base de ToF mensal original
    for topo in topos_de_funil:
      ToF_diarizado[topo] = ToF_diarizado[topo].astype('float')*ToF_diarizado[f'fator_{topo}']
    
    # Com base nos volumes diários recalculados de acordo com os volumes mensais, calculamos os 
    # volumes diários dos dias que estão fora dos meses oficiais com base no volume incompleto das semanas
    # que iniciam e terminam o mês e extrapolamos esse volume para os outros dias com base no share diário
    # recalculado.
    ToF_diarizado = pondera_semanas_incompletas(ToF_diarizado, topos_de_funil, share_diario_recalculado)

    # Check e remoção de valores negativos
    #-------------------------------------------------------------------------------------------------
    qtd_valores_negativos = ToF_diarizado[topos_de_funil].where(ToF_diarizado[topos_de_funil] < 0).count().sum()
    if qtd_valores_negativos > 0:
      qtd_valores_positivos = ToF_diarizado[topos_de_funil].where(ToF_diarizado[topos_de_funil] >= 0).count().sum()
      datas_negativas = ToF_diarizado[(ToF_diarizado[topos_de_funil].values < 0)]['data'].unique()

      print("_______________________________________")
      print("Atenção: Foram encontrados",qtd_valores_negativos,"valores negativos na base diária nas seguintes posições:")
      print(ToF_diarizado[(ToF_diarizado[topos_de_funil].values < 0)][['data']+aberturas])
      print('Valores negativos substituídos por zero: o total mensal não irá bater')

      # Removendo valores negativos
      #ToF_diarizado[topos_de_funil] = ToF_diarizado[topos_de_funil].clip(lower=0)
    

    # Redefinindo ToF semanal após a redistribuição
    n_ToF_semanal = ToF_diarizado.groupby(['semana']+aberturas, as_index=False)[topos_de_funil].sum().rename(columns={'semana':coluna_de_semanas})
    n_ToF_semanal = n_ToF_semanal.fillna(0)

    ToF_diarizado['data'] = pd.to_datetime(ToF_diarizado['data'], infer_datetime_format=True)

    ToF_diarizado = ToF_diarizado[['data','semana','mês','ano','dia da semana']+aberturas+topos_de_funil]

    # Arredondamos os valores finais:
    if round_output:
      n_ToF_semanal = rounding_tool(df = n_ToF_semanal,
                                    aberturas = ['week start']+aberturas,
                                    col_valores = topos_de_funil,
                                    ordem_hirarquica = ['data']+aberturas_das_bases)
  
      ToF_diarizado = rounding_tool(df = ToF_diarizado,
                                    aberturas = ['data','semana','mês','ano','dia da semana']+aberturas,
                                    col_valores = topos_de_funil,
                                    ordem_hirarquica = ['data']+aberturas_das_bases)

    print("_______________________________________")
    print("Correção dos valores mensais finalizada")

    return ToF_diarizado,n_ToF_semanal

  
  else:
    return [],df_ToF_semanal
