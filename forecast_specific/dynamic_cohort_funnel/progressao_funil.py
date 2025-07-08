#@title Def progressao_funil
import numpy as np
import pandas as pd
# Multiplica conv x vol, gera vol_coh e soma o volume da nova etapa.

# Essa função literalmente faz a conta de progressão do funil. Com a base única, multiplicamos
# as colunas contendo ToF pelas colunas correspondentes de conversão, obtendo o volume da cohort.
# Somando esse volume obtemos o volume da etapa seguinte.

# Além disso, essa função também já faz o split da etapa em fluxos distintos, caso ele exista

def progressao_funil(base_merged, # DataFrame total gerado pela função auxiliar "def base_geral"
                     etapa_coh,   # string da etapa cohort que está sendo calculada
                     etapas_split,# lista com as etapas split ou vetor contendo a informação que não existe split gerado pela função auxiliar "def base_geral"
                     chaves_coh,  # lista com as aberturas cohort gerado pela função auxiliar "def base_geral"
                     max_origin): # inteiro representando a cohort máxima, gerado pela função auxiliar "def base_geral"
  #print(base_merged.loc[base_merged['week_start'] == '2022-07-04'])
  # com base na etapa de conversão, criamos o nome da coluna que irá conter o volume cohort da conversão
  etapa_coh_vol = "coh_vol_"+etapa_coh
  # definimos qual é a etapa de volume anterior e posterior da conversão também com base na etapa de conversão
  etapa_vol = etapa_coh.split("2")[0]
  proxima_etapa_vol = etapa_coh.split("2")[1]

  '''
  # as chaves usadas para agredar o volume das cohorts no volume coincident.
  chaves = list(base_merged.columns.values)
  chaves = chaves[:chaves.index('Week Origin')]
  '''

  chaves = chaves_coh.copy()

  # Definimos o nome da coluna com as datas:
  col_data = chaves[0]

  # o volume coincident deve ser agregado na "diagonal", por isso substituímos a chave da data pela
  # chave da data deslocada "Shifted"
  chaves[0] = 'Shifted'

  # Calculamos o volume da cohort multiplicando a coluna da conversão cohort pela coluna do volume da
  # etapa anterior

  base_merged[etapa_coh_vol] = base_merged[etapa_coh].astype('float')*\
                               base_merged[etapa_vol].astype('float')

  #print(base_merged.loc[base_merged['Week Origin'] == 'Coincident',['week_start',etapa_coh,etapa_vol,etapa_coh_vol]])

  # definimos a base que vai somar o volume coincident excluíndo a última cohort, pois para recompor
  # a coincident utilizamos no lugar da maior cohort a cohort de ajuste
  if max_origin > 0:                     
    agg_vol_novo = base_merged.loc[base_merged['Week Origin'] != str(max_origin)]
  else:
    agg_vol_novo = base_merged.copy()
    
  # agrupamos e somamos a coluna de vol cohort para obter o volume coincident da próxima etapa
  agg_vol_novo = agg_vol_novo.groupby(chaves, as_index=False)[[etapa_coh_vol]].sum()

  # Mudamos o nome da coluna das datas, pois estas são as datas finais e não consideramos mais que
  # estão deslocadas. Também redefinimos o nome da coluna com o volume cohort para o nome
  # provisório da próxima etapa do funil. Utilizamos um nome provisório ("P_+etapa") para não
  # confundir com o nome da etapa de ToF, caso ela já exista na base, como seria o caso de funis com
  # dois ToF's, como o caso de demanda com VB e OS
  agg_vol_novo = agg_vol_novo.rename(columns = {'Shifted': col_data, etapa_coh_vol: "p_"+proxima_etapa_vol})

  # definimos uma lista com o cabeçalho da base agregada de volume coincident
  cb_agg = list(agg_vol_novo.columns.values)[:-1]

  # unimos a base geral com a base de volume coincident
  base_merged = pd.merge(base_merged,agg_vol_novo,how='left',on=cb_agg)


  #print(base_merged.loc[(base_merged['Week Origin'] == 'Coincident'),['week_start',etapa_coh,etapa_vol,etapa_coh_vol,"p_"+proxima_etapa_vol]])
  #print(base_merged.loc[base_merged['week_start'] == '2021-11-15',['Week Origin',etapa_coh,etapa_vol,etapa_coh_vol,"p_"+proxima_etapa_vol]])

  # verificamos se a etapa de volume coincident recém calculada já existia na base geral:
  if proxima_etapa_vol in base_merged.columns:
    # caso já existisse, somamos o volume coincident calculado ao ToF já existente
    base_merged[proxima_etapa_vol] = base_merged[proxima_etapa_vol].astype('float').add(base_merged["p_"+proxima_etapa_vol].astype('float'), fill_value=0)
    # removemos a coluna com o volume calculado (pois já foi somado ao ToF)
    base_merged.drop(["p_"+proxima_etapa_vol],axis=1, inplace=True)

  # Caso contrário, mudamos o nome provisório para o nome da etapa definitivo
  else:
    base_merged = base_merged.rename(columns = {"p_"+proxima_etapa_vol: proxima_etapa_vol})

  # Aqui verificamos se existe um split de fluxos no volume da etapa recém calculado
  if etapas_split[0][0] != "sem split":
    # caso exista, utilizamos uma função auxiliar que já calcula o volume resultante do split de fluxos
    vol_split = split_etapa(base_merged,proxima_etapa_vol,etapas_split,chaves_coh) # @função_auxiliar
    if vol_split[0] != "sem split":
      print('     --> Realizado split de volume da etapa ',proxima_etapa_vol)
      base_merged[proxima_etapa_vol] = vol_split
  

  return base_merged # <-- retorna o mesmo DataFrame total, mas acrescido do volume da cohort e o volume
                     #     coincident da etapa seguinte
#---------------------------------------------------------------------------------------------------

