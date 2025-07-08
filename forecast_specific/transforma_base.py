#@title Def transforma_base
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import calendar
from week_of_month import week_of_month
from adiciona_feriados import adiciona_feriados

#@title Def adiciona_feriados


                            
'''
Recebe uma base com as seguintes colunas:

df_conversoes = ['week_start' 'city_group' 'mkt_channel' 'lead' 'conversion' 'OP2Q' 'Q2Opp' 'Opp2FL' 'FL']

Transforma o histórico numa base com as seguintes colunas:

out = ['week_start', 'city_group', 'mkt_channel', 'lead', 'Etapa',
      'Volume',
      'Volume Aberta', '0', '1', '2', '3', '4', '5', 'Coincident',
      '%__0','%__5', '%__1', '%__2', '%__3', '%__4', '%__Coincident','%__Volume Aberta',
      's__0', 's__1', 's__2', 's__3', 's__4', 's__5','s__Coincident',
      'ordem_semana']
'''

def transforma_base(df_conversoes,
                    df_exogenous,
                    df_feriados,
                    df_targets,
                    data_end_forecast,
                    ultima_data_hist):
                          

  # Vamos fazer algumas definições iniciais sobre a base
  cb_conversoes = list(df_conversoes.columns.values) # Colunas no cabeçalho
  chaves = cb_conversoes[:cb_conversoes.index('conversion')] # Colunas que contém as aberturas da base
  etapas = cb_conversoes[cb_conversoes.index('conversion')+1:] # Etapas do funil na base
  conversoes = list(pd.unique(df_conversoes['conversion'].values)) # Quais conversões cohort existem na base
  conversoes.remove('Não Convertido') # Vamos remover da lista de conversões o "Não Convertido"
  conversoes_int = conversoes.copy()
  conversoes_int.remove('Coincident')
  max_origin = np.max(np.array(conversoes_int,dtype=int))
  conversoes = list(map(str, list(range(max_origin+1))))
  conversoes = conversoes+['Coincident']

  # Definir automaticamente a coluna de região:
  # @ aqui melhorar
  if "city_group" not in list(df_exogenous.columns.values) and len(df_exogenous) > 0:
    col_regiao = "city_group_tier"
  else:
    col_regiao = "city_group"


  # Transformamos a coluna de conversões em texto (não lembro pq precisei fazer isso)
  df_conversoes = df_conversoes.replace(r'^\s*$', '0', regex=True)

  # Vamos transformar os valores da base em float
  df_conversoes[etapas] = df_conversoes[etapas].astype(float)
  #df_targets[etapas] = df_targets[etapas].astype(float)

  # Transformar a coluna de datas em DateTime
  df_conversoes['week_start'] = pd.to_datetime(df_conversoes['week_start'])
  #df_targets['week_start'] = pd.to_datetime(df_targets['week_start'])
  if len(df_exogenous) != 0:
    df_exogenous['week_start'] = pd.to_datetime(df_exogenous['week_start'])

  ultima_data_hist = pd.to_datetime(ultima_data_hist, infer_datetime_format=True)

  # Vamos definir a última data do histórico
  df_conversoes = df_conversoes.loc[df_conversoes['week_start'] <= ultima_data_hist]


  # Vamos criar uma base com a repetição de todas as aberturas nas menores e maiores datas
  # Esse vai ser nosso modelo para mergear com o histórico e variáveis exógenas, de forma que
  # quando uma abertura não existir em alguma dessas bases a partir de alguma data, conseguimos incluir
  # um valor zerado no lugar e não prejudicar as equações do forecast com buracos nas bases:
  chaves_modelo = chaves+['conversion']
  '''
  list_of_columns_list = chaves_modelo.copy()
  posi = 0
  for coluna in chaves_modelo:
    # Caso seja a primeira coluna, vamos preencher a base modelo com todas as datas do histórico
    if coluna == chaves_modelo[0]:
      start = np.min(df_conversoes[coluna].values)
      end = data_end_forecast
      lista = list(pd.date_range(start=start, end=end, freq='7D'))
    else:
      lista = list(pd.unique(df_conversoes[coluna].values))

    list_of_columns_list[posi] = lista

    posi = posi+1

  df_base_modelo = pd.DataFrame(list(product(*list_of_columns_list)), columns=chaves_modelo)
  '''
  # Lista com todas as datas do histórico:
  start = np.min(df_conversoes[chaves_modelo[0]].values)
  end = data_end_forecast
  lista_datas = list(pd.date_range(start=start, end=end, freq='7D'))

  # Gerar base com as combinações únicas de aberturas de toda base histórica:
  df_modelo_aberturas = df_conversoes.copy()
  df_modelo_aberturas['aux'] = 1
  df_modelo_aberturas = df_modelo_aberturas.groupby(chaves[1:], as_index=False)['aux'].sum()
  df_modelo_aberturas = df_modelo_aberturas[chaves[1:]]

  # Gerar base modelo repetindo aberturas para cada cohort:
  conversoes_completas = conversoes+['Não Convertido']
  df_base_modelo_c = df_modelo_aberturas.copy()
  df_base_modelo_c['conversion'] = conversoes_completas[0]
  lista_df_base_modelo_c = [df_base_modelo_c]
  for c in conversoes_completas[1:]:
    aux = df_modelo_aberturas.copy()
    aux['conversion'] = c
    lista_df_base_modelo_c = lista_df_base_modelo_c+[aux.copy()]
  df_base_modelo_c = pd.concat(lista_df_base_modelo_c)

  # Gerar base modelo repetindo aberturas para cada data do histórico:
  df_base_modelo = df_base_modelo_c.copy()
  df_base_modelo[chaves[0]] = lista_datas[0]
  lista_df_base_modelo = [df_base_modelo]
  for d in lista_datas[1:]:
    aux = df_base_modelo_c.copy()
    aux[chaves[0]] = d
    lista_df_base_modelo = lista_df_base_modelo + [aux.copy()]
  df_base_modelo = pd.concat(lista_df_base_modelo)

  # Reordenando as colunas:
  df_base_modelo = df_base_modelo[chaves_modelo]

  # Vamos selecionar as datas historicas da base de modelo para recri
  df_base_modelo_hist = df_base_modelo.loc[df_base_modelo['week_start'] <= np.max(df_conversoes['week_start'].values)]

  # Agora vamos redefinir a base de conversões históricas com base em todas as combinações de aberturas
  # e datas da base modelo:
  df_conversoes = pd.merge(df_base_modelo_hist,df_conversoes,how='left',on=chaves_modelo)
  df_conversoes = df_conversoes.fillna(0)

  # Fazemos o mesmo com as exógenas, se existirem
  if len(df_exogenous) != 0:
    df_base_modelo_exo = df_base_modelo.loc[df_base_modelo['conversion'] == '0',chaves]
    df_exogenous = pd.merge(df_base_modelo_exo,df_exogenous,how='left',on=chaves)
    df_exogenous = df_exogenous.fillna(0)
  else:
    df_exogenous = df_base_modelo.loc[df_base_modelo['conversion'] == '0',chaves]



  # Criamos uma base que irá conter somente o volume coincident de cada etapa por semana
  df_volumes = df_conversoes.loc[df_conversoes['conversion'] != 'Coincident'] # Não somamos a conversão de ajuste para obter o volume total
  df_volumes = df_volumes.groupby(chaves, as_index=False)[etapas].sum() # Agrupamos somando os volumes das conversões
  df_volumes = df_volumes.melt(id_vars=chaves,var_name='etapa', value_name='Volume') # Transformamos as etapas do funil numa única coluna, e os valores de volume numa única coluna de 'Volume'

  # Criamos uma base com o volume da cohort aberta (esse volume não é o coincident, é a soma dos volumes das cohorts fechadas)
  df_aberta = df_conversoes.loc[(df_conversoes['conversion'] != 'Coincident') & (df_conversoes['conversion'] != 'Não Convertido')] # Não somamos oq não foi convertido nem a conversão de ajuste
  df_aberta = df_aberta.groupby(chaves, as_index=False)[etapas].sum() # Agrupamos somando as conversões
  df_aberta = df_aberta.melt(id_vars=chaves,var_name='etapa', value_name='Volume Aberta') # Transformamos as etapas do funil numa única coluna, e os valores de volume numa única coluna de 'Volume Aberta'

  # Criamos uma base somente com os volumes das cohorts fechadas
  df_cohorts = df_conversoes.loc[df_conversoes['conversion'] != 'Não Convertido']
  df_cohorts = df_cohorts.melt(id_vars=chaves+['conversion'],var_name='etapa', value_name='Volume Cohort') # Transformamos as etapas do funil numa única coluna, e os valores de volume numa única coluna de 'Volume Cohort'
  df_cohorts = df_cohorts.groupby(chaves+['etapa','conversion'], as_index=False)['Volume Cohort'].sum()
  df_cohorts = df_cohorts.pivot(index=chaves+['etapa'],columns='conversion',values='Volume Cohort').reset_index(inplace=False).fillna(0) # Transformamos novamente para que tenhamos uma coluna com valores para cada conversão.
  # Vamos garantir que existem todas as colunas de conversões na base:
  for conv in conversoes:
    if str(conv) not in df_cohorts.columns.values:
      df_cohorts[str(conv)] = 0.0
  df_cohorts = df_cohorts[chaves+['etapa']+conversoes]

  # Mergeamos as 3 bases criadas para calcular as conversões em % e os shares que elas representam da cohort aberta
  df_merged = pd.merge(df_volumes,df_aberta,how='outer',on=chaves+['etapa'])
  df_merged = pd.merge(df_merged,df_cohorts,how='outer',on=chaves+['etapa'])

  # Calculamos as cohorts em %
  conversoes = conversoes+['Volume Aberta'] # Selecionamos as conversões que queremos calcular em termos de % (inclusive a cohort aberta)
  conversoes_p =  ["%__" + c for c in conversoes] # Criamos as colunas na base que irão conter os valores das conversões em termos de %
  df_merged[conversoes_p] = df_merged[conversoes].div(df_merged['Volume'], axis=0).fillna(0) # Dividimos os volumes das cohorts pelo volume coincident da semana

  # Calculamos os shares relativos das cohorts %
  conversoes_n = conversoes.copy() # Criamos uma lista somente com as conversões numéricas
  conversoes_n.remove('Volume Aberta')
  conversoes_n.remove('Coincident')
  conversoes_n = [int(i) for i in conversoes_n] # Transformamos em inteiro para conseguir ordenar
  conversoes_n = np.sort(conversoes_n) # Ordenamos as cohorts fechadas (vai ficar clara a importância disso mais tarde)

  conversoes_s = [str(i) for i in conversoes_n] # Trasformamos as conversões numéricas em string
  conversoes_s = conversoes_s+['Coincident'] # Adicionamos a conversão de ajuste
  conversoes_o = conversoes_s.copy() # Criamos o nome das colunas que irão conter o share das cohorts fechadas
  conversoes_s =  ["s__" + c for c in conversoes_s]

  # O share relativo de uma conversão é o share da conversão em relação à cohort aberta menos as cohorts anteriores:
  # share W0 = W0/Cohort Aberta, share W1 = W1/(Cohort Abreta - W0), share W2 = W2/(Cohort Abreta - W0 - W1)
  # Assim, podemos reescrever as fórmulas acima para os casos que não sejam share W0 e share W1:
  # s_W(n) = W(n) / [ W(n-1) * ((1 / s_W(n-1)) - 1) ]
  for c in range(len(conversoes_o)): # Aqui fica evidente o motivo de ordenarmos as cohorts. Calculamos da W0 até a W5+ e a de ajuste
    if conversoes_o[c] == '0': # No caso da W0, o share é somente W0 / Cohort Aberta(CA)
      df_merged[conversoes_s[c]] = df_merged["%__"+conversoes_o[c]] / df_merged['%__Volume Aberta']
    elif conversoes_o[c] == '1': # No caso da W1, o share é W1 / (CA - W0)
      df_merged[conversoes_s[c]] = df_merged["%__"+conversoes_o[c]] / (df_merged['%__Volume Aberta']-df_merged['%__0'])

    #elif conversoes_o[c] != 'Coincident' and conversoes_o[c] != str(max_origin):  # s_W(n) = W(n) / [ W(n-1) * ((1 / s_W(n-1)) - 1) ]
    elif conversoes_o[c] != 'Coincident':
      df_merged[conversoes_s[c]] = df_merged["%__"+conversoes_o[c]] / (df_merged["%__"+conversoes_o[c-1]]*(1/df_merged[conversoes_s[c-1]]-1))

    #elif conversoes_o[c] == str(max_origin): #A última conversão cohort é sempre 100% do que sobra
    #  df_merged[conversoes_s[c]] = 1

    else:
      df_merged[conversoes_s[c]] = 0
      #df_merged[conversoes_s[c]] = df_merged["%__"+conversoes_o[c]] / (df_merged["%__"+conversoes_o[c-2]]*(1/df_merged[conversoes_s[c-2]]-1))

  df_merged = df_merged.fillna(0)
  df_merged.replace([np.inf, -np.inf], 0, inplace=True)

  # Queremos eliminar os shares que ultrapassam 1000% (Investigar as situações que passa de 100% e entender como lidar com elas. Não tem )
  for s in conversoes_s:
    if s != 's__Coincident':
      df_merged.loc[df_merged[s] > 10, s] = 1
      df_merged.loc[df_merged[s] < -10, s] = 0
    else:
      df_merged.loc[df_merged[s] > 10, s] = 10
      df_merged.loc[df_merged[s] < -10, s] = -10


  '''
  # Vamos verificar onde o share da conversão de ajuste é zero. Isso significa que a conversão de ajuste
  # ficou maior do que a soma da cohort aberta. Nesses casos vamos igualar a conversão de ajuste à
  # última cohort e fazer o share = 100%:
  df_merged.loc[df_merged['s__Coincident'] == 0,['%__Coincident','s__Coincident']] = df_merged.loc[df_merged['s__Coincident'] == 0,['%__'+str(max_origin),'s__'+str(max_origin)]].values

  # Agora vamos ver se ainda restaram conversões de ajuste muito estranhas. Vamos remover todas
  # conversões de ajuste que forem maiores do que a cohort aberta inteira.
  df_merged['Comparação'] = df_merged['%__Volume Aberta'].astype(float).values - df_merged['%__Coincident'].astype(float).values
  df_merged.loc[df_merged['Comparação'] <= 0,['%__Coincident','s__Coincident']] = df_merged.loc[df_merged['Comparação'] <= 0,['%__'+str(max_origin),'s__'+str(max_origin)]].values
  df_merged = df_merged.drop(columns=['Comparação'])
  '''
  # Vamos adicionar as colunas exógenas de ordem de semana e feriado na bas exógena, que contém
  # todas as datas, inclusive as do histórico e projeção futura.
  df_exogenous['ordem_semana'] = df_exogenous['week_start'].apply(week_of_month)

  df_exogenous = adiciona_feriados(df = df_exogenous,
                                col_data = 'week_start',
                                col_regiao = col_regiao,
                                df_feriados = df_feriados)

  # Também vamos adicionar nas exógenas uma coluna de tempo numérico. Essa série vai servir para incluir
  # tendências gerais de crescimento ou queda temporal das séries endógenas:
  df_exogenous['Tempo Numérico'] = df_exogenous['week_start'].astype(int)/10**18

  # Vamos adicionar uma coluna com o ano (para ajudar a treinar sazonalidade)
  df_exogenous['Year'] = pd.DatetimeIndex(df_exogenous['week_start']).year

  # Vamos adicionar colunas com os meses (para ajudar a treinar sazonalidade)
  df_exogenous['Month'] = pd.DatetimeIndex(df_exogenous['week_start']).month.astype(int)
  meses = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
  im=1
  for m in meses:
    df_exogenous[m] = 0
    df_exogenous.loc[df_exogenous['Month'] == im,[m]] = 1
    im=im+1
  df_exogenous = df_exogenous.drop(columns = ['Month'])

  # Precisamos adicionar a coluna com a etapa na base de exógenos.
  # Vamos somente repetir a base um número de vezes igual ao número de etapas:
  df_exogenous['etapa'] = etapas[0]
  aux = df_exogenous.copy()
  lista_df_exogenous = [df_exogenous]
  for e in etapas[1:]:
    aux['etapa'] = e
    lista_df_exogenous = lista_df_exogenous+[aux.copy()]
  df_exogenous = pd.concat(lista_df_exogenous)


  # Agora, vamos criar uma base única com histórico e exógenas futuras:
  base_unica = pd.merge(df_merged,df_exogenous,how='outer',on=chaves+['etapa'])
  base_unica = base_unica.rename(columns={'etapa':'Etapa'})
  base_unica = base_unica.fillna(0)

  # Vamos remover as combinações de aberturas que não possuem nenhum valor:
  base_unica_max_vol = base_unica.groupby(chaves[1:],as_index=False)['Volume'].max()
  base_unica_max_vol = base_unica_max_vol.rename(columns={'Volume':'Vol_max'})
  base_unica = pd.merge(base_unica,base_unica_max_vol,how='left',on=chaves[1:])
  base_unica = base_unica.loc[base_unica['Vol_max'] > 0]
  base_unica = base_unica.drop(columns='Vol_max')

  conversoes_s = conversoes_s[:-1] # Não queremos mais projetar a conversão de ajuste

  return base_unica,max_origin,conversoes_s

