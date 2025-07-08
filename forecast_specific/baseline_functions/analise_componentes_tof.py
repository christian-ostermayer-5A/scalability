#@title Def analise_componentes_tof
import numpy as np
import pandas as pd

def analise_componentes_tof(df_forecast_cohort,
                                 df_forecast_diario,
                                 df_matriz_forecast,
                                 df_actual_diario,
                                 df_actual_cohort,
                                 max_origin,
                                 etapas_volume,
                                 topo,
                                 tipo_de_tof,
                                 tipo_de_projecao,
                                 col_data,
                                 aberturas,
                                 primeira_data_forecast,
                                  fit_intercept):

  pivot_tof_completo_share_semanal = pd.DataFrame()
  pivot_tof_completo_share_mensal = pd.DataFrame()
  pivot_tof_completo_share_semanal_wow = pd.DataFrame()
  output_tof_completo_share_semanal = pd.DataFrame()
  tof_share = pd.DataFrame()

  if len(df_matriz_forecast) > 0 and tipo_de_projecao != "Average":


    tof_completo_share_mensal = pd.DataFrame()

    primeiro_ano = primeira_data_forecast.year
    primeiro_mes = primeira_data_forecast.month
    if primeiro_mes == 12:
      segundo_mes = 1
      segundo_ano =  primeiro_ano+1
    else:
      segundo_mes = primeiro_mes+1
      segundo_ano = primeiro_ano


    # 1) Calcular uma base de share de exógena por abertura e por semana
    #-------------------------------------------------------------------------------------------------
    share_de_exogenas_semanal_soma = df_matriz_forecast.copy()
    share_de_exogenas_semanal_soma['Valor Absoluto'] = share_de_exogenas_semanal_soma['Valor'].abs()
    share_de_exogenas_semanal_soma = share_de_exogenas_semanal_soma.groupby([col_data,'Endógena','Etapa']+aberturas, as_index=False)['Valor Absoluto'].sum()



    # Calcular o share das exógenas em relação à soma total:
    share_de_exogenas_semanal = pd.merge(df_matriz_forecast,share_de_exogenas_semanal_soma,how='left',on=[col_data,'Endógena','Etapa']+aberturas)
    share_de_exogenas_semanal['Share'] = share_de_exogenas_semanal['Valor'].values / share_de_exogenas_semanal['Valor Absoluto'].values
    share_de_exogenas_semanal = share_de_exogenas_semanal.fillna(0)
    share_de_exogenas_semanal[col_data] = pd.to_datetime(share_de_exogenas_semanal[col_data], infer_datetime_format=True)
    #exogenas_puras = share_de_exogenas_semanal
    share_de_exogenas_semanal = share_de_exogenas_semanal.drop(columns=['Valor Absoluto'])

    # Se o modelo tiver coeficiente linear (intercept), a composição positiva e negativa dos componentes
    # das exógenas (slope) são relativos e precisam ser calculados de forma aproximada:
    if fit_intercept:

      share_de_exogenas_semanal = share_de_exogenas_semanal.drop(columns=['Valor'])

      # Calcular o share médio das exógenas em todas as semanas de forma que a parte positiva some 200%
      # e a parte negativa -100%
      share_de_exogenas_semanal['Positivos/Negativos'] = 1
      share_de_exogenas_semanal.loc[share_de_exogenas_semanal['Share'] < 0,['Positivos/Negativos']] = -1



      # Calcular a soma dos negativos e positivos
      share_de_exogenas_semanal_soma_positivos = share_de_exogenas_semanal.loc[share_de_exogenas_semanal['Positivos/Negativos'] > 0].groupby([col_data,'Positivos/Negativos','Endógena','Etapa']+aberturas, as_index=False)['Share'].sum()
      share_de_exogenas_semanal_soma_negativos = share_de_exogenas_semanal.loc[share_de_exogenas_semanal['Positivos/Negativos'] < 0].groupby([col_data,'Positivos/Negativos','Endógena','Etapa']+aberturas, as_index=False)['Share'].sum()

      share_de_exogenas_semanal_soma_total = pd.concat([share_de_exogenas_semanal_soma_positivos,share_de_exogenas_semanal_soma_negativos])
      share_de_exogenas_semanal_soma_total = share_de_exogenas_semanal_soma_total.rename(columns={'Share':'Soma Share'})

      # Calcular a proporção média entre positivos e negativos:
      proporcao = pd.merge(share_de_exogenas_semanal_soma_positivos,share_de_exogenas_semanal_soma_negativos,how='left',on=[col_data,'Endógena','Etapa']+aberturas)
      proporcao = proporcao.fillna(0)
      proporcao['Proporcao'] = proporcao['Share_x'].values / proporcao['Share_y'].values

      proporcao = proporcao[[col_data,'Endógena','Etapa']+aberturas+['Proporcao']]

      share_de_exogenas_semanal = pd.merge(share_de_exogenas_semanal,share_de_exogenas_semanal_soma_total,how='left',on=[col_data,'Positivos/Negativos','Endógena','Etapa']+aberturas)
      share_de_exogenas_semanal = pd.merge(share_de_exogenas_semanal,proporcao,how='left',on=[col_data,'Endógena','Etapa']+aberturas)

      share_de_exogenas_semanal['Fator Multiplicador'] = 1
      share_de_exogenas_semanal.loc[share_de_exogenas_semanal['Positivos/Negativos'] > 0,['Fator Multiplicador']] = np.abs(1/(1-share_de_exogenas_semanal.loc[share_de_exogenas_semanal['Positivos/Negativos'] > 0,['Proporcao']].values)) + 1
      share_de_exogenas_semanal.loc[share_de_exogenas_semanal['Positivos/Negativos'] < 0,['Fator Multiplicador']] = -1*np.abs(1/(1-share_de_exogenas_semanal.loc[share_de_exogenas_semanal['Positivos/Negativos'] < 0,['Proporcao']].values))

      share_de_exogenas_semanal['Share Redistribuido'] = share_de_exogenas_semanal['Share'] * (share_de_exogenas_semanal['Fator Multiplicador'].values/share_de_exogenas_semanal['Soma Share'].values)

      share_de_exogenas_semanal = share_de_exogenas_semanal[[col_data,'Endógena','Etapa']+aberturas+['Exógenas','Share Redistribuido']]

      share_de_exogenas_semanal = share_de_exogenas_semanal.rename(columns={'Share Redistribuido':'Share'})


    # Precisamos adicionar os shares de dados histórico na base de share de exógenas, para ter o efeito completo
    # da projeção não-maturada nas primeiras semanas de forecast:

    modelo_aberturas_share = share_de_exogenas_semanal.groupby(['Endógena','Etapa']+aberturas,as_index=False)['Share'].sum()[['Endógena','Etapa']+aberturas]
    modelo_aberturas_share['Exógenas'] = 'Past'
    modelo_aberturas_share['Share'] = 1
    lista_share_de_exogenas_semanal = [share_de_exogenas_semanal]
    if not fit_intercept:
      modelo_aberturas_share['Valor'] = 0
    for d in range(2*int(max_origin)):
      modelo_aberturas_share[col_data] = primeira_data_forecast-pd.Timedelta((1+int(d))*7, unit='D')
      lista_share_de_exogenas_semanal = lista_share_de_exogenas_semanal + [modelo_aberturas_share.copy()]
    share_de_exogenas_semanal = pd.concat(lista_share_de_exogenas_semanal)


  else:
    share_de_exogenas_semanal = pd.DataFrame(columns=[col_data,'Endógena','Etapa']+aberturas+['Exógenas','Share'])




  # Aplicar o share de exógenas no volume de ToF da base de forecast diária:
  #_______________________________________________________________________________________________

  # Selecionar apenas o forecast ToF da base de forecast diário e realizar um melt das etapas de ToF
  #forecast_tof_diario = df_forecast_diario.loc[df_forecast_diario[col_data] >= primeira_data_forecast,['date',col_data]+aberturas+topo]
  forecast_tof_diario = df_forecast_diario[['date',col_data]+aberturas+topo]
  forecast_tof_diario = forecast_tof_diario.melt(id_vars=['date',col_data]+aberturas, value_vars=topo,var_name='Etapa', value_name='Valor')
  forecast_tof_diario[col_data] = pd.to_datetime(forecast_tof_diario[col_data], infer_datetime_format=True)
  forecast_tof_semanal = forecast_tof_diario.groupby([col_data,'Etapa']+aberturas, as_index = False)['Valor'].sum()


  if 'Volume' in list(share_de_exogenas_semanal['Endógena'].values) and tipo_de_tof != 'Input Externo':


    # Selecionar apenas o share do volume de ToF da base de share
    share_de_exogenas_semanal_tof = share_de_exogenas_semanal.loc[share_de_exogenas_semanal['Endógena'] == 'Volume']
    share_de_exogenas_semanal_tof['Etapa'] = share_de_exogenas_semanal_tof['Etapa'].apply(lambda x: x.split('2')[0])

    # Aplicar o share de exógenas na base de forecast diária no caso de termos intercept:
    if fit_intercept:
      tof_share = pd.merge(forecast_tof_semanal,share_de_exogenas_semanal_tof,how='left',on=[col_data,'Etapa']+aberturas)
      tof_share['Endógena'] = tof_share['Endógena'].fillna('Volume')
      tof_share['Exógenas'] = tof_share['Exógenas'].fillna('Média')
      tof_share['Share'] = tof_share['Share'].fillna(1)
      tof_share['Valor x Share'] = tof_share['Valor'].values * tof_share['Share'].values

      tof_completo_share = tof_share


    else:
      tof_share_past = share_de_exogenas_semanal_tof.loc[share_de_exogenas_semanal_tof['Exógenas'] == 'Past']
      last_past_date = tof_share_past[col_data].max()

      forecast_tof_semanal_past = forecast_tof_semanal.loc[forecast_tof_semanal[col_data] <= last_past_date]

      tof_share_past = tof_share_past.drop(columns=['Valor'])
      tof_share_past = pd.merge(forecast_tof_semanal_past,tof_share_past,how='left',on=[col_data,'Etapa']+aberturas)
      tof_share_past['Endógena'] = tof_share_past['Endógena'].fillna('Volume')
      tof_share_past['Exógenas'] = tof_share_past['Exógenas'].fillna('Past')
      tof_share_past['Share'] = tof_share_past['Share'].fillna(1)
      tof_share_past['Valor x Share'] = tof_share_past['Valor'].values * tof_share_past['Share'].values

      tof_share = share_de_exogenas_semanal_tof.loc[share_de_exogenas_semanal_tof['Exógenas'] != 'Past']

      forecast_tof_semanal_future = forecast_tof_semanal.loc[forecast_tof_semanal[col_data] > last_past_date]

      tof_share = pd.merge(forecast_tof_semanal_future,tof_share,how='left',on=[col_data,'Etapa']+aberturas)
      #print(tof_share.loc[(tof_share[col_data] == '2022-12-19') & (tof_share['city_group'] == 'RMSP') & (tof_share['is_b2b_demand'] == 'CIQ_MANAGER')])
      tof_share['Endógena'] = tof_share['Endógena'].fillna('Volume')
      tof_share['Exógenas'] = tof_share['Exógenas'].fillna('Média')
      tof_share['Share'] = tof_share['Share'].fillna(1)

      tof_share.loc[tof_share['Exógenas'] != 'Média',['Valor x Share']] = tof_share.loc[tof_share['Exógenas'] != 'Média',['Valor_y']].values

      tof_share.loc[tof_share['Exógenas'] == 'Média',['Valor x Share']] = tof_share.loc[tof_share['Exógenas'] == 'Média',['Valor_x']].values * tof_share.loc[tof_share['Exógenas'] == 'Média',['Share']].values

      tof_share = tof_share.drop(columns=['Valor_x'])
      tof_share['Valor'] = tof_share['Valor x Share'].values
      tof_completo_share = pd.concat([tof_share,tof_share_past])

      tof_completo_share_mensal = pd.DataFrame()
      pivot_tof_completo_share_mensal = pd.DataFrame()



    # Agrupando por semana
    tof_completo_share_semanal = tof_completo_share.groupby([col_data,'Endógena','Exógenas','Etapa']+aberturas, as_index=False)['Valor x Share'].sum()

    # Calculando a variação MoM
    tof_completo_share_mensal_mom = tof_completo_share_mensal.copy()

    # Calculando a variação WoW
    tof_completo_share_semanal_wow = tof_completo_share_semanal.copy()
    tof_completo_share_semanal_wow_aux = tof_completo_share_semanal_wow.copy()
    tof_completo_share_semanal_wow_aux['Shifted'] = tof_completo_share_semanal_wow_aux[col_data].apply(lambda x: x-pd.Timedelta(7, unit='D'))
    tof_completo_share_semanal_wow['Shifted'] = tof_completo_share_semanal_wow[col_data].values
    tof_completo_share_semanal_wow = pd.merge(tof_completo_share_semanal_wow[['Shifted','Endógena','Exógenas','Etapa','Valor x Share']+aberturas],tof_completo_share_semanal_wow_aux[['Shifted','Endógena','Exógenas','Etapa','Valor x Share']+aberturas],how='left',on=['Shifted','Endógena','Exógenas','Etapa']+aberturas)
    tof_completo_share_semanal_wow = tof_completo_share_semanal_wow.fillna(0)
    tof_completo_share_semanal_wow['Delta Valor x Share'] = tof_completo_share_semanal_wow['Valor x Share_y'].values - tof_completo_share_semanal_wow['Valor x Share_x'].values
    tof_completo_share_semanal_wow = tof_completo_share_semanal_wow.rename(columns = {'Shifted':col_data})
    tof_completo_share_semanal_wow[col_data] = tof_completo_share_semanal_wow[col_data].apply(lambda x: x+pd.Timedelta(7, unit='D'))
    tof_completo_share_semanal = pd.merge(tof_completo_share_semanal,tof_completo_share_semanal_wow,how='left',on=[col_data,'Endógena','Exógenas','Etapa']+aberturas)
    output_tof_completo_share_semanal = tof_completo_share_semanal[[col_data]+aberturas+['Etapa','Endógena','Exógenas','Valor x Share','Delta Valor x Share']]

  else:
    output_tof_completo_share_semanal = pd.DataFrame()





  # Aplicar o share de exógenas da cohort aberta na base de forecast cohort:
  #_______________________________________________________________________________________________

  # Selecionar apenas o forecast ToF da base de forecast diário e realizar um melt das etapas de ToF
  #forecast_tof_diario = df_forecast_diario.loc[df_forecast_diario[col_data] >= primeira_data_forecast,['date',col_data]+aberturas+topo]
  col_conversoes = df_forecast_cohort.columns.values
  col_conversoes = col_conversoes[-len(etapas_volume):-1]

  forecast_cohort_semanal = df_forecast_cohort.loc[df_forecast_cohort['Week Origin'] != 'Coincident']
  forecast_cohort_semanal = forecast_cohort_semanal.groupby([col_data]+aberturas, as_index=False)[col_conversoes].sum()

  forecast_cohort_semanal = forecast_cohort_semanal.melt(id_vars=[col_data]+aberturas, value_vars=col_conversoes, var_name='Etapa', value_name='Valor')
  forecast_cohort_semanal[col_data] = pd.to_datetime(forecast_cohort_semanal[col_data], infer_datetime_format=True)

  if '%__Volume Aberta' in list(share_de_exogenas_semanal['Endógena'].values):


    # Selecionar apenas o share do volume de cohort da base de share
    share_de_exogenas_semanal_cohort = share_de_exogenas_semanal.loc[share_de_exogenas_semanal['Endógena'] == '%__Volume Aberta']
    share_de_exogenas_semanal_cohort['Etapa_Anterior'] = share_de_exogenas_semanal_cohort['Etapa'].apply(lambda x: x.split('2')[0])
    share_de_exogenas_semanal_cohort['Volume Aberta'] = 0

    # Vamos transformar as componentes das cohorts de conversões em volumes para poderem ser agregadas.
    # Multiplicar pelo volume da etapa anterior como se fossem coincidents:
    vol_etapa_anterior_tof = forecast_tof_semanal.copy()
    vol_etapa_anterior_tof = vol_etapa_anterior_tof.rename(columns={'Valor':'Valor_Etapa_Anterior','Etapa':'Etapa_Anterior'})
    vol_etapa_anterior = vol_etapa_anterior_tof.copy()

    for e in range(len(etapas_volume[:-1])):
      vol_etapa_anterior = pd.merge(vol_etapa_anterior,vol_etapa_anterior_tof,how='left',on=[col_data,'Etapa_Anterior']+aberturas)
      vol_etapa_anterior = vol_etapa_anterior.fillna(0)
      vol_etapa_anterior.loc[vol_etapa_anterior['Valor_Etapa_Anterior_x'] == 0,['Valor_Etapa_Anterior_x']] = \
      vol_etapa_anterior.loc[vol_etapa_anterior['Valor_Etapa_Anterior_x'] == 0,['Valor_Etapa_Anterior_y']].values
      vol_etapa_anterior = vol_etapa_anterior.rename(columns={'Valor_Etapa_Anterior_x':'Valor_Etapa_Anterior'})
      vol_etapa_anterior = vol_etapa_anterior.drop(columns=['Valor_Etapa_Anterior_y'])

      share_de_exogenas_semanal_cohort = pd.merge(share_de_exogenas_semanal_cohort,vol_etapa_anterior,how='left',on=[col_data,'Etapa_Anterior']+aberturas)

      share_de_exogenas_semanal_cohort.loc[share_de_exogenas_semanal_cohort['Etapa_Anterior'] == etapas_volume[e],['Volume Aberta']] = \
      share_de_exogenas_semanal_cohort.loc[share_de_exogenas_semanal_cohort['Etapa_Anterior'] == etapas_volume[e],['Valor']].values * \
      share_de_exogenas_semanal_cohort.loc[share_de_exogenas_semanal_cohort['Etapa_Anterior'] == etapas_volume[e],['Valor_Etapa_Anterior']].values

      vol_etapa_anterior = share_de_exogenas_semanal_cohort.groupby([col_data,'Etapa_Anterior']+aberturas, as_index = False)['Volume Aberta'].sum()
      vol_etapa_anterior = vol_etapa_anterior.loc[vol_etapa_anterior['Etapa_Anterior'] == etapas_volume[e]]
      vol_etapa_anterior['Etapa_Anterior'] = etapas_volume[e+1]

    share_de_exogenas_semanal_cohort = share_de_exogenas_semanal_cohort[[col_data,'Endógena','Etapa']+aberturas+['Exógenas','Volume Aberta','Share']]
    share_de_exogenas_semanal_cohort = share_de_exogenas_semanal_cohort.rename(columns={'Volume Aberta':'Valor'})


    # Aplicar o share de exógenas na base de forecast diária no caso de termos intercept:
    if fit_intercept:
      cohort_share = pd.merge(forecast_cohort_semanal,share_de_exogenas_semanal_cohort,how='left',on=[col_data,'Etapa']+aberturas)
      cohort_share['Endógena'] = cohort_share['Endógena'].fillna('%__Volume Aberta')
      cohort_share['Exógenas'] = cohort_share['Exógenas'].fillna('Média')
      cohort_share['Share'] = cohort_share['Share'].fillna(1)
      cohort_share['Valor x Share'] = cohort_share['Valor'].values * cohort_share['Share'].values

      cohort_completo_share = cohort_share


    else:
      cohort_share_past = share_de_exogenas_semanal_cohort.loc[share_de_exogenas_semanal_cohort['Exógenas'] == 'Past']
      forecast_cohort_semanal_past = forecast_cohort_semanal.loc[forecast_cohort_semanal[col_data] <= cohort_share_past[col_data].max()]
      cohort_share_past = cohort_share_past.drop(columns=['Valor'])
      cohort_share_past = pd.merge(forecast_cohort_semanal_past,cohort_share_past,how='left',on=[col_data,'Etapa']+aberturas)
      cohort_share_past['Endógena'] = cohort_share_past['Endógena'].fillna('%__Volume Aberta')
      cohort_share_past['Exógenas'] = cohort_share_past['Exógenas'].fillna('Past')
      cohort_share_past['Share'] = cohort_share_past['Share'].fillna(1)
      cohort_share_past['Valor x Share'] = cohort_share_past['Valor'].values * cohort_share_past['Share'].values

      cohort_share = share_de_exogenas_semanal_cohort.loc[share_de_exogenas_semanal_cohort['Exógenas'] != 'Past']
      forecast_cohort_semanal_future = forecast_cohort_semanal.loc[forecast_cohort_semanal[col_data] > cohort_share_past[col_data].max()]

      cohort_share = pd.merge(forecast_cohort_semanal_future,cohort_share,how='left',on=[col_data,'Etapa']+aberturas)
      #print(cohort_share.loc[(cohort_share[col_data] == '2022-12-19') & (cohort_share['city_group'] == 'RMSP') & (cohort_share['is_b2b_demand'] == 'CIQ_MANAGER')])
      cohort_share['Endógena'] = cohort_share['Endógena'].fillna('%__Volume Aberta')
      cohort_share['Exógenas'] = cohort_share['Exógenas'].fillna('Média')
      cohort_share['Share'] = cohort_share['Share'].fillna(1)
      cohort_share.loc[cohort_share['Exógenas'] != 'Média',['Valor x Share']] = cohort_share.loc[cohort_share['Exógenas'] != 'Média',['Valor_y']].values
      cohort_share.loc[cohort_share['Exógenas'] == 'Média',['Valor x Share']] = cohort_share.loc[cohort_share['Exógenas'] == 'Média',['Valor_x']].values * cohort_share.loc[cohort_share['Exógenas'] == 'Média',['Share']].values
      cohort_share = cohort_share.drop(columns=['Valor_x'])
      cohort_share['Valor'] = cohort_share['Valor x Share'].values
      cohort_completo_share = pd.concat([cohort_share,cohort_share_past])

      cohort_completo_share_mensal = pd.DataFrame()
      pivot_cohort_completo_share_mensal = pd.DataFrame()



    # Agrupando por semana
    cohort_completo_share_semanal = cohort_completo_share.groupby([col_data,'Endógena','Exógenas','Etapa']+aberturas, as_index=False)['Valor x Share'].sum()

    # Calculando a variação MoM
    cohort_completo_share_mensal_mom = cohort_completo_share_mensal.copy()

    # Calculando a variação WoW
    cohort_completo_share_semanal_wow = cohort_completo_share_semanal.copy()
    cohort_completo_share_semanal_wow_aux = cohort_completo_share_semanal_wow.copy()
    cohort_completo_share_semanal_wow_aux['Shifted'] = cohort_completo_share_semanal_wow_aux[col_data].apply(lambda x: x-pd.Timedelta(7, unit='D'))
    cohort_completo_share_semanal_wow['Shifted'] = cohort_completo_share_semanal_wow[col_data].values
    cohort_completo_share_semanal_wow = pd.merge(cohort_completo_share_semanal_wow[['Shifted','Endógena','Exógenas','Etapa','Valor x Share']+aberturas],cohort_completo_share_semanal_wow_aux[['Shifted','Endógena','Exógenas','Etapa','Valor x Share']+aberturas],how='left',on=['Shifted','Endógena','Exógenas','Etapa']+aberturas)
    cohort_completo_share_semanal_wow = cohort_completo_share_semanal_wow.fillna(0)
    cohort_completo_share_semanal_wow['Delta Valor x Share'] = cohort_completo_share_semanal_wow['Valor x Share_y'].values - cohort_completo_share_semanal_wow['Valor x Share_x'].values
    cohort_completo_share_semanal_wow = cohort_completo_share_semanal_wow.rename(columns = {'Shifted':col_data})
    cohort_completo_share_semanal_wow[col_data] = cohort_completo_share_semanal_wow[col_data].apply(lambda x: x+pd.Timedelta(7, unit='D'))
    cohort_completo_share_semanal = pd.merge(cohort_completo_share_semanal,cohort_completo_share_semanal_wow,how='left',on=[col_data,'Endógena','Exógenas','Etapa']+aberturas)
    output_cohort_completo_share_semanal = cohort_completo_share_semanal[[col_data]+aberturas+['Etapa','Endógena','Exógenas','Valor x Share','Delta Valor x Share']]
  else:
    output_cohort_completo_share_semanal = pd.DataFrame()


  # Unimos as componentes cohort com as componentes ToF:
  output_completo_share_semanal = pd.concat([output_tof_completo_share_semanal,output_cohort_completo_share_semanal])

  return output_completo_share_semanal

  #return pd.DataFrame()


