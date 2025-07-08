#@title Def forecast_2 (fdf 2)
import pandas as pd
import numpy as np
from Auxiliar_Forecast import *
'''
Descrição Geral:

# Projeta volume de ToF:
_________________
(fdf 3) Auxiliar_Forecast
_________________
# Depois de projetar o volume do topo, precisamos recalcular as colunas exógenas de volume transformado, pois antes dessa projeção elas nem tinham volume futuro para serem calculadas

# Para cada cohort fechada, vamos projetar o share a partir do histórico maturado:

  _________________
  (fdf 3) Auxiliar_Forecast
  _________________

# Agora vamos projetar a cohort aberta dentro do histórico não maturado:
Aplicamos a equação que determina a cohort aberta a partir da projeção dos shares e das cohorts já maturadas: CA(w) = W(w-1) / s_(w-1) + [sum(W) from W0 to W(w-2)]
# Calculamos as cohorts fechadas.
Como o cálculo de uma cohort fechada depende da projeção das fechadas anteriores, precisamos calcular uma por uma. W = (s_W) * (CA - soma das cohorts que já existem ou já foram projetadas anteriormente)

# Vamos projetar a cohort Aberta a partir do histórico completo, contando com a cohort projetada não maturada
_________________
(fdf 3) Auxiliar_Forecast
_________________
# Vamos projetar as cohorts fechadas a partir da projeção da aberta e dos shares
Wn = share Wn * (Cohort Aberta - soma das cohorts fechadas anteriores)

return:
 df_completo_projetado,
 out_parametros

'''

def forecast_2(df_completo,      # DataFrame filtrado, somente datas e valores
            df_parametros,
            df_pareto_aberturas,
            df_aberturas_clusterizadas, # DataFrame filtrado,
            col_data,   # DataFrame filtrado, somente datas e valores
            data_corte,
            primeira_data_forecast,
            maturacao,
            data_maturacao,
            abertura,
            etapa,                    # String com o nome da etapa do funil
            topo,                      # Lista com o nome das etapas do ToF.
            max_origin,               # int indicando qual a maior cohort do histórico
            conversoes,               # Lista com os nomes das conversões sem '%__Volume Aberta' ('s__0','s__1',...,'s__Coincident')
            qtd_semanas_media,
            df_inputs_exogenas,
            tipo_de_projecao,
            remover_outliers,
            estimativa_outliers,
            retorna_parametros,
            tipo_de_tof,
              premissa_dummy,
              aberta_zerada,
              share_w0_zerado,
              limite_share,
              limite_delta_vol,
              limite_delta_share,
              limite_delta_aberta,
              limite_proj,
               fit_intercept):
  #print(df_completo[[col_data,'Volume Aberta','%__Volume Aberta','1','s__1','%__1']])
  df_completo_projetado = df_completo.copy()

  out_parametros = pd.DataFrame()
  matriz_forecast = pd.DataFrame()
  base_contencao_de_danos = pd.DataFrame()
  base_outliers = pd.DataFrame()


  #-------------------------------------------------------------------------------------------------

  # Abaixo vamos projetar o volume de ToF e também os shares das cohorts fechadas.
  # Com base nos shares das cohorts fechadas vamos projetar a cohort aberta no histórico não maturado
  # Depois vamos usar o histórico completo para projetar a cohort aberta e calcular a projeção das fechadas
  #_________________________________________________________________________________________________
  # Caso a etapa sendo projetada inclui o ToF, faremos uma projeção específica:

  if tipo_de_tof != 'Input Externo' and etapa.split('2')[0] in topo:

    endogenous = 'Volume'

    # Definimos o tipo de projeção de acordo com a categoria da abertura na análise de pareto e também
    # se a abertura requer projeção via cluster:

    if len(df_pareto_aberturas) == 0:
      categoria_pareto = 0
    else:
      categoria_pareto = df_pareto_aberturas['pareto_'+etapa].values[0]


    if categoria_pareto == 3 or tipo_de_projecao == 'Cluster':
      tipo_de_projecao_local = 'Average'
    else:
      tipo_de_projecao_local = tipo_de_projecao

    # Realizamos a projeção
    df_completo_projetado,out_parametros,matriz_forecast,base_outliers_local,base_contencao_de_danos_local =  Auxiliar_Forecast(df_completo = df_completo_projetado, # DF filtrado somente a abertura e Endog, ordenado com a data mais antiga no topo
                                                                                                                                df_inputs_exogenas = df_inputs_exogenas,
                                                                                                                                df_parametros = df_parametros,
                                                                                                                                abertura = abertura,
                                                                                                                                etapa = etapa,
                                                                                                                                col_data = col_data,
                                                                                                                                maturacao = maturacao,
                                                                                                                                data_corte = data_corte,  # DateTime indicando onde acaba o historico
                                                                                                                                primeira_data_forecast = primeira_data_forecast,
                                                                                                                                endogenous = endogenous,   # String com o nome da variável endog
                                                                                                                                tipo_de_projecao = tipo_de_projecao_local,
                                                                                                                                qtd_semanas_media = qtd_semanas_media,
                                                                                                                                max_origin = max_origin,
                                                                                                                                remover_outliers = remover_outliers,
                                                                                                                                estimativa_outliers = estimativa_outliers,
                                                                                                                                retorna_parametros = retorna_parametros,
                                                                                                                                  premissa_dummy = premissa_dummy,
                                                                                                                                  aberta_zerada = aberta_zerada,
                                                                                                                                  share_w0_zerado = share_w0_zerado,
                                                                                                                                  limite_share = limite_share,
                                                                                                                                  limite_delta_vol = limite_delta_vol,
                                                                                                                                  limite_delta_share = limite_delta_share,
                                                                                                                                  limite_delta_aberta = limite_delta_aberta,
                                                                                                                                fit_intercept = fit_intercept,
                                                                                                                               limite_proj = limite_proj)

    # Atualizamos as bases periféricas:
    base_contencao_de_danos = pd.concat([base_contencao_de_danos,base_contencao_de_danos_local])
    base_outliers = pd.concat([base_outliers,base_outliers_local])

  # Depois de projetar o volume do topo, precisamos recalcular as colunas exógenas de volume
  # transformado, pois antes dessa projeção elas nem tinham volume futuro para serem calculadas:
  if etapa.split('2')[0] in topo:
    col_exogs_volume = list(df_completo_projetado.columns.values)
    col_exogs_volume = col_exogs_volume[col_exogs_volume.index('s__Coincident')+1:]
    col_exogs_volume = col_exogs_volume = [l for l in col_exogs_volume if 'Volume___' in l]
    df_completo_projetado = df_completo_projetado.drop(columns=col_exogs_volume)
    df_completo_projetado = transforma_exogs_2(df = df_completo_projetado, # Dataframe filtrado de dados futuros
                                                exogenous_l_d = col_exogs_volume, # Lista com o nome das variaveis exogenas (com os suffixos "l" e "d")
                                                col_data = col_data) # String com o nome da coluna de datas


  #-------------------------------------------------------------------------------------------------
  # Para cada cohort fechada, vamos projetar o share a partir do histórico maturado:

  for c in conversoes:

    endogenous = c

    # Definimos o tipo de projeção de acordo com a categoria da abertura na análise de pareto
    if len(df_pareto_aberturas) == 0:
      categoria_pareto = 0
    else:
      categoria_pareto = df_pareto_aberturas['pareto_etapa_final'].values[0]


    if categoria_pareto == 3:
      tipo_de_projecao_local = 'Average'
    else:
      tipo_de_projecao_local = tipo_de_projecao

    # Precisamos alterar a data de corte e início da projeção com base na maturação da cohort do share que vamos projetar:
    proxy_maturacao = int(endogenous.split('__')[1])
    data_limite_hist = data_corte - pd.Timedelta(proxy_maturacao*7, unit='D')
    data_limite_hist = min(data_maturacao,data_limite_hist)


    # Realizamos a projeção
    if tipo_de_projecao != 'Cluster':

      df_completo_projetado,out_parametros_c,matriz_forecast_c,base_outliers_local,base_contencao_de_danos_local =  Auxiliar_Forecast(df_completo = df_completo_projetado, # DF filtrado somente a abertura e Endog, ordenado com a data mais antiga no topo
                                                                                                                                      df_inputs_exogenas = df_inputs_exogenas,
                                                                                                                                      df_parametros = df_parametros,
                                                                                                                                      abertura = abertura,
                                                                                                                                      etapa = etapa,
                                                                                                                                      col_data = col_data,
                                                                                                                                      maturacao = maturacao,
                                                                                                                                      data_corte = data_limite_hist,  # DateTime indicando onde acaba o historico
                                                                                                                                      primeira_data_forecast = primeira_data_forecast,
                                                                                                                                      endogenous = endogenous,   # String com o nome da variável endog
                                                                                                                                      tipo_de_projecao = tipo_de_projecao_local,
                                                                                                                                      qtd_semanas_media = qtd_semanas_media,
                                                                                                                                      max_origin = max_origin,
                                                                                                                                      remover_outliers = remover_outliers,
                                                                                                                                      estimativa_outliers = estimativa_outliers,
                                                                                                                                      retorna_parametros = retorna_parametros,
                                                                                                                                        premissa_dummy = premissa_dummy,
                                                                                                                                        aberta_zerada = aberta_zerada,
                                                                                                                                        share_w0_zerado = share_w0_zerado,
                                                                                                                                        limite_share = limite_share,
                                                                                                                                        limite_delta_vol = limite_delta_vol,
                                                                                                                                        limite_delta_share = limite_delta_share,
                                                                                                                                        limite_delta_aberta = limite_delta_aberta,
                                                                                                                                      fit_intercept = fit_intercept,
                                                                                                                                     limite_proj = limite_proj)

      
      # Atualizamos as bases periféricas:
      base_contencao_de_danos = pd.concat([base_contencao_de_danos,base_contencao_de_danos_local])
      base_outliers = pd.concat([base_outliers,base_outliers_local])


    else:
      df_completo_projetado.loc[(df_completo_projetado[col_data] > data_maturacao),[endogenous]] = df_aberturas_clusterizadas[endogenous].values[0]
      out_parametros_c = pd.DataFrame()
      matriz_forecast_c = pd.DataFrame()


    # Vamos adicionar os parâmetros calculados dos shares de conversão aos de volume:
    out_parametros = pd.concat([out_parametros,out_parametros_c])
    # Vamos adicionar a matriz de forecast dos shares à matriz anterior
    matriz_forecast = pd.concat([matriz_forecast,matriz_forecast_c])


  #-------------------------------------------------------------------------------------------------
  # No caso da cohort de ajuste, vamos projeta-la com base na média simples da conversão em %, pois
  # o share da mesma não é bem definido, principalmente quando seu valor é superior a 100%:

  endogenous = '%__Coincident'

  # Precisamos alterar a data de corte e início da projeção com base na maturação da cohort do share que vamos projetar:
  proxy_maturacao = int(max_origin)
  data_limite_hist = data_corte - pd.Timedelta(proxy_maturacao*7, unit='D')

  if tipo_de_projecao != 'Cluster':
    df_completo_projetado,out_parametros_c,matriz_forecast_c,base_outliers_local,base_contencao_de_danos_local =  Auxiliar_Forecast(df_completo = df_completo_projetado, # DF filtrado somente a abertura e Endog, ordenado com a data mais antiga no topo
                                                                                                                                    df_inputs_exogenas = df_inputs_exogenas,
                                                                                                                                    df_parametros = df_parametros,
                                                                                                                                    abertura = abertura,
                                                                                                                                    etapa = etapa,
                                                                                                                                    col_data = col_data,
                                                                                                                                    maturacao = maturacao,
                                                                                                                                    data_corte = data_limite_hist,  # DateTime indicando onde acaba o historico
                                                                                                                                    primeira_data_forecast = primeira_data_forecast,
                                                                                                                                    endogenous = endogenous,   # String com o nome da variável endog
                                                                                                                                    tipo_de_projecao = 'Average',
                                                                                                                                    qtd_semanas_media = qtd_semanas_media,
                                                                                                                                    max_origin = max_origin,
                                                                                                                                    remover_outliers = remover_outliers,
                                                                                                                                    estimativa_outliers = estimativa_outliers,
                                                                                                                                    retorna_parametros = retorna_parametros,
                                                                                                                                      premissa_dummy = premissa_dummy,
                                                                                                                                      aberta_zerada = aberta_zerada,
                                                                                                                                      share_w0_zerado = share_w0_zerado,
                                                                                                                                      limite_share = limite_share,
                                                                                                                                      limite_delta_vol = limite_delta_vol,
                                                                                                                                      limite_delta_share = limite_delta_share,
                                                                                                                                      limite_delta_aberta = limite_delta_aberta,
                                                                                                                                    fit_intercept = fit_intercept,
                                                                                                                                   limite_proj = limite_proj)

    # Atualizamos as bases periféricas:
    base_contencao_de_danos = pd.concat([base_contencao_de_danos,base_contencao_de_danos_local])
    base_outliers = pd.concat([base_outliers,base_outliers_local])

  else:
    df_completo_projetado.loc[(df_completo_projetado[col_data] > data_maturacao),[endogenous]] = df_aberturas_clusterizadas[endogenous].values[0]
    out_parametros_c = pd.DataFrame()
    matriz_forecast_c = pd.DataFrame()

  #_________________________________________________________________________________________________



  # Agora vamos projetar a cohort aberta dentro do histórico não maturado:
  #_________________________________________________________________________________________________


  # Vamos selecionar o hisórico não maturado a partir da base completa
  df_nao_maturado_projetado = df_completo_projetado.loc[(df_completo_projetado[col_data] > data_maturacao) & (df_completo_projetado[col_data] < primeira_data_forecast)]
  df_maturado_projetado = df_completo_projetado.loc[(df_completo_projetado[col_data] <= data_maturacao) & (df_completo_projetado[col_data] >= data_maturacao - np.timedelta64(qtd_semanas_media*7,'D'))]
  # Caso a abretura mature na W0, não vai existir histórico não maturado. Nesse caso não precisamos
  # projetar o histórico não maturado e ir direto para a projeção da cohort aberta considerando o
  # histórico realizado completo:
  #print("*******len(df_nao_maturado_projetado)",len(df_nao_maturado_projetado),maturacao)
  if len(df_nao_maturado_projetado) != 0 and int(max_origin)>0:

    # Vamos criar uma coluna auxiliar com o número da semana dentro do historico:
    n_semanas = int(maturacao)
    #print(np.unique(df_nao_maturado_projetado[col_data].values))
    #print(n_semanas)
    data_min = np.min(df_nao_maturado_projetado[col_data].values)

    # A semana mais recente é 1 e a mais antiga é n_semanas
    df_nao_maturado_projetado['week_number'] = n_semanas - (((df_nao_maturado_projetado[col_data] - data_min)/ np.timedelta64(1,'D')) / 7)
    df_nao_maturado_projetado['week_number'] = df_nao_maturado_projetado['week_number'].astype(int)



    # Agora vamos percorrer cada semana e aplicar as equações que utilizam as projeções de shares
    # e as cohorts fechadas maturadas para projetar a cohort aberta e em seguida usar a cohort
    # aberta projetada e os shares para projetar as outras cohorts:

    # Para cada semana do histórico não maturado:
    for i in range(n_semanas):

      # Com base no índice da semana, sabemos quais cohorts da base já estão maturadas
      # (a maior cohort e a Coincident nunca estão maturadas)
      lista_cohorts_maturadas = list(range(i+1))
      for w in range(len(lista_cohorts_maturadas)):
        lista_cohorts_maturadas[w] = "%__"+str(lista_cohorts_maturadas[w])

      #print(i,"lista_cohorts_maturadas")
      #print(lista_cohorts_maturadas)

      # Fazemos o equivalente para identificar as cohorts que ainda não maturaram e precisam ser projetadas
      lista_cohorts_projetadas = list(range(i+1,n_semanas))
      for w in range(len(lista_cohorts_projetadas)):
        lista_cohorts_projetadas[w] = "%__"+str(lista_cohorts_projetadas[w])

      # Mesmo que a maturação da abertura seja rápida, devemos incluir a projeção não maturada da
      # última cohort e da cohort de ajuste
      lista_cohorts_projetadas = lista_cohorts_projetadas+["%__"+str(max_origin)]
      #lista_cohorts_projetadas = lista_cohorts_projetadas+["%__Coincident"]

      # Selecionamos apenas a semana a ser calculada
      historico_nao_maturado_semana = df_nao_maturado_projetado.loc[df_nao_maturado_projetado['week_number'] == i+1]

      # Identificamos qual cohort cuja projeção do share vai ser utilizada no cálculo da cohort aberta
      share_p_ca = "s__"+str(i)

      # Calculamos qual o peso acumulado do share das fechadas dentro da aberta naquela semana
      if i == 0:
        share_acumulado = historico_nao_maturado_semana["s__0"].values
      else:
        share_acumulado = 0
      for s in range(i):
        if s == 0:
          share_acumulado = historico_nao_maturado_semana["s__0"].values
        else:
          share_acumulado = share_acumulado + (1-share_acumulado)*historico_nao_maturado_semana["s__"+str(s)].values

      #---------------------------------------------------------------------------------------------

      # Aplicamos a equação que determina a cohort aberta a partir da projeção dos shares e das cohorts
      # já maturadas:
      # CA(w) = W(w-1) / s_(w-1) + [sum(W) from W0 to W(w-2)]
      # Melhoria: vamos ponderar a projeção da aberta pelo peso que as fechadas maturadas tem dentro dela e usar a média da cohort aberta maturada como parâmetro:
      # CA(w) = {W(w-1) / s_(w-1) + [sum(W) from W0 to W(w-2)]}*share_acumulado + CA(média maturada)*(1-share_acumulado)

      CA_media_maturada = projeta_por_media(df_maturado_projetado,'%__Volume Aberta',1,qtd_semanas_media)

      # Não vamos considerar o share da cohort Coincident para calcular a cohort aberta. Precisamos
      # somente da última cohort para isso

      # Caso o share da cohort seja zero, precisamos calcular com base na W0 somente
      if sum(historico_nao_maturado_semana[share_p_ca].values) == 0 and sum(historico_nao_maturado_semana["s__0"].values) != 0:
        # Calculamos a cohort aberta dessa semana
        CA_pelo_share_w0 = historico_nao_maturado_semana["%__0"].values / historico_nao_maturado_semana["s__0"].values
        CA_ponderada = CA_pelo_share_w0 * share_acumulado + CA_media_maturada * (1-share_acumulado)
        # Caso a cohort aberta ponderada seja menor do que a w0 já realizada, vamos pela primeira opção:
        if CA_ponderada < historico_nao_maturado_semana["%__0"].values:
          historico_nao_maturado_semana['%__Volume Aberta'] = CA_pelo_share_w0
        else:
          historico_nao_maturado_semana['%__Volume Aberta'] = CA_ponderada


      elif sum(historico_nao_maturado_semana[share_p_ca].values) == 0 and sum(historico_nao_maturado_semana["s__0"].values) == 0:
        historico_nao_maturado_semana['%__Volume Aberta'] = projeta_por_media(df_maturado_projetado,'%__Volume Aberta',1,qtd_semanas_media)

      else:
        # Calculamos a cohort aberta dessa semana

        cohort_parcial_maturada = np.sum(historico_nao_maturado_semana[lista_cohorts_maturadas[:-1]].values)

        cohort_total = np.sum(historico_nao_maturado_semana[lista_cohorts_maturadas].values)

        cohort_faltante_projetada = historico_nao_maturado_semana[lista_cohorts_maturadas[-1]].values \
                                  / historico_nao_maturado_semana[share_p_ca].values

        CA_pelo_share = cohort_faltante_projetada + cohort_parcial_maturada
        CA_ponderada = CA_pelo_share * share_acumulado + CA_media_maturada * (1-share_acumulado)
        # Se a cohort aberta ponderada for menor do que as cohorts que já maturaram, vamos pela primeira opção:
        if CA_ponderada < cohort_total:
          historico_nao_maturado_semana['%__Volume Aberta'] = CA_pelo_share
        else:
          historico_nao_maturado_semana['%__Volume Aberta'] = CA_ponderada

      '''
      print('____________________calculo aberta_______________')
      print('CA(w) = W(w-1) / s_(w-1) + [sum(W) from W0 to W(w-2)]')
      print('Semena',i)
      print(historico_nao_maturado_semana[['%__Volume Aberta',"%__0","s__0"]])
      print("W(w-1)")
      print(lista_cohorts_maturadas[-1],historico_nao_maturado_semana[lista_cohorts_maturadas[-1]].values)
      print("s_(w-1)")
      print(share_p_ca,historico_nao_maturado_semana[share_p_ca])
      print("[sum(W) from W0 to W(w-2)]")
      print(lista_cohorts_maturadas[:-1],np.sum(historico_nao_maturado_semana[lista_cohorts_maturadas[:-1]].values))
      '''

      # Vamos tratar o forecast da cohort aberta parcialmente maturada para remover possíveis
      # projeções muito divergentes da média:
      #---------------------------------------------------------------------------------------------

      cohort_aberta_media = df_maturado_projetado['%__Volume Aberta'].mean()
      cohort_aberta_std = df_maturado_projetado['%__Volume Aberta'].std()

      cohort_aberta_projetada_original = historico_nao_maturado_semana['%__Volume Aberta'].values

      if cohort_aberta_projetada_original > limite_delta_aberta*cohort_aberta_std+cohort_aberta_media:
        historico_nao_maturado_semana['%__Volume Aberta'] = limite_delta_aberta*cohort_aberta_std+cohort_aberta_media

      if cohort_aberta_projetada_original < cohort_aberta_media-limite_delta_aberta*cohort_aberta_std:
        historico_nao_maturado_semana['%__Volume Aberta'] = cohort_aberta_media-limite_delta_aberta*cohort_aberta_std

      if cohort_aberta_projetada_original > limite_proj:
        historico_nao_maturado_semana['%__Volume Aberta'] = limite_proj

      if cohort_aberta_projetada_original < 0:
        historico_nao_maturado_semana['%__Volume Aberta'] = 0.

      # Como premissa, vamos alterar o share da w0 para compensar a possível mudança na cohort aberta:
      if historico_nao_maturado_semana['%__Volume Aberta'].values != cohort_aberta_projetada_original:
        historico_nao_maturado_semana['s__0'] = historico_nao_maturado_semana['%__Volume Aberta'].values/historico_nao_maturado_semana['%__0'].values

      #---------------------------------------------------------------------------------------------
      # Calculamos as cohorts fechadas.
      # Como o cálculo de uma cohort fechada depende da projeção das fechadas anteriores, precisamos
      # calcular uma por uma.
      # W = (s_W) * (CA - soma das cohorts que já existem ou já foram projetadas anteriormente)

      # Iniciamos uma lista que irá conter as cohorts que já foram calculadas ou que já maturaram na
      # semana que estamos calculando:
      lista_cohorts_calculadas = lista_cohorts_projetadas.copy()
      lista_cohorts_calculadas = [int(x.split('__')[-1]) for x in lista_cohorts_calculadas]
      min_calculada = min(lista_cohorts_calculadas)
      lista_cohorts_calculadas = []
      for l in range(min_calculada):
        lista_cohorts_calculadas = lista_cohorts_calculadas+['%__'+str(l)]
      '''
      lista_cohorts_calculadas = ['%__0']
      for l in range(1,max_origin-len(lista_cohorts_projetadas)+2):
        lista_cohorts_calculadas = lista_cohorts_calculadas+['%__'+str(l)]
      '''

      for w in lista_cohorts_projetadas:


        # Identificamos qual cohort cuja projeção do share vai ser utilizada no cálculo das cohorts fechadas
        share_p_w = 's__'+w.split('__')[-1]

        # O cálculo em si é o seguinte:
        '''
        if w == '%__Coincident': # Caso seja o cálculo da cohort de ajuste, a soma das cohorts anteriores não pode incluir a última cohort (W5>)
          lista_cohorts_calculadas = lista_cohorts_calculadas[:-1]
        '''

        historico_nao_maturado_semana[w] = historico_nao_maturado_semana[share_p_w] \
                                                    * (historico_nao_maturado_semana['%__Volume Aberta'] \
                                                        - np.sum(historico_nao_maturado_semana[lista_cohorts_calculadas].values))
        '''
        print("___________________________________________________")
        print(w,historico_nao_maturado_semana[w].values,'%__0',historico_nao_maturado_semana['%__0'].values)
        print(share_p_w,historico_nao_maturado_semana[share_p_w].values)
        print("CA",historico_nao_maturado_semana['%__Volume Aberta'].values)
        print(lista_cohorts_calculadas,np.sum(historico_nao_maturado_semana[lista_cohorts_calculadas].values))

        print("--------------------------------------------------------------")
        print(i,n_semanas)
        print(lista_cohorts_projetadas)
        print(lista_cohorts_calculadas)
        print(lista_cohorts_maturadas[:-1])
        print("cohort",w," =")
        print(historico_nao_maturado_semana[w].values)
        print("= ",historico_nao_maturado_semana[share_p_w].values)
        print("* (",historico_nao_maturado_semana['%__Volume Aberta'].values)
        print("- sum:   ",np.sum(historico_nao_maturado_semana[lista_cohorts_calculadas].values),")")
        '''
        # Caso as cohorts já maturadas e calculadas, se somadas, ultrapassam o valor da cohort aberta
        # projetada, zeramos a cohort fechada para não calcular valores negativos:
        if historico_nao_maturado_semana[w].values < 0 and w != '%__Coincident':
          historico_nao_maturado_semana[w] = 0.0


        # Adicionamos à lista a cohort que já foi calculada. No início, a cohort já
        # calculada é a W0, que na verdade já está maturada
        lista_cohorts_calculadas = lista_cohorts_calculadas+[w]





      # Atualizamos os valores da base não maturada com a projeção da semana não maturada:
      df_nao_maturado_projetado.loc[df_nao_maturado_projetado['week_number'] == i+1]  = historico_nao_maturado_semana

    # Atualizamos a base completa com o histórico não maturado projetado:
    df_nao_maturado_projetado = df_nao_maturado_projetado.drop(columns=['week_number'])
    #print(df_completo_projetado[[col_data]+['%__Volume Aberta','%__0','s__0','s__5','s__Coincident','%__Coincident']])
    #print(df_nao_maturado_projetado[[col_data]+['%__Volume Aberta','%__0','s__0','s__5','s__Coincident','%__Coincident']])
    df_completo_projetado.loc[(df_completo_projetado[col_data] > data_maturacao) & (df_completo_projetado[col_data] < primeira_data_forecast)] = df_nao_maturado_projetado

  #_________________________________________________________________________________________________





  # Vamos projetar a cohort Aberta a partir do histórico completo, contando com a cohort projetada
  # não maturada
  #_________________________________________________________________________________________________

  endogenous = '%__Volume Aberta'

  # Definimos o tipo de projeção de acordo com a categoria da abertura na análise de pareto
  if len(df_pareto_aberturas) == 0:
    categoria_pareto = 0
  else:
    categoria_pareto = df_pareto_aberturas['pareto_etapa_final'].values[0]
  if categoria_pareto == 3:
    tipo_de_projecao_local = 'Average'
  else:
    tipo_de_projecao_local = tipo_de_projecao

  # Realizamos a projeção
  if tipo_de_projecao != 'Cluster':

    df_completo_projetado,out_parametros_a,matriz_forecast_a,base_outliers_local,base_contencao_de_danos_local =  Auxiliar_Forecast(df_completo = df_completo_projetado, # DF filtrado somente a abertura e Endog, ordenado com a data mais antiga no topo
                                                                                                                                    df_inputs_exogenas = df_inputs_exogenas,
                                                                                                                                    df_parametros = df_parametros,
                                                                                                                                    abertura = abertura,
                                                                                                                                    etapa = etapa,
                                                                                                                                    col_data = col_data,
                                                                                                                                    maturacao = maturacao,
                                                                                                                                    data_corte = data_corte,  # DateTime indicando onde acaba o historico
                                                                                                                                    primeira_data_forecast = primeira_data_forecast,
                                                                                                                                    endogenous = endogenous,   # String com o nome da variável endog
                                                                                                                                    tipo_de_projecao = tipo_de_projecao_local,
                                                                                                                                    qtd_semanas_media = qtd_semanas_media,
                                                                                                                                    max_origin = max_origin,
                                                                                                                                    remover_outliers = remover_outliers,
                                                                                                                                    estimativa_outliers = estimativa_outliers,
                                                                                                                                    retorna_parametros = retorna_parametros,
                                                                                                                                      premissa_dummy = premissa_dummy,
                                                                                                                                      aberta_zerada = aberta_zerada,
                                                                                                                                      share_w0_zerado = share_w0_zerado,
                                                                                                                                      limite_share = limite_share,
                                                                                                                                      limite_delta_vol = limite_delta_vol,
                                                                                                                                      limite_delta_share = limite_delta_share,
                                                                                                                                      limite_delta_aberta = limite_delta_aberta,
                                                                                                                                    fit_intercept = fit_intercept,
                                                                                                                                   limite_proj = limite_proj)


    # Atualizamos as bases periféricas:
    base_contencao_de_danos = pd.concat([base_contencao_de_danos,base_contencao_de_danos_local])
    base_outliers = pd.concat([base_outliers,base_outliers_local])

  else:
    df_completo_projetado.loc[(df_completo_projetado[col_data] >= primeira_data_forecast),[endogenous]] = df_aberturas_clusterizadas['Cohort Aberta Média'].values[0]
    out_parametros_a = pd.DataFrame()
    matriz_forecast_a = pd.DataFrame()

    # Realizamos a contenção de danos:
    '''
    endog_projetada,mensagem,base_contencao_de_danos_local = contencao_de_danos(df = df_completo_projetado, # filtrado na etapa e abertura
                                                                          endogenous = endogenous,
                                                                          col_data = col_data,
                                                                          data_corte = data_corte,
                                                                          primeira_data_forecast = primeira_data_forecast,
                                                                          tipo_de_forecast = tipo_de_projecao,
                                                                          qtd_semanas_media = qtd_semanas_media,
                                                                          subs_aberta_zerada = aberta_zerada,
                                                                          subs_share_w0_zerado = share_w0_zerado,
                                                                          limite_inferior_share = limite_share,
                                                                          limite_delta_media_vol = limite_delta_vol,
                                                                          limite_delta_media_share = limite_delta_share,
                                                                          limite_delta_media_aberta = limite_delta_aberta)

    # Preenchemos a base transformada com a projeção corrigida
    df_completo_projetado.loc[df_completo_projetado[col_data] >= primeira_data_forecast,[endogenous]] = endog_projetada

    # Atualizamos as bases periféricas:
    base_contencao_de_danos = pd.concat([base_contencao_de_danos,base_contencao_de_danos_local])
    '''
  # Removemos os shares projetados que não fazem sentido
  if endogenous != 's__Coincident':
    df_completo_projetado.loc[df_completo_projetado[endogenous] < 0,[endogenous]] = 0.0
    df_completo_projetado.loc[df_completo_projetado[endogenous] > limite_proj,[endogenous]] = limite_proj


  # Vamos adicionar os parâmetros calculados da cohort aberta aos de volume e shares:
  out_parametros = pd.concat([out_parametros,out_parametros_a])
  matriz_forecast = pd.concat([matriz_forecast,matriz_forecast_a])

  #_________________________________________________________________________________________________



  # Vamos projetar as cohorts fechadas a partir da projeção da aberta e dos shares
  #_________________________________________________________________________________________________

  cohorts_projetadas = conversoes.copy()
  for i in range(len(cohorts_projetadas)):
    cohorts_projetadas[i] = cohorts_projetadas[i].replace("s","%")

    # Precisamos alterar a data de corte e início da projeção com base na maturação da cohort do share que vamos projetar:
    #maturacao = maturacao = int(cohorts_projetadas[i].split('__')[1])
    data_limite_hist = data_corte - pd.Timedelta(maturacao*7, unit='D')
    data_limite_hist = data_corte

    if primeira_data_forecast < data_limite_hist:
      data_limite_hist = primeira_data_forecast
    else:
      data_limite_hist = data_limite_hist

    # O cálcula da cohort fechada é o seguinte:
    # Wn = share Wn * (Cohort Aberta - soma das cohorts fechadas anteriores)
    if i == 0:
      df_completo_projetado.loc[df_completo_projetado[col_data] > data_limite_hist, [cohorts_projetadas[i]]] =\
      df_completo_projetado.loc[df_completo_projetado[col_data] > data_limite_hist, [conversoes[i]]].values*\
      (df_completo_projetado.loc[df_completo_projetado[col_data] > data_limite_hist, ['%__Volume Aberta']].values)

      # Caso a conta resulte em algum número >100% ou <0, vamos remover aqui:
      valores = df_completo_projetado[cohorts_projetadas[i]].values
      valores = np.where(valores<0,0,valores)
      valores = np.where(valores>limite_proj,limite_proj,valores)
      df_completo_projetado[cohorts_projetadas[i]] = valores
      

    # Caso seja uma abertura clusterizada, vamos forçar a existência do share de 100% da última,
    # assim garantimos que a cohort aberta vai ser a definida pelo cluster, já que os shares
    # das fechadas são médias que podem não bater 100% a aberta definida.
    elif tipo_de_projecao == 'Cluster' and i == len(cohorts_projetadas)-1:


      df_completo_projetado.loc[df_completo_projetado[col_data] > data_limite_hist, [cohorts_projetadas[i]]] =\
      1*\
      (df_completo_projetado.loc[df_completo_projetado[col_data] > data_limite_hist, ['%__Volume Aberta']].values-\
      np.transpose(np.array([df_completo_projetado.loc[df_completo_projetado[col_data] > data_limite_hist, cohorts_projetadas[0:i]].sum(axis=1).values])))

      # Caso a conta resulte em algum número >100% ou <0, vamos remover aqui:
      valores = df_completo_projetado[cohorts_projetadas[i]].values
      valores = np.where(valores<0,0,valores)
      valores = np.where(valores>limite_proj,limite_proj,valores)

      df_completo_projetado[cohorts_projetadas[i]] = valores

    else:
    #elif cohorts_projetadas[i] != '%__Coincident':  Não projetamos mais a cohort coincident via share

      df_completo_projetado.loc[df_completo_projetado[col_data] > data_limite_hist, [cohorts_projetadas[i]]] =\
      df_completo_projetado.loc[df_completo_projetado[col_data] > data_limite_hist, [conversoes[i]]].values*\
      (df_completo_projetado.loc[df_completo_projetado[col_data] > data_limite_hist, ['%__Volume Aberta']].values-\
      np.transpose(np.array([df_completo_projetado.loc[df_completo_projetado[col_data] > data_limite_hist, cohorts_projetadas[0:i]].sum(axis=1).values])))

      # Caso a conta resulte em algum número >100% ou <0, vamos remover aqui:
      valores = df_completo_projetado[cohorts_projetadas[i]].values
      valores = np.where(valores<0,0,valores)
      valores = np.where(valores>limite_proj,limite_proj,valores)
      df_completo_projetado[cohorts_projetadas[i]] = valores

    #print(df_completo_projetado.loc[df_completo_projetado[col_data] >= data_limite_hist, [col_data,cohorts_projetadas[i],conversoes[i],'%__Volume Aberta']])
    # Não projetamos mais a cohort coincident via share
    '''
    else:
      df_completo_projetado.loc[df_completo_projetado[col_data] >= primeira_data_forecast, [cohorts_projetadas[i]]] =\
      df_completo_projetado.loc[df_completo_projetado[col_data] >= primeira_data_forecast, [conversoes[i]]].values*\
      (df_completo_projetado.loc[df_completo_projetado[col_data] >= primeira_data_forecast, ['%__Volume Aberta']].values-\
      np.transpose(np.array([df_completo_projetado.loc[df_completo_projetado[col_data] >= primeira_data_forecast, cohorts_projetadas[0:i-1]].sum(axis=1).values])))
    '''

  #if tipo_de_projecao == 'Cluster':
    #print(df_completo_projetado[['Etapa','Volume','%__Volume Aberta','0']])
  return df_completo_projetado,out_parametros,matriz_forecast,base_outliers,base_contencao_de_danos
