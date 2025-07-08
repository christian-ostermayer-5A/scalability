#@title Def Auxiliar_Forecast (fdf 3)
import pandas as pd
import numpy as np
from multilinear import *
from projeta_por_media import projeta_por_media
from outliers_detector import *
from inputs_exogenas import inputs_exogenas
from acf_pac import *
from transforma_dummy import transforma_dummy
from transforma_exogs_2 import *
from contencao_de_danos import *
'''
Descrição Geral:

# Recebe base filtrada na etapa e abertura e na endog

# Caso exista uma base de parametros já calculada, vamos definir a lista de séries exógenas (incluindo as endógenas com lag) a partir dessa base:
# Incluimos ou removemos series exogenas com base nos inputs de series exogenas:
# Vamos transformar as exógenas dummy
# Vamos transformar as colunas l_d

# Se o tipo de projeção for média:
  aplica_removedor_outliers
  projeta_por_media

# Se for Multilinear ou Treinada:
  Se não tiver lag_list treinado, calcula via autocorrelação
  _______________
  (fdf 4) multilinear
  _______________

# contencao_de_danos

return:
 df_completo_projetado,
 out_parametros

'''
def Auxiliar_Forecast(df_completo, # DF filtrado somente a abertura e Endog, ordenado com a data mais antiga no topo
                      df_inputs_exogenas,
                      df_parametros,
                      abertura,
                      etapa,
                      col_data,
                      maturacao,
                      data_corte,  # DateTime indicando onde acaba o historico
                      primeira_data_forecast,
                      endogenous,   # String com o nome da variável endog
                      tipo_de_projecao,
                      qtd_semanas_media,
                      max_origin,
                      remover_outliers,
                      estimativa_outliers,
                      retorna_parametros,
                       premissa_dummy,
                       aberta_zerada,
                       share_w0_zerado,
                       limite_share,
                       limite_delta_vol,
                       limite_delta_share,
                       limite_delta_aberta,
                       fit_intercept,
                     limite_proj = 1):

  df_completo_projetado = df_completo.copy()


  cb_df = list(df_completo_projetado.columns.values)
  aberturas = cb_df[1:cb_df.index('Etapa')]

  out_parametros = pd.DataFrame(columns = aberturas+['Endógena'])

  matriz_forecast = pd.DataFrame()
  base_outliers = pd.DataFrame()

  # Caso a endógena seja a última cohort, não vamos realizar nenhuma projeção. Vamos apenas fixar o share
  # em 100%
  if endogenous != 's__'+str(max_origin) and endogenous != 's__'+str(maturacao):

    # Caso exista uma base de parametros já calculada, vamos definir a lista de séries
    # exógenas (incluindo as endógenas com lag) a partir dessa base:
    #-------------------------------------------------------------------------------------------------
    # Verificamos a existência da base e também se a variável endógena não é a última cohort fechada,
    # pois essa não tem modelo. Seu share é sempre 100%
    if len(df_parametros) > 0 and tipo_de_projecao == "Treinado":

      # Filtramos a variável endógena na tabela de parâmetros
      df_parametros_filtrados = df_parametros.loc[df_parametros['Endógena'] == endogenous]

      # Caso exista um modelo para a métrica, podemos prosseguir com as definições. (Pode não existir
      # modelo devido à todas as séries exógenas propostas terem ficado menos relevantes que as aleatórias )
      if len(df_parametros_filtrados) > 0:
        exog_list = list(df_parametros_filtrados['Exógena'].values)

        exog_list_endog = exog_list.copy()
        exog_list_endog = [l for l in exog_list if endogenous in l]
        lag_list = [int(l.split('___l___')[1]) for l in exog_list_endog]

        tipo_de_projecao_local = tipo_de_projecao

      else:
        df_parametros_filtrados =[]
        exog_list = []
        exog_list_endog = []
        lag_list = []

        tipo_de_projecao_local = 'Average'

    else:
      df_parametros_filtrados = []
      exog_list = []
      exog_list_endog = []
      lag_list = []

      tipo_de_projecao_local = tipo_de_projecao


    if len(df_completo_projetado) >0:


      # Incluimos ou removemos series exogenas com base nos inputs de series exogenas, caso já não estejam
      # pré-determinadas no modelo treinado. Além disso, definimos aqui o número de lags da endógena caso
      # o tipo de projeção seja multilinear:
      #-------------------------------------------------------------------------------------------------
      if tipo_de_projecao_local == 'Multilinear':
        exog_list = inputs_exogenas(df_inputs_exogenas = df_inputs_exogenas,
                                    abertura = abertura,
                                    etapa = etapa,
                                    endogenous = endogenous,
                                    exog_list = exog_list)

        # Caso não tenha o parâmetros de lag da variável endógena, vamos criar uma a partir da função
        # de autocorrelação. Caso a autocorrelação seja muito pequena, vamos usar o número de semanas médio
        if len(lag_list) == 0:
          ar,ma = acf_pac(df_completo_projetado.loc[df_completo_projetado[col_data]<=data_corte],endogenous)
          if ar < qtd_semanas_media:
            ar = qtd_semanas_media
          lag_list = list(range(1,ar+1))
        # Vamos checar se o número máximo de lags não é superior ao tamanho da base histórica mais o número mínimo de pontos para treinar o modelo:
        data_limite = np.min([primeira_data_forecast,data_corte])
        tamanho_historico = len(df_completo_projetado.loc[df_completo_projetado[col_data] <= data_limite,[endogenous]].values)
        #print(tamanho_historico,primeira_data_forecast,data_corte)
        if tamanho_historico <= max(lag_list) + qtd_semanas_media and tamanho_historico > qtd_semanas_media:
          lag_list = list(range(1,tamanho_historico-qtd_semanas_media+1))
        elif tamanho_historico <= max(lag_list) + qtd_semanas_media:
          lag_list = [1]


      #-------------------------------------------------------------------------------------------------


      # Vamos transformar as exógenas dummy:
      #-------------------------------------------------------------------------------------------------
      exog_dummy = [d for d in exog_list if "(dummy)" in d]
      exog_dummy = [d.split("___")[0] for d in exog_dummy] # selecionando apenas a série dummy sem outras transformações
      exog_dummy = [d.replace("(dummy)_t","(dummy)") for d in exog_dummy if "_t" in d] # Removemos o marador de transformação da exógena dummy, pois este estará presente se a exógena vier de uma lista de parâmetros treinados

      df_completo_projetado = transforma_dummy(df = df_completo_projetado, # Dataframe filtrado (somente uma etapa e uma abertura especifica, e sem as colunas das mesmas. Somente data e valores)
                                        exogenous_dummy = exog_dummy, # Lista com o nome das variaveis exogenas que são dummy
                                        endogenous = endogenous, # variável endógena
                                        n_avg = premissa_dummy, # número de pontos passados usados na média móvel da endógena
                                        col_data = col_data) # String com o nome da coluna de datas
      #-------------------------------------------------------------------------------------------------

      # Vamos transformar as colunas necessárias
      #-------------------------------------------------------------------------------------------------
      col_exogs_l_d = [l for l in exog_list if '___' in l]

      df_completo_projetado =  transforma_exogs_2(df = df_completo_projetado, # Dataframe filtrado de dados futuros
                                                  exogenous_l_d = col_exogs_l_d, # Lista com o nome das variaveis exogenas (com os suffixos "l" e "d")
                                                  col_data = col_data) # String com o nome da coluna de datas

      #-------------------------------------------------------------------------------------------------


    else:
      tipo_de_projecao_local = 'Average'



    # Vamos realizar as projeções de acordo com o modelo local determinado:
    #_________________________________________________________________________________________________
    if tipo_de_projecao_local == 'Average':

      # Vamos remover os outliers da série endógena com base nas séries exógenas:
      df_completo_historico = df_completo_projetado.loc[df_completo_projetado[col_data] <= data_corte]

      if remover_outliers != 'Não':
        df_completo_historico,flag,base_outliers =  aplica_removedor_outliers(df = df_completo_historico,
                                                                                    endogenous = endogenous,
                                                                                    exog_list = exog_list,
                                                                                    threshold = estimativa_outliers,
                                                                                    col_data = col_data,
                                                                                    method = remover_outliers)



        if not flag:
          mensagem = 'Não Removeu Outliers'
          print('Não Removeu Outliers')

      # Vamos projetar via média:
      qtd_semanas_projetadas = len(df_completo_projetado.loc[df_completo_projetado[col_data] > data_corte,[endogenous]].values)

      df_completo_projetado.loc[df_completo_projetado[col_data] > data_corte,[endogenous]] = projeta_por_media(df = df_completo_historico,
                                                                                                                endogenous = endogenous,
                                                                                                                qtd_semanas_projetadas = qtd_semanas_projetadas,
                                                                                                                qtd_semanas_media = qtd_semanas_media)


    # Vamos projetar via Multilinear ou modelo Treinado
    #-------------------------------------------------------------------------------------------------
    else:


      # Projetamos o share das cohorts fechadas que não são a W0 nem as últimas.
      # A projeção dessas cohorts obrigatoriamente leva em consideração o share da W0 caso não seja um modelo já treinado
      if endogenous != 's__'+str(max_origin) and endogenous != 's__0' and endogenous != '%__Volume Aberta' and endogenous != 'Volume' and tipo_de_projecao_local != 'Treinado':
        if 's__0' not in exog_list:
          if len(exog_list) == 0:
            exog_list = ['s__0']
          else:
            exog_list = exog_list+['s__0']


      # Realizamos a projeção multilinear

      df_completo_projetado,out_parametros,matriz_forecast,base_outliers =  multilinear(df_completo = df_completo_projetado, # DF filtrado somente a abertura e Endog, ordenado com a data mais antiga no topo
                                                                                              col_data = col_data,
                                                                                              data_corte = data_corte,  # DateTime indicando onde acaba o historico
                                                                                              primeira_data_forecast = primeira_data_forecast,
                                                                                              col_exogs = exog_list,   # Lista com o nome das colunas de exogs
                                                                                              col_endog = endogenous,   # String com o nome da variável endog
                                                                                              col_endog_exog = exog_list_endog, # lista com as colunas endógenas transformadas
                                                                                              lag_list = lag_list,    # Lista com inteiros indicando os lags da variável endog
                                                                                              diff = False,        # Booleano indicando se existe diferenciação da endog
                                                                                              parametros = df_parametros_filtrados, # DataFrame com etapa, abertura e métrica filtrada
                                                                                              remover_outliers = remover_outliers,
                                                                                              estimativa_outliers = estimativa_outliers,
                                                                                              retorna_parametros = retorna_parametros,
                                                                                              premissa_dummy = premissa_dummy,
                                                                                              fit_intercept = fit_intercept)



    #-------------------------------------------------------------------------------------------------



  # Após cada forecast, devemos verificar se não foram projetados valores muito absurdos:
  #_________________________________________________________________________________________________
  # Caso a endógena seja um share de cohort, devemos verificar e corrigir os valores a partir da primeira
  # data após a data de corte, que no caso vai ser a primeira data após a data de maturação:
  if 's_' in endogenous or endogenous == '%__Coincident':
    primeira_data = data_corte+pd.Timedelta(7,'D')
  else:
    primeira_data = primeira_data_forecast

  # Caso a endógena tenha sido a última cohort, vamos preencher aqui com os valores de 100%:
  if endogenous == 's__'+str(max_origin) or endogenous == 's__'+str(maturacao):
    df_completo_projetado.loc[df_completo_projetado[col_data] >= primeira_data,[endogenous]] = 1
    tipo_de_projecao_local = tipo_de_projecao
  # Caso a maturação seja antes da última cohort, vamos zerar o share da última:
  if int(max_origin) > int(maturacao):
    df_completo_projetado.loc[df_completo_projetado[col_data] >= primeira_data,['s__'+str(max_origin)]] = 0


  # Realizamos a contenção de danos:
  endog_projetada,mensagem,base_contencao_de_danos = contencao_de_danos(df = df_completo_projetado, # filtrado na etapa e abertura
                                                                        endogenous = endogenous,
                                                                        col_data = col_data,
                                                                        data_corte = data_corte,
                                                                        primeira_data_forecast = primeira_data,
                                                                        tipo_de_forecast = tipo_de_projecao_local,
                                                                        qtd_semanas_media = qtd_semanas_media,
                                                                        subs_aberta_zerada = aberta_zerada,
                                                                        subs_share_w0_zerado = share_w0_zerado,
                                                                        limite_inferior_share = limite_share,
                                                                        limite_delta_media_vol = limite_delta_vol,
                                                                        limite_delta_media_share = limite_delta_share,
                                                                        limite_delta_media_aberta = limite_delta_aberta,
                                                                       limite_proj = limite_proj)

  # Preenchemos a base transformada com a projeção corrigida
  df_completo_projetado.loc[df_completo_projetado[col_data] >= primeira_data,[endogenous]] = endog_projetada

  print(mensagem)




  # Adicionar nome da abertura e endógena no out_parametros:
  #_________________________________________________________________________________________________
  out_parametros[aberturas] = abertura
  out_parametros['Endógena'] = endogenous



  return df_completo_projetado,out_parametros,matriz_forecast,base_outliers,base_contencao_de_danos
