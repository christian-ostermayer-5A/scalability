#@title Def multilinear (fdf 4)
import timeit
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from outliers_detector import *
from transforma_exogs_2 import transforma_exogs_2
from parametros_modelo import parametros_modelo
from transforma_dummy import transforma_dummy
'''
Descrição Geral:

# Recebe DF filtrado somente a abertura e Endog, ordenado com a data mais antiga no topo

# aplica_removedor_outliers

# Caso o modelo não seja treinado:
  # transforma_exogs_2
  # treina modelo
  # corrige número de lags
  # Preenchemos a base transformada com a projeção de 1 semana

  # Para cada semana restante de projeção:
    # Excluímos as colunas dummy transformadas anteriormente
    # transforma_dummy
    # Excluímos as colunas ld transformadas anteriormente
    # transforma_exogs_2
    # y_forecast = model.predict(X_forecast)

# Para o caso onde já temos os modelos treinados:
  # transforma_exogs_2
  # Encontramos os valores dos parametros lineares do modelo já treinado
  # Realizamos a projeção multiplicando os coeficientes pelas séries de X_forecast
  # Preenchemos a base transformada com a projeção de 1 semana

  # Para cada semana restante de projeção:
    # Excluímos as colunas dummy transformadas anteriormente
    # transforma_dummy
    # Excluímos as colunas ld transformadas anteriormente
    # transforma_exogs_2
    # Realizamos a projeção multiplicando os coeficientes pelas séries de X_forecast


return:
 df_transformado,
 out_parametros

'''
def multilinear(df_completo, # DF filtrado somente a abertura e Endog, ordenado com a data mais antiga no topo
                col_data,
                data_corte,  # DateTime indicando onde acaba o historico
                primeira_data_forecast,
                col_exogs,   # Lista com o nome das colunas de exogs
                col_endog,   # String com o nome da variável endog
                col_endog_exog, # lista com as colunas endógenas transformadas
                lag_list,    # Lista com inteiros indicando os lags da variável endog
                diff,        # Booleano indicando se existe diferenciação da endog
                parametros, # DataFrame com os parâmetros do modelo
                remover_outliers,
                estimativa_outliers,
                retorna_parametros,
                 premissa_dummy,
                fit_intercept):


  #time_start = timeit.default_timer() # registramos o início da execução

  # Criamos uma cópia da base original que vai sofrer algumas transformações
  df_transformado = df_completo.copy()

  # Criamos a matriz forecast (que conterá a contribuição de cada exógena no forecast)
  matriz_forecast = pd.DataFrame()

  base_outliers = pd.DataFrame()

  # definimos a lista de séries exógenas que são "dummy"
  col_exogs_dummy = [x for x in col_exogs if "(dummy)" in x]
  col_exogs_dummy_l_d = [x for x in col_exogs if "(dummy)" in x and "___" in x]
  col_exogs_dummy_sem_t = [d.replace("_t","") for d in col_exogs_dummy if "_t" in d]
  col_exogs_dummy_originais = [d.split("___")[0] for d in col_exogs_dummy_sem_t] # selecionando apenas a série dummy sem outras transformações


  # Vamos criar colunas auxiliares que conterão a série dummy transformada
  col_exogs_dummy_t = [d.replace("(dummy)","(dummy)_t") for d in col_exogs_dummy_sem_t if "_t" not in d]
  col_exogs_dummy_originais_t = [d.replace("(dummy)","(dummy)_t") for d in col_exogs_dummy_originais if "_t" not in d]



  # Redefinimos as colunas exogenas para excluir as dummy não transformadas e incluir as transformadas:
  col_exogs = list(set(col_exogs) - set(col_exogs_dummy_sem_t) - set(col_exogs_dummy_t)) + col_exogs_dummy_t


  # Criamos uma cópia da lista de séries exógenas
  col_exogs_originais = col_exogs.copy()

  # Vamos remover os outliers da série endógena com base nas séries exógenas:
  df_transformado.dropna(axis=0, how='any', inplace=True)

  #df_transformado = df_transformado.fillna(0)

  df_transformado_historico = df_transformado.loc[df_transformado[col_data] <= data_corte]

  if len(df_transformado_historico) == 0:
    df_transformado = df_completo.copy()
    out_parametros = parametros

  else:

    if remover_outliers != 'Não':
      # Para o caso do tipo de projeção com modelo já treinado, as series exogenas já incluem as
      # endog transformadas. Precisamos remover essas da lista para remover os outliers somente
      # com base nas exógenas de forma a deixar todos os modelos iguais:
      col_exogs_outliers = [o for o in col_exogs if col_endog not in o]

      df_sem_outliers,flag,base_outliers =  aplica_removedor_outliers(df = df_transformado_historico,
                                                        endogenous = col_endog,
                                                        exog_list = col_exogs_outliers,
                                                        threshold = estimativa_outliers,
                                                        col_data = col_data,
                                                          method = remover_outliers)

      df_transformado.loc[(df_transformado[col_data] <= data_corte)] = df_sem_outliers



      if not flag:
        mensagem = "Não removeu outliers"
        #print("Não removeu outliers")



    # Caso não exista nenhuma lista de parâmetros salva, vamos treinar e aplicar o modelo
    # aqui mesmo:
    ##################################################################################################
    if len(parametros) == 0:


      # Definimos o modelo linear
      model = LinearRegression(fit_intercept=fit_intercept)

      # Para cada inteiro na lista "lag_list" vamos adicionar à lista de variáveis exógenas
      # transformadas a variável endógena com o lag indicado. Além disso, vamos adicionar a variável
      # endógena diferenciada, caso o "diff" = True indique
      colunas_endog_transformadas = []
      if len(lag_list) != 0:
        for l in lag_list:
          colunas_endog_transformadas = colunas_endog_transformadas + [col_endog+"___l___"+str(l)]
      if diff:
        colunas_endog_transformadas = colunas_endog_transformadas + [col_endog+"___d"]

      # Salvamos uma cópia das colunas transformadas originais, pois as mesmas podem mudar quando recalcularmos
      # o lag com base na variação dos parâmetros. Precisamos saber a totalidade de colunas que foram modificadas
      # para poder excluir as mesmas da base final e não acumular NaN's a cada endógena que projetamos.
      colunas_endog_transformadas_originais = colunas_endog_transformadas.copy()

      # Antes de adicionar as colunas endogenas transformadas na base completa, vamos verificar se elas
      # já não foram calculadas previamente:
      colunas_endog_transformadas_faltantes = list(set(colunas_endog_transformadas+col_exogs) - set(list(df_transformado.columns.values)))


      # Adicionamos os endogs transformados na base geral
      df_transformado = transforma_exogs_2(df = df_transformado,
                                        exogenous_l_d = colunas_endog_transformadas_faltantes, # Lista com o nome das variaveis exogenas (com os suffixos "l" e "d")
                                        col_data = col_data) # String com o nome da coluna de datas


      # Criamos uma cópia da base transformada que vai ter os 'NaN's removidos e ser usada no modelo.
      # Não removemos na própria base pois vamos repetir o processo de projeção várias vezes. A cada vez
      # que removemos os NaN's a base vai diminuindo. Queremos que seja a mesma inicial toda vez:
      df_limpa = df_transformado.copy()

      # Removemos os NaN's onde não havia histórico para o lag ou diff
      df_limpa.dropna(axis=0, how='any', inplace=True)
      df_limpa[col_exogs] = df_limpa[col_exogs].replace(np.inf, 0, inplace=False).values
      df_limpa[col_exogs] = df_limpa[col_exogs].replace(-np.inf, 0, inplace=False).values
      data_limpa = df_limpa[col_data].min()


      # Definimos o total de colunas de variáveis exógenas juntamente com as endógenas transformadas
      col_exogs = col_exogs_originais + colunas_endog_transformadas

      # Definimos as séries "X" e "y" para treinar o modelo com base na data de corte do histórico
      '''
      print("---------------------------------------")
      print(colunas_endog_transformadas)
      print("---------------------------------------")
      print(colunas_endog_transformadas_faltantes)
      print("---------------------------------------")
      print(np.sort(df_limpa.columns.values))
      print("---------------------------------------")
      '''
      X_train = df_limpa.loc[df_limpa[col_data] <= data_corte,col_exogs]

      y_train = df_limpa.loc[df_limpa[col_data] <= data_corte,[col_endog]]

      # Treinamos o modelo com os dados selecionados
      '''
      print(col_endog)
      print(data_corte)
      print("-----df_trans---------")
      print(df_transformado)
      print("-----df_limpa---------")
      print(df_limpa)

      #if col_endog == 's__0':
      print(col_endog)
      print(data_corte)
      print(lag_list)
      for z in col_exogs+[col_endog]:
        print(z,df_limpa[z].max())
      '''

      #print(df_limpa.loc[df_limpa[col_data] <= data_corte,[col_data]+col_exogs])
      #print(df_limpa.loc[df_limpa[col_data] <= data_corte,[col_data,col_endog]])

      model.fit(X_train, y_train)


      # Caso os parâmetros retornados pelo modelo nos lags estejam oscinando muito,
      # vamos refazer o modelo com menos lags
      #_______________________________________________________________________________________________

      parametros = parametros_modelo(modelo = model,
                                    X = X_train,
                                    y = y_train,
                                    pivot = False,
                                    erro = retorna_parametros)

      lag_slope = parametros.loc[colunas_endog_transformadas,['slope']].values
      delta_lag_slope = lag_slope[:-1] - lag_slope[1:]

      delta_maximo_coefs = 1

      posi_coef_ruins = np.where(np.abs(delta_lag_slope) > delta_maximo_coefs)[0]


      if len(posi_coef_ruins) > 0:

        max_lag = min(posi_coef_ruins)

        if max_lag != max(lag_list):

          lag_list = list(range(1,max_lag+1))
          if len(lag_list) == 0:
            lag_list = [1]


          # Definimos o modelo linear
          model = LinearRegression(fit_intercept=fit_intercept)

          colunas_endog_transformadas = []
          for l in lag_list:
            colunas_endog_transformadas = colunas_endog_transformadas + [col_endog+"___l___"+str(l)]


          # Definimos o total de colunas de variáveis exógenas juntamente com as endógenas transformadas
          col_exogs = col_exogs_originais + colunas_endog_transformadas

          # Definimos as séries "X" e "y" para treinar o modelo com base na data de corte do histórico

          X_train = df_limpa.loc[df_limpa[col_data] <= data_corte,col_exogs]

          y_train = df_limpa.loc[df_limpa[col_data] <= data_corte,[col_endog]]

          '''
          print("__________________________________")
          print("-----Train-------")
          print(col_endog)
          print(data_corte)
          print(lag_list)
          print(df_limpa.loc[df_limpa[col_data] <= data_corte,[col_data]+col_exogs])
          print(df_limpa.loc[df_limpa[col_data] <= data_corte,[col_data,col_endog]])
          '''
          # Treinamos o modelo com os dados selecionados
          model.fit(X_train, y_train)

          parametros = parametros_modelo(modelo = model,
                                        X = X_train,
                                        y = y_train,
                                        pivot = False,
                                        erro = retorna_parametros)


      out_parametros = parametros

      out_parametros['Exógena'] = out_parametros.index.values


      # Projetamos a variável endog com base nas séries exógenas futuras selecionadas
      #_______________________________________________________________________________________________

      # Caso a variável endógena seja algum share, devemos projetá-la a partir da tada mais antiga entre
      # a data de corte ou início da projeção, pois precisamos garantir que os shares estejam projetados
      # no período do histórico não maturado:
      if "s__" in col_endog:
        if primeira_data_forecast > data_corte:
          X_forecast = df_limpa.loc[df_limpa[col_data] > data_corte,col_exogs].astype(float)
        else:
          X_forecast = df_limpa.loc[df_limpa[col_data] >= primeira_data_forecast,col_exogs].astype(float)
      else:
        X_forecast = df_limpa.loc[df_limpa[col_data] >= primeira_data_forecast,col_exogs].astype(float)


      try:
        y_forecast = model.predict(X_forecast)
      except:
        y_forecast = [0]



      # Preenchemos a base transformada com a projeção
      try:
        if "s__" in col_endog:
          if primeira_data_forecast > data_corte:
            df_transformado.loc[df_transformado[col_data] > data_corte,[col_endog]] = y_forecast
          else:
            df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = y_forecast
        else:
          df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = y_forecast

      except:
        if "s__" in col_endog:
          if primeira_data_forecast > data_corte:
            df_transformado.loc[df_transformado[col_data] > data_corte,[col_endog]] = 0
          else:
            df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = 0
        else:
          df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = 0


      #_______________________________________________________________________________________________
      # Para cada ponto de projeção, precisamos projetar toda a série novamente e ir atualizando
      # as colunas de séries exógenas. Fazemos isso pois dentre as colunas de séries exógenas está
      # a variável endógena com lag ou diferenciação. Como precisamos da projeção futura para saber qual é
      # o valor do lag, precisamos repetir a projeção para cada ponto novo projetado. Assim, repetimos
      # o processo acima um número de vezes até que todos os ponto sejam projetados corretamente
      for f in range(1,len(y_forecast)): # O primeiro ponto já foi projetado

        # Excluímos as colunas dummy transformadas anteriormente, pois essas serão substituídas a cada novo ponto projetado
        df_transformado = df_transformado.drop(columns=col_exogs_dummy_originais_t)

        # recalculamos as exogenas dummy com base no forecast da endog do ponto anterior
        df_transformado = transforma_dummy(df = df_transformado, # Dataframe filtrado (somente uma etapa e uma abertura especifica, e sem as colunas das mesmas. Somente data e valores)
                                          exogenous_dummy = col_exogs_dummy_originais, # Lista com o nome das variaveis exogenas que são dummy
                                          endogenous = col_endog, # variável endógena
                                          n_avg = premissa_dummy, # número de pontos passados usados na média móvel da endógena
                                          col_data = col_data) # String com o nome da coluna de datas

        # Excluímos as colunas ld transformadas anteriormente, pois essas serão substituídas a cada novo ponto projetado
        df_transformado = df_transformado.drop(columns=colunas_endog_transformadas+col_exogs_dummy_l_d)

        df_transformado = transforma_exogs_2(df = df_transformado, # Dataframe filtrado de dados futuros
                                                exogenous_l_d = colunas_endog_transformadas+col_exogs_dummy_l_d, # Lista com o nome das variaveis exogenas (com os suffixos "l" e "d")
                                                col_data = col_data) # String com o nome da coluna de datas

        # Criamos uma cópia da base transformada que vai ter os 'NaN's removidos e ser usada no modelo.
        # Não removemos na própria base pois vamos repetir o processo de projeção várias vezes. A cada vez
        # que removemos os NaN's a base vai diminuindo. Queremos que seja a mesma inicial toda vez:
        df_limpa = df_transformado.copy()

        # Selecionamos apenas as colunas que importam
        df_limpa = df_limpa[[col_data]+[col_endog]+col_exogs]

        # Removemos os NaN's onde não havia histórico para o lag ou diff
        df_limpa.dropna(axis=0, how='any', inplace=True)
        df_limpa[col_exogs] = df_limpa[col_exogs].replace(np.inf, 0, inplace=False).values
        df_limpa[col_exogs] = df_limpa[col_exogs].replace(-np.inf, 0, inplace=False).values

        # Projetamos a variável endog com base nas séries exógenas futuras selecionadas

        # Caso a variável endógena seja algum share, devemos projetá-la a partir da data mais antiga entre
        # a data de corte ou início da projeção, pois precisamos garantir que os shares estejam projetados
        # no período do histórico não maturado:
        if "s__" in col_endog:
          if primeira_data_forecast > data_corte:
            X_forecast = df_limpa.loc[df_limpa[col_data] > data_corte,col_exogs].astype(float)
          else:
            X_forecast = df_limpa.loc[df_limpa[col_data] >= primeira_data_forecast,col_exogs].astype(float)
        else:
          X_forecast = df_limpa.loc[df_limpa[col_data] >= primeira_data_forecast,col_exogs].astype(float)

        y_forecast = model.predict(X_forecast)


        # Para compor a matriz com a participação de cada exógena, temos que
        # extrair os parâmetros e aplicar no x_train:
        #-----------------------------------------------------------------------------------------

        # Encontramos os valores dos parametros lineares do modelo já treinado:
        slope = out_parametros[['Exógena','slope']]
        slope['slope'] = slope['slope'].astype(float)
        slope = slope.pivot_table(columns='Exógena',values='slope').reset_index(inplace=False).fillna(0)[col_exogs]


        matriz_forecast = X_forecast.multiply( slope.values , axis='columns')

        # Preenchemos a base transformada com a projeção
        if "s__" in col_endog:
          if primeira_data_forecast > data_corte:
            df_transformado.loc[df_transformado[col_data] > data_corte,[col_endog]] = y_forecast
            matriz_forecast[col_data] = df_transformado.loc[df_transformado[col_data] > data_corte,[col_data]].values
          else:
            df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = y_forecast
            matriz_forecast[col_data] = df_transformado.loc[df_transformado[col_data] > primeira_data_forecast,[col_data]].values
        else:
          df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = y_forecast
          matriz_forecast[col_data] = df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_data]].values

        '''
        if col_endog == 's__0':
          print("-----Forecast-------")
          #print(primeira_data_forecast)
          #print(df_transformado)
          #print(df_limpa)
          print(df_transformado[['week_start',col_endog]])
        '''




    # Para o caso onde já temos os modelos treinados:
    ##################################################################################################
    else:

      out_parametros = parametros

      col_exogs = col_exogs_originais

      # Precisamos saber quais são as colunas endog que foram transformadas na base para poder atualiza-las
      # toda vez que projetamos um ponto novo
      colunas_endog_transformadas = col_endog_exog

      # Salvamos uma cópia das colunas transformadas originais, pois as mesmas podem mudar quando recalcularmos
      # o lag com base na variação dos parâmetros. Precisamos saber a totalidade de colunas que foram modificadas
      # para poder excluir as mesmas da base final e não acumular NaN's a cada endógena que projetamos.
      colunas_endog_transformadas_originais = colunas_endog_transformadas.copy()

      # Antes de adicionar as colunas endogenas transformadas na base completa, vamos verificar se elas
      # já não foram calculadas previamente:
      colunas_endog_transformadas_faltantes = list(set(colunas_endog_transformadas+col_exogs) - set(list(df_transformado.columns.values)))


      # Adicionamos os endogs transformados na base geral
      df_transformado = transforma_exogs_2(df = df_transformado, # Dataframe filtrado de dados futuros
                                              exogenous_l_d = colunas_endog_transformadas_faltantes, # Lista com o nome das variaveis exogenas (com os suffixos "l" e "d")
                                              col_data = col_data) # String com o nome da coluna de datas


      # Criamos uma cópia da base transformada que vai ter os 'NaN's removidos e ser usada no modelo.
      # Não removemos na própria base pois vamos repetir o processo de projeção várias vezes. A cada vez
      # que removemos os NaN's a base vai diminuindo. Queremos que seja a mesma inicial toda vez:
      df_limpa = df_transformado.copy()

      # Removemos os NaN's onde não havia histórico para o lag ou diff
      df_limpa.dropna(axis=0, how='any', inplace=True)
      df_limpa[col_exogs] = df_limpa[col_exogs].replace(np.inf, 0, inplace=False).values
      df_limpa[col_exogs] = df_limpa[col_exogs].replace(-np.inf, 0, inplace=False).values

      #-----------------------------------------------------------------------------------------------
      # Encontramos os valores dos parametros lineares do modelo já treinado:
      slope = parametros[['Exógena','slope']]
      slope['slope'] = slope['slope'].astype(float)
      slope = slope.pivot_table(columns='Exógena',values='slope').reset_index(inplace=False).fillna(0)[col_exogs]

      intercept = parametros['intercept'].astype(float).values[0]

      # Exógenas futuras selecionadas
      # Caso a variável endógena seja algum share, devemos projetá-la a partir da tada mais antiga entre
      # a data de corte ou início da projeção, pois precisamos garantir que os shares estejam projetados
      # no período do histórico não maturado:
      if "s__" in col_endog:
        if primeira_data_forecast > data_corte:
          X_forecast = df_limpa.loc[df_limpa[col_data] > data_corte,col_exogs].astype(float)
        else:
          X_forecast = df_limpa.loc[df_limpa[col_data] >= primeira_data_forecast,col_exogs].astype(float)
      else:
        X_forecast = df_limpa.loc[df_limpa[col_data] >= primeira_data_forecast,col_exogs].astype(float)

      # Realizamos a projeção multiplicando os coeficientes pelas séries de X_forecast:

      matriz_forecast = X_forecast.multiply( slope.values , axis='columns')
      y_forecast = matriz_forecast.sum(axis=1)
      y_forecast = y_forecast + intercept
      '''
      if col_endog == 'Volume':
        print("#########################")
        print(len(X_forecast.columns.values),len(slope.columns.values))
        print("________________________")
        print(slope)
        print("________________________")
        print(X_forecast)
        print("________________________")
        print(X_forecast.multiply( slope.values , axis='columns'))
        print("________________________")
        print(X_forecast.multiply( slope.values , axis='columns').sum(axis=1))
        print("________________________")
        print(y_forecast)
      '''

      # Preenchemos a base transformada com a projeção
      try:
        if "s__" in col_endog:
          if primeira_data_forecast > data_corte:
            df_transformado.loc[df_transformado[col_data] > data_corte,[col_endog]] = y_forecast
          else:
            df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = y_forecast
        else:
          df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = y_forecast

      except:
        if "s__" in col_endog:
          if primeira_data_forecast > data_corte:
            df_transformado.loc[df_transformado[col_data] > data_corte,[col_endog]] = 0
          else:
            df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = 0
        else:
          df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = 0

      #-----------------------------------------------------------------------------------------------


      #_______________________________________________________________________________________________
      # Para cada ponto de projeção, precisamos projetar toda a série novamente e ir atualizando
      # as colunas de séries exógenas. Fazemos isso pois dentre as colunas de séries exógenas está
      # a variável endógena com lag ou diferenciação. Como precisamos da projeção futura para saber qual é
      # o valor do lag, precisamos repetir a projeção para cada ponto novo projetado. Assim, repetimos
      # o processo acima um número de vezes até que todos os ponto sejam projetados corretamente
      for f in range(1,len(y_forecast)): # O primeiro ponto já foi projetado

        # Excluímos as colunas transformadas anteriormente, pois essas serão substituídas a cada novo ponto projetado
        df_transformado = df_transformado.drop(columns=col_exogs_dummy_originais_t)

        # recalculamos as exogenas dummy com base no forecast da endog do ponto anterior
        df_transformado = transforma_dummy(df = df_transformado, # Dataframe filtrado (somente uma etapa e uma abertura especifica, e sem as colunas das mesmas. Somente data e valores)
                                          exogenous_dummy = col_exogs_dummy_originais, # Lista com o nome das variaveis exogenas que são dummy
                                          endogenous = col_endog, # variável endógena
                                          n_avg = premissa_dummy, # número de pontos passados usados na média móvel da endógena
                                          col_data = col_data) # String com o nome da coluna de datas

        # Excluímos as colunas transformadas anteriormente, pois essas serão substituídas a cada novo ponto projetado
        df_transformado = df_transformado.drop(columns=colunas_endog_transformadas+col_exogs_dummy_l_d)

        df_transformado = transforma_exogs_2(df = df_transformado, # Dataframe filtrado de dados futuros
                                                exogenous_l_d = colunas_endog_transformadas+col_exogs_dummy_l_d, # Lista com o nome das variaveis exogenas (com os suffixos "l" e "d")
                                                col_data = col_data) # String com o nome da coluna de datas

        # Criamos uma cópia da base transformada que vai ter os 'NaN's removidos e ser usada no modelo.
        # Não removemos na própria base pois vamos repetir o processo de projeção várias vezes. A cada vez
        # que removemos os NaN's a base vai diminuindo. Queremos que seja a mesma inicial toda vez:
        df_limpa = df_transformado.copy()

        # Selecionamos apenas as colunas que importam
        df_limpa = df_limpa[[col_data]+[col_endog]+col_exogs]

        # Removemos os NaN's onde não havia histórico para o lag ou diff
        df_limpa.dropna(axis=0, how='any', inplace=True)
        df_limpa[col_exogs] = df_limpa[col_exogs].replace(np.inf, 0, inplace=False).values
        df_limpa[col_exogs] = df_limpa[col_exogs].replace(-np.inf, 0, inplace=False).values

        # Exógenas futuras selecionadas
        # Caso a variável endógena seja algum share, devemos projetá-la a partir da tada mais antiga entre
        # a data de corte ou início da projeção, pois precisamos garantir que os shares estejam projetados
        # no período do histórico não maturado:
        if "s__" in col_endog:
          if primeira_data_forecast > data_corte:
            X_forecast = df_limpa.loc[df_limpa[col_data] > data_corte,col_exogs].astype(float)
          else:
            X_forecast = df_limpa.loc[df_limpa[col_data] >= primeira_data_forecast,col_exogs].astype(float)
        else:
          X_forecast = df_limpa.loc[df_limpa[col_data] >= primeira_data_forecast,col_exogs].astype(float)



        # Realizamos a projeção multiplicando os coeficientes pelas séries de X_forecast:


        matriz_forecast = X_forecast.multiply( slope.values , axis='columns')
        y_forecast = matriz_forecast.sum(axis=1)
        y_forecast = y_forecast + intercept
        #matriz_forecast = matriz_forecast + intercept
        '''
        if col_endog == 'Volume' and f == len(y_forecast)-1:
          print("#########################")
          print(len(X_forecast.columns.values),len(slope.columns.values))
          print("slope________________________")
          print(slope)
          print("X_forecast________________________")
          print(X_forecast[:20])
          print("X_forecast.multiply________________________")
          print(X_forecast.multiply( slope.values , axis='columns')[:20])
          print("X_forecast.multiply.sum(________________________")
          print(X_forecast.multiply( slope.values , axis='columns').sum(axis=1)[:20])
          print("y_forecast________________________")
          print(y_forecast[:20])
          print("matriz_forecast________________________")
          print(matriz_forecast)
        '''
        # Preenchemos a base transformada com a projeção
        if "s__" in col_endog:
          if primeira_data_forecast > data_corte:
            df_transformado.loc[df_transformado[col_data] > data_corte,[col_endog]] = y_forecast
            matriz_forecast[col_data] = df_transformado.loc[df_transformado[col_data] > data_corte,[col_data]].values
          else:
            df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = y_forecast
            matriz_forecast[col_data] = df_transformado.loc[df_transformado[col_data] > primeira_data_forecast,[col_data]].values
        else:
          df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_endog]] = y_forecast
          matriz_forecast[col_data] = df_transformado.loc[df_transformado[col_data] >= primeira_data_forecast,[col_data]].values

        #---------------------------------------------------------------------------------------------



    # Retornamos a base transformada sem as colunas extra. Essa base contém toda a projeção
    df_transformado = df_transformado.drop(columns=colunas_endog_transformadas_originais)

    # Retornamos uma base com a abertura da contribuição de cada exógena na projeção
    if len(parametros) != 0 and len(matriz_forecast) > 0:
      matriz_forecast['Endógena'] = col_endog
      # Transformamos as colunas de exógenas em uma coluna só:
      col_originais_matrix = list(matriz_forecast.columns.values)
      col_exogs = list(set(col_originais_matrix) - set(['Endógena',col_data]))
      try:
        matriz_forecast = matriz_forecast.melt(id_vars=['Endógena',col_data], value_vars=col_exogs,var_name='Exógenas', value_name='Valor')
      except:
        matriz_forecast = pd.DataFrame()
    else:
      matriz_forecast = pd.DataFrame()

    '''
    time_9 = timeit.default_timer()
    print(' --> Loop ',round(time_9 - time_8,5)," s")
    '''
    #time_9 = timeit.default_timer()
    #print(' --> TOTAL ',round(time_9 - time_start,5)," s")

  return df_transformado,out_parametros,matriz_forecast,base_outliers

