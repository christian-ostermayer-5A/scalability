#@title Def Fitness Function (AG 4)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
'''
Essa função serve para retornar um número relacionado à qualidade
do ajuste multilinear dado um grupo de séries exógenas e lags da
endógena na forma de genes para o algoritmo genético otimizar.

"endog_fitness" é uma string com o nome da variável endógena, definido externamente antes de chamar essa função.

"df_fitness" é um DataFrame definido externamente antes de chamar essa função.
ele deve ser o histórico maturado da variável endógena a ser testada, juntamente
com todas as séries exógenas possíveis já transformadas além das endógenas possíveis já transformadas,
limpa, sem qualquer valor inf ou NaN.

"vetor_exogs_fitness" vetor com o nome de todas as séries exógenas possíveis e transformadas, definido externamente antes de chamar essa função

"pcf_fitness" é o número de lags referente ao resultado da função de auto correlação, definido externamente antes de chamar essa função

"max_lags_fitness" é o número de lags máximo pré-estabelecido, definido externamente antes de chamar essa função
'''

def fitness_func_qualidade_do_modelo(ga_instance,
                                      solution, # Lista de genes com inteiros que vairiam entre 0 e 2.
                                      solution_idx): # índice das soluções (não é utilizado)


    # Primeiramente, vamos definir quais são as séries exógenas a serem usadas com base no vetor
    # da solução proposta pelos genes (a primeira posição é sobre o lag da endógena):
    posi_series = np.where(np.array(solution[1:]) != 0)[0]
    exogs_fitness = list(vetor_exogs_fitness[posi_series])
    fitness = 0
                                  
    # Pode acontecer do cromossomo retornar nenhuma série exógena. Nesse caso o modelo vai
    # retornar um erro. Para evitar esse problema, checamos se as bases não estão vazias
    # antes de treinar o modelo. Se estiverem, retornamos "fitness" = 0:
    if len(exogs_fitness) == 0:
      #fitness = 0
      fitness = -10000

    else:


      # Vamos definir os lags da série endógena:
      endog_lag = []
      if solution[0] == 0:
        endog_lag = []
      elif solution[0] == 1:
        for l in range(1,pcf_fitness+1):
          endog_lag = endog_lag + [endog_fitness+"___l___"+str(l)]
      else:
        for l in range(1,max_lags_fitness+1):
          endog_lag = endog_lag + [endog_fitness+"___l___"+str(l)]

      # A lista final de séries exógenas é a soma das endógenas transformadas com as exógenas:
      exogs_fitness = exogs_fitness + endog_lag

      # Vamos adicionar as exógenas obrigatórias e removê-las caso estejam duplicadas:
      exogs_fitness = exogs_fitness + exogs_obrigatorias_fitness
      exogs_fitness = list(dict.fromkeys(exogs_fitness)) # remove duplicadas
      if len(exogs_fitness) == 0:
        exogs_fitness = exogs_obrigatorias_fitness

      df_fitness[exogs_fitness] = df_fitness[exogs_fitness].replace(np.inf, 0, inplace=False)
      df_fitness[exogs_fitness] = df_fitness[exogs_fitness].replace(-np.inf, 0, inplace=False)
      df_fitness[exogs_fitness] = df_fitness[exogs_fitness].replace(np.nan, 0, inplace=False)

      # Definimos o modelo linear
      model = LinearRegression(fit_intercept=fit_intercept)

      X_train = df_fitness[exogs_fitness]

      y_train = df_fitness[[endog_fitness]]

      model.fit(X_train, y_train)

      # Verificamos o sinal dos coeficientes e excluímos os modelos que não fazem sentido para o negócio:
      # Retornamos os parâmetros treinados do modelo:
      if len(df_sanity_check_fitness) > 0:
        parametros = parametros_modelo(modelo = model,
                                        X = X_train,
                                        y = y_train,
                                        pivot = False,
                                        erro = False)
        parametros['exógenas_originais'] = parametros.index
        parametros = parametros.reset_index()
        parametros['exógena'] = parametros['exógenas_originais'].apply(lambda x: x.split('___')[0])
        parametros['exógena'] = parametros['exógena'].apply(lambda x: x.split('_t')[0])
        parametros['slope'] = np.where(parametros['slope'] >= 0, 1, -1)
        sanity_check_merged = pd.merge(parametros,df_sanity_check_fitness,how='left',on='exógena')
        sanity_check_merged['sanity check'] = sanity_check_merged['slope'].values + sanity_check_merged['slope sanity check'].values
        sanity_check_merged['sanity check'] = sanity_check_merged['sanity check'].fillna(1)
        sanity_check_merged = sanity_check_merged.loc[sanity_check_merged['sanity check'] == 0]
        if len(sanity_check_merged) > 0:
          fitness = -20000

      if fitness > -10000:

        # Vamos definir o valor a ser otimizado como sendo o R2 ajustado do modelo:
        #r2_ajustado = 1 - (1-model.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
        r2_ajustado = model.score(X_train, y_train)
        n_series = len(exogs_fitness)
        fitness = 2*np.log(r2_ajustado) - overfitting_vs_underfitting_fitness * n_series
        if np.isnan(fitness):
          fitness = -30000

      #fitness = 1 - (1-model.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)

    return fitness
