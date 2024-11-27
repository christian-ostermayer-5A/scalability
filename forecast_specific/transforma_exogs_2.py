#@title Def transforma_exogs_2
import timeit
import numpy as np
'''
Essa função serve para realizar as transformações "laged" ou "diferenciated" na base historica e
base futura das variáveis exogenas que assim foram transformadas e identificadas e salvas no modelo
'''
def transforma_exogs_2(df, # Dataframe filtrado (somente uma etapa e uma abertura especifica, e sem as colunas das mesmas. Somente data e valores)
                     exogenous_l_d, # Lista com o nome das variaveis exogenas (com os suffixos "l" e "d")
                     col_data): # String com o nome da coluna de datas

  time_t = timeit.default_timer()

  if len(exogenous_l_d) == 0:
    return df
  else:

    c_df = df.copy()

    # Vamos checar se a coluna de datas já está ordenada:
    #-----------------------------------------------------------------------------------------------
    c_df['Teste'] = c_df[col_data].astype(int)
    teste = c_df['Teste'].astype(int).values[1:]-c_df['Teste'].astype(int).values[:-1]

    # Se não estiver, ordenamos pelas datas
    if c_df['Teste'].values[0] > c_df['Teste'].values[-1] or np.average(teste) != 604800000000000:
      c_df = c_df.sort_values(by=col_data)

    c_df = c_df.drop(columns='Teste')
    #-----------------------------------------------------------------------------------------------

    cb_df = c_df.columns.values


    # Calcular a variável transformada:
    for i in range(len(exogenous_l_d)):
      split = exogenous_l_d[i].split("___")
      if split[1] == "d":
        exogenous,lag,dif = split[0],0,1

        c_df[exogenous_l_d[i]] = np.append(np.array([0.]),c_df[exogenous].astype(float).values[1:]-c_df[exogenous].astype(float).values[:-1],0)

      elif split[1] == "l":
        exogenous,lag,dif = split[0],int(split[2]),0

        c_df[exogenous_l_d[i]] = c_df[exogenous].shift(periods=lag)

      elif split[1] == "log":
        exogenous = split[0]
        c_df[exogenous_l_d[i]] = np.log(c_df[exogenous].astype(float).values)

      else:
        exogenous = split[0]
        c_df[exogenous_l_d[i]] = c_df[exogenous].values

      # Substituir os NaN's pelos valores originais:
      c_df = c_df.reset_index(drop=True)
      df = df.reset_index(drop=True)
      nans = c_df[c_df[exogenous_l_d[i]].isnull()].index.values
      valores_originais = df.iloc[nans][exogenous].values
      c_df[exogenous_l_d[i]].iloc[nans] = valores_originais


    return c_df
