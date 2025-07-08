#@title Def outliers_detector
from scipy.spatial import distance
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

'''
Detectção multivariada de outliers.

O método "CorrelationDistance" considera as variáveis exógenas como sendo as distâncias euclidianas
entre a variável endógena normalizada e a exógena normalizada, invertida caso a correlação entre ambas seja
negativa. Assim, mesmo se um ponto endógeno apresentar um valor extremo, se ele apresentar o mesmo
padrão de distanciamento com relação às outras variáveis, ele não será conseiderado um outlier.

Utilizamos a distância relativa ao invés dos valores absolutos das variáveis exógenas pelo motivo de
que ambas as variáveis apresentarem valores extremos, elas estariam isoladas e poderiam ser consideradas outliers

Nesse método, não comparamos a variável endógena com as exógenas. Só comparamos as distâncias relativas
entre elas mesmas.
'''

def outliers_detector(train_data,endogenous,exogenous,threshold,method):

  # Caso não exista variável exógena, o método será Mahalanobis simples
  if (method == 'CorrelationDistance_IsolationForest' or method == 'CorrelationDistance_Mahalanobis') and len(exogenous) == 0:
    method = 'Mahalanobis'

  # Criamos uma matriz que contem apenas os valores numéricos das variáveis endógenas e exógenas
  df = train_data[[endogenous]+exogenous].astype(float).to_numpy(dtype=float)


  #-------------------------------------------------------------------------------------------------

  if method == 'Mahalanobis':

    if len(exogenous) == 0:
      # Covariance matrix
      covariance  = 1

      # Covariance matrix power of -1
      covariance_pm1 = 1
    else:
      # Covariance matrix
      covariance  = np.cov(df , rowvar=False)

      # Covariance matrix power of -1
      covariance_pm1 = np.linalg.matrix_power(covariance, -1)

    # Center point
    centerpoint = np.mean(df , axis=0)
    distances = []
    for i, val in enumerate(df):
      p1 = val
      p2 = centerpoint
      if len(exogenous) == 0:
        d = np.linalg.norm(p1-p2)
      else:
        d = distance.mahalanobis(p1, p2, covariance)
        #distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
      distances.append(d)

    distances = np.array(distances)

    # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
    #cutoff = chi2.ppf(1-threshold, df.shape[1])
    sorted_distances = np.sort(distances)
    index = int(np.round(((1-threshold)*(len(distances)-1)),0))
    cutoff = sorted_distances[index]

    # Index of outliers
    outlierIndexes = np.where(distances > cutoff )

  #-------------------------------------------------------------------------------------------------

  elif method == 'IsolationForest':

    model=IsolationForest(max_samples='auto', contamination=float(threshold),max_features=1.0)
    model.fit(df)
    outliers = model.predict(df)
    outlierIndexes = np.where(outliers == -1)

  #-------------------------------------------------------------------------------------------------

  elif method == 'CorrelationDistance_IsolationForest':

    new_df = df.copy()

    endog_normalizado = (df[:,0]-min(df[:,0]))/(-max(df[:,0])-min(df[:,0]))

    new_df[:,0] = endog_normalizado[:]

    for i in range(len(exogenous)):
      correl = np.correlate(df[:,0],df[:,i+1])
      # Caso a correlação seja negativa, invertemos os exogs antes de normalizar
      if correl < 0:
        exog_normalizado = 1/df[:,i+1]
        exog_normalizado = (exog_normalizado-min(exog_normalizado))/(-max(exog_normalizado)-min(exog_normalizado))
      else:
        exog_normalizado = (df[:,i+1]-min(df[:,i+1]))/(-max(df[:,i+1])-min(df[:,i+1]))

      distances = abs(endog_normalizado-exog_normalizado)

      new_df[:,i+1] = distances[:]

    # Verificamos apenas os outliers das distancias relativas, sem a variável endógena
    new_df = new_df[:,1:]

    where_are_NaNs = np.isnan(new_df)
    new_df[where_are_NaNs] = 0

    model=IsolationForest(max_samples='auto', contamination=float(threshold),max_features=1.0)
    model.fit(new_df)
    outliers = model.predict(new_df)
    outlierIndexes = np.where(outliers == -1)

  #-------------------------------------------------------------------------------------------------

  elif method == 'CorrelationDistance_Mahalanobis':

    new_df = df.copy()

    endog_normalizado = (df[:,0]-min(df[:,0]))/(-max(df[:,0])-min(df[:,0]))

    new_df[:,0] = endog_normalizado[:]

    for i in range(len(exogenous)):
      correl = np.correlate(df[:,0],df[:,i+1])
      if correl < 0:
        exog_normalizado = 1/df[:,i+1]
        exog_normalizado = (exog_normalizado-min(exog_normalizado))/(-max(exog_normalizado)-min(exog_normalizado))
      else:
        exog_normalizado = (df[:,i+1]-min(df[:,i+1]))/(-max(df[:,i+1])-min(df[:,i+1]))

      distances = abs(endog_normalizado-exog_normalizado)

      new_df[:,i+1] = distances[:]

    new_df = new_df[:,1:]

    if len(new_df[0,:]) <= 1:
      # Covariance matrix
      covariance  = 1

      # Covariance matrix power of -1
      covariance_pm1 = 1
    else:
      # Covariance matrix
      covariance  = np.cov(df , rowvar=False)

      # Covariance matrix power of -1
      covariance_pm1 = np.linalg.matrix_power(covariance, -1)

    # Center point
    centerpoint = np.mean(df , axis=0)

    distances = []
    for i, val in enumerate(df):
          p1 = val
          p2 = centerpoint
          if len(new_df[0,:]) <= 1:
            d = np.linalg.norm(p1-p2)
          else:
            d = distance.mahalanobis(p1, p2, covariance)
            #distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
          distances.append(d)
    distances = np.array(distances)

    # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
    #cutoff = chi2.ppf(1-threshold, df.shape[1])
    sorted_distances = np.sort(distances)
    index = int(np.round(((1-threshold)*(len(distances)-1)),0))
    cutoff = sorted_distances[index]

    # Index of outliers
    outlierIndexes = np.where(distances > cutoff )

  else:
    print('erro')

  return outlierIndexes[0]

#_________________________________________________________________________________________________

#@title Def remove_ouliers


def remove_ouliers(df,endogenous,exogenous,outlierIndexes):

  df_sem_outlier = df.copy()

  outliers = df_sem_outlier.iloc[outlierIndexes,:]
  outliers[[endogenous]+exogenous] = np.nan
  df_sem_outlier.iloc[outlierIndexes,:] = outliers.values

  # Aqui também vamos remover os shares da W0 que estiverem zerados
  if endogenous == 's__0':
    df_sem_outlier_copy = df_sem_outlier.copy()
    df_sem_outlier_copy.loc[df_sem_outlier_copy['s__0'] == 0,[endogenous]+exogenous] = np.nan
    # Mas se todos os shares estiverem zerados,mantemos zerados
    if len(df_sem_outlier_copy.loc[df_sem_outlier_copy['s__0'].isnull(),[endogenous]+exogenous]) < len(df_sem_outlier[[endogenous]+exogenous]):
      df_sem_outlier = df_sem_outlier_copy

  return df_sem_outlier

#_________________________________________________________________________________________________

#@title Def interpol_NaN

def interpol_NaN(df,col_data,col_valores):

  colunas = df.columns.values

  interpol = df.copy()
  #print(interpol[[col_data]+col_valores])

  interpol[col_data] = pd.to_datetime(interpol[col_data], infer_datetime_format=True)
  interpol[col_valores] = interpol[col_valores].astype(float).interpolate(method='linear', limit_direction='both', axis=0)

  return interpol

#_________________________________________________________________________________________________

 #@title Def aplica_removedor_outliers

def aplica_removedor_outliers(df,
                              endogenous,
                              exog_list,
                              threshold,
                              col_data,
                              method):

  # Remover outliers
  teste_outliers = True
  exog_list = [] # @ Aqui: ainda temos que testar os casos com exógenas melhor


  if len(exog_list) != 0:

    tipo_de_remocao = "CorrelationDistance_"+method

    try:
      outliers = outliers_detector(train_data = df,
                                  endogenous = endogenous,
                                  exogenous = exog_list,
                                  threshold = threshold,
                                  method = tipo_de_remocao)
    except:
      teste_outliers = False
      outliers = []


  else:
    tipo_de_remocao = "CorrelationDistance_"+method
    try:
      outliers = outliers_detector(train_data = df,
                                  endogenous = endogenous,
                                  exogenous = exog_list,
                                  threshold = threshold,
                                  method = tipo_de_remocao)
    except:
      teste_outliers = False
      outliers = []

  # Existem casos onde ele reconheceu todos os pontos como sendo
  # outliers
  if len(outliers) < len(df)/2 and len(outliers)>0:

    df = remove_ouliers(df = df,
                        endogenous = endogenous,
                        exogenous = exog_list,
                        outlierIndexes = outliers)

    df = interpol_NaN(df = df,
                      col_data = col_data,
                      col_valores = [endogenous]+exog_list)



  df.dropna(axis=0, how='any', inplace=True)

  #print(df.iloc[outliers,:])

  # Vamos salvar uma base contendo apenas os outliers removidos e os valores interpolados:
  col_base_outlier = list(df.columns.values[:list(df.columns.values).index('Volume')])+[endogenous]
  base_outliers = df.iloc[outliers,:][col_base_outlier]
  base_outliers['Métrica'] = endogenous
  base_outliers = base_outliers.rename(columns={endogenous:'Valor Interpolado'})
  col_base_outlier = list(df.columns.values[:list(df.columns.values).index('Volume')])+['Métrica','Valor Interpolado'] # reordenando as colunas
  base_outliers = base_outliers[col_base_outlier]


  return df,teste_outliers,base_outliers

