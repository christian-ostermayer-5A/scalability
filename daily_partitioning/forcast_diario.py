#@title Def forcast_diario
import pandas as pd

def forcast_diario(df_out_forecast,
                  df_share_diario,
                   df_diario,
                   primeira_data_forecast,
                   etapas_coh,
                   etapas_vol,
                   etapas_share):

  cb = list(df_out_forecast.columns.values)

  cb_d = list(df_diario.columns.values)
  col_data_date = cb_d[0]

  aberturas = cb[1:cb.index('Week Origin')]

  col_data_week = cb[0]

  out_coincident = df_out_forecast.loc[df_out_forecast['Week Origin'] != 'Coincident']
  out_coincident = out_coincident.groupby([col_data_week]+aberturas, as_index=False)[etapas_coh+[etapas_vol[-1]]].sum()

  out_coincident[col_data_week] = pd.to_datetime(out_coincident[col_data_week])

  out_coincident[col_data_date] = out_coincident[col_data_week].values

  aux = out_coincident.copy()
  lista_out_coincident = [out_coincident]
  for d in range(1,7):
    aux[col_data_date] = aux[col_data_week].values+pd.Timedelta(d, unit='D')
    lista_out_coincident = lista_out_coincident + [aux.copy()]
  out_coincident = pd.concat(lista_out_coincident)

  out_coincident['WeekDay'] = out_coincident[col_data_date].dt.dayofweek

  out_coincident = pd.merge(out_coincident,df_share_diario,how='left',on=aberturas+['WeekDay'])

  out_coincident[etapas_vol] = out_coincident[etapas_coh+[etapas_vol[-1]]].values*out_coincident[etapas_share].values

  out_coincident = out_coincident.loc[out_coincident[col_data_date] >= primeira_data_forecast]

  df_diario = df_diario.loc[df_diario[col_data_date] < primeira_data_forecast]

  out_coincident = pd.concat([df_diario,out_coincident])

  # Vamos inserir as colunas de mês, quarter, halfyear e year:

  out_coincident['month'] = pd.DatetimeIndex(out_coincident[col_data_date]).month

  out_coincident['halfyear'] = 0
  out_coincident.loc[out_coincident['month'] <= 6,'halfyear'] = 1
  out_coincident.loc[out_coincident['month'] > 6,'halfyear'] = 2

  out_coincident['quarter'] = out_coincident[col_data_date].dt.quarter

  out_coincident['year'] = pd.DatetimeIndex(out_coincident[col_data_date]).year

  out_coincident = out_coincident[[col_data_date,col_data_week,'month','halfyear','quarter','year']+aberturas+etapas_vol]


  return out_coincident
