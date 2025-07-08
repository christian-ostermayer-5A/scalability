#@title Def projeta_tof_externo
import numpy as np
import pandas as pd
from contencao_de_danos import contencao_de_danos

def projeta_tof_externo(df_completo, # base completa sem estar filtrada
                        df_ToF_externo,
                        col_data,
                        aberturas,
                        primeira_data_forecast,
                        lista_de_etapas,
                        topos,
                        qtd_semanas_media,
                        limite_delta_media_vol):

  df_ToF_externo[topos] = df_ToF_externo[topos].astype(float)

  df_ToF_externo[col_data] = pd.to_datetime(df_ToF_externo[col_data], infer_datetime_format=True)

  df_ToF_externo = df_ToF_externo.groupby([col_data]+aberturas, as_index=False)[topos].sum()

  df_completo = pd.merge(df_completo,df_ToF_externo,how='left',on=[col_data]+aberturas)

  df_final = pd.DataFrame()

  base_contencao_de_danos = pd.DataFrame()


  for tof in topos:

    for e in lista_de_etapas:

      if tof == e.split('2')[0]:
        etapa_tof = e

    df_completo.loc[(df_completo[col_data] >= primeira_data_forecast) & (df_completo['Etapa'] == etapa_tof),['Volume']] = df_completo[tof]

    # Caso seja necessário, vamos aplicar a contenção de danos no ToF externo:
    if limite_delta_media_vol < 1000:
      df_ToF_externo['aux'] = 0
      lista_das_aberturas = df_ToF_externo.groupby(aberturas, as_index=False)['aux'].sum()
      df_ToF_externo = df_ToF_externo.drop(columns='aux')
      lista_das_aberturas = lista_das_aberturas[aberturas].to_numpy()
      lista_base_contencao_de_danos = [base_contencao_de_danos]
      lista_df_final = [df_final]

      for a in range(len(lista_das_aberturas[:,0])):

        df_completo_abertura_aux = df_completo.loc[df_completo['Etapa'] == etapa_tof]

        for i in range(len(aberturas)):

          df_completo_abertura_aux = df_completo_abertura_aux.loc[df_completo_abertura_aux[aberturas[i]] == lista_das_aberturas[a][i]]

        endog_projetada,mensagem,base_contencao_de_danos_local = contencao_de_danos(df = df_completo_abertura_aux, # filtrado na etapa e abertura
                                                                              endogenous = 'Volume',
                                                                              col_data = col_data,
                                                                              data_corte = primeira_data_forecast - pd.Timedelta(7, unit='D'),
                                                                              primeira_data_forecast = primeira_data_forecast,
                                                                              tipo_de_forecast = 'Input Externo',
                                                                              qtd_semanas_media = qtd_semanas_media,
                                                                              subs_aberta_zerada = 0,
                                                                              subs_share_w0_zerado = 0,
                                                                              limite_inferior_share = 0,
                                                                              limite_delta_media_vol = limite_delta_media_vol,
                                                                              limite_delta_media_share = 0,
                                                                              limite_delta_media_aberta = 0)


        # Adicionamos a informação da abertura e etapa na base de contenção de danos:
        if len(base_contencao_de_danos_local)>0:
          base_contencao_de_danos_local[aberturas] = lista_das_aberturas[a]
          base_contencao_de_danos_local['Etapa'] = etapa_tof

        lista_base_contencao_de_danos = lista_base_contencao_de_danos+[base_contencao_de_danos_local.copy()]

        df_completo_abertura_aux.loc[df_completo_abertura_aux[col_data] >= primeira_data_forecast,['Volume']] = endog_projetada

        lista_df_final = lista_df_final + [df_completo_abertura_aux.copy()]


      base_contencao_de_danos = pd.concat(lista_base_contencao_de_danos)

      df_final = pd.concat(lista_df_final)

      df_completo = pd.merge(df_completo,df_final[[col_data]+aberturas+['Etapa','Volume']],how='left',on=[col_data]+aberturas+['Etapa'])

      df_completo.loc[(df_completo[col_data] >= primeira_data_forecast) & (df_completo['Etapa'] == etapa_tof),['Volume_x']] = df_completo.loc[(df_completo[col_data] >= primeira_data_forecast) & (df_completo['Etapa'] == etapa_tof),['Volume_y']].values

      df_completo = df_completo.rename(columns={'Volume_x':'Volume'})

      df_completo = df_completo.drop(columns=['Volume_y'])


  df_completo = df_completo.drop(columns=topos)

  df_completo = df_completo.fillna(0)

  return df_completo,base_contencao_de_danos




