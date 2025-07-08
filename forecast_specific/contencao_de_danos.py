#@title Def contencao_de_danos
import numpy as np
import pandas as pd
from projeta_por_media import projeta_por_media
'''
Essa função serve para fazer ajustes nos resultados do forecast, abertura por abertura
e em cada métrica, previnindo números negativos, conversões acima de 100% ou mesmo
números muito "estourados"
'''

def contencao_de_danos(df, # filtrado na etapa e abertura
                       endogenous,
                       col_data,
                       data_corte,
                       primeira_data_forecast,
                       tipo_de_forecast,
                       qtd_semanas_media,
                       subs_aberta_zerada,
                       subs_share_w0_zerado,
                       limite_inferior_share,
                       limite_delta_media_vol,
                       limite_delta_media_share,
                       limite_delta_media_aberta,
                      limite_proj = 1):

  mensagem = str(endogenous)
  colunas_contencao_de_danos = ['Métrica','Passagem','Mensagem','Valor Histórico Médio','Valor Projetado Médio','Valor Mínimo','Valor Máximo','Delta Máximo']
  base_contencao_de_danos = pd.DataFrame(columns=colunas_contencao_de_danos)
  passagem = -1
  valor_medio_projetado = ''
  limite_minimo = ''
  limite_maximo = ''
  limite_delta_max = ''

  endog_projetada = df.loc[df[col_data] >= primeira_data_forecast,[endogenous]].values
  endog_projetada = endog_projetada.reshape(len(endog_projetada),)

  endog_hist_df = df.loc[df[col_data] <= data_corte]
  endog_hist = endog_hist_df[endogenous].values

  if len(endog_hist) > 0 and qtd_semanas_media > 0:

    # Se a projeção for da cohort aberta, vamos remover os >100% e <0 do histórico só para garantir:
    if endogenous != 's__Coincident' and endogenous != '%__Coincident':
      endog_hist = np.where(endog_hist < 0,0,endog_hist)
      if endogenous != 'Volume':
        endog_hist = np.where(endog_hist > limite_proj,limite_proj,endog_hist)
      endog_hist_df[endogenous] = endog_hist

    # Caso a projeção seja Multilinear ou Treinada, vamos fazer a projeção via média também para poder fazer as
    # comparações
    qtd_semanas_projetadas = 0
    if tipo_de_forecast != 'Average':
      if len(endog_projetada) == 0:
        qtd_semanas_projetadas = len(df.loc[df[col_data] >= primeira_data_forecast,['Volume']].values)
      else:
        qtd_semanas_projetadas = len(endog_projetada)
      endog_media = projeta_por_media(endog_hist_df,endogenous,qtd_semanas_projetadas,qtd_semanas_media)
    else:
      endog_media = endog_projetada




    # Caso a projeção não tenha funcionado, vamos tentar remediar os casos para não parar todo o Forecast
    if len(endog_projetada) == 0:
      mensagem = mensagem+' - Projeção Vazia'

      if len(endog_media) == 0 and tipo_de_forecast != 'Average':
        if qtd_semanas_projetadas > 0:
          endog_projetada = np.zeros(qtd_semanas_projetadas)
          endog_media = [0]
        else:
          endog_projetada = []
      elif len(endog_media) == 0:
        endog_media = [0]
      elif len(endog_media) != 0 and tipo_de_forecast != 'Average':
        endog_projetada = endog_media
      else:
        if qtd_semanas_projetadas > 0:
          endog_projetada = np.zeros(qtd_semanas_projetadas)
        else:
          endog_projetada = []
      
      if len(endog_media) == 0:
        endog_media = [0]
      values_concat = pd.DataFrame([[endogenous,passagem+1,mensagem.split(' - ')[-1],endog_media[0],valor_medio_projetado,limite_minimo,limite_maximo,limite_delta_max]],columns=colunas_contencao_de_danos,index=[1])
      base_contencao_de_danos = pd.concat([base_contencao_de_danos,values_concat])

    if len(endog_projetada) == 0:
      return endog_projetada,mensagem,base_contencao_de_danos



    # Caso exista projeção, vamos comparar com os valores do histórico
    #-------------------------------------------------------------------------------------------------
    else:


      # Definimos valores mínimos, máximos e std dos valores do histórico para basearmos nossas comparações
      endog_hist_sem_zero = endog_hist[np.where(endog_hist>0)[0]]
      if len(endog_hist_sem_zero) > 0:
        valor_minimo = np.min(endog_hist_sem_zero)
      else:
        valor_minimo = np.min(endog_hist)
      if valor_minimo < 0:
        valor_minimo = 0

      valor_maximo = np.max(endog_hist)
      if valor_maximo > limite_proj and endogenous != 'Volume':
        valor_maximo = limite_proj

      valor_std = np.std(endog_hist[-qtd_semanas_media:])
      valor_medio = endog_media[0]

      # Definimos a variação WoW absoluta média e std das variações do histórico:
      wow_abs = endog_hist[-qtd_semanas_media:]
      wow_abs = np.abs(wow_abs[1:]/wow_abs[0:-1]-1)
      wow_abs[wow_abs == np.inf] = 0

      wow_abs_medio = np.average(wow_abs)
      wow_abs_std = np.std(wow_abs)


      # Definimos os valores médios e std da projeção
      valor_medio_projetado = np.average(endog_projetada)
      valor_std_projetado = np.std(endog_projetada)

      # Definimos a variação WoW absoluta média e std das variações do histórico:
      wow_abs_projetado = np.append(np.array(endog_hist[-1]),endog_projetada) # queremos incluir o último ponto histórico para calcular o WoW
      wow_abs_projetado = np.abs(wow_abs_projetado[1:]/wow_abs_projetado[0:-1]-1)
      wow_abs_projetado[wow_abs_projetado == np.inf] = 0

      wow_abs_medio_projetado = np.average(wow_abs_projetado)
      wow_abs_std_projetado = np.std(wow_abs_projetado)

      # Definimos os limites máximos e mínimos permitidos:
      if endogenous == 'Volume':
        limite_delta_media = limite_delta_media_vol
      elif endogenous == '%__Volume Aberta' or endogenous == '%__Coincident':
        limite_delta_media = limite_delta_media_aberta
      else:
        limite_delta_media = limite_delta_media_share

      limite_minimo = valor_medio-limite_delta_media*valor_std
      limite_maximo = valor_medio+limite_delta_media*valor_std

      #-----------------------------------------------------------------------------------------------

      # Para o caso do tipo de projeção Multilienar, vamos rodar a contenção de dados 2 vezes. 1 inicial
      # substituíndo os erros pela projeção média e outra substituíndo os erros da projeção média
      # pelos valores pre-estabelecidos.
      if tipo_de_forecast != 'Average':
        passagem = 0
      else:
        passagem = 1
      while passagem < 2:
        #---------------------------------------------------------------------------------------------



        # Rebaixar curvas inteiras destoantes da média
        if tipo_de_forecast != 'Average' and passagem == 0:

          # Caso a mudança de nível da projeção ainda gere pontos destoantes da média,
          # estes ainda serão tratados. Caso gere valores negativos ou zerados, também
          # serão tratados.
          if valor_medio_projetado < limite_minimo or valor_medio_projetado > limite_maximo:
            if valor_medio_projetado < limite_minimo:
              endog_projetada = endog_projetada-valor_medio_projetado+limite_minimo
              mensagem = mensagem+' - Patamar Médio Projetado Vs. Média Aumentado'
            if valor_medio_projetado > limite_maximo:
              endog_projetada = endog_projetada-valor_medio_projetado+limite_maximo
              mensagem = mensagem+' - Patamar Médio Projetado Vs. Média Reduzido'


            values_concat = pd.DataFrame([[endogenous,passagem+1,mensagem.split(' - ')[-1],endog_media[0],valor_medio_projetado,limite_minimo,limite_maximo,limite_delta_max]],columns=colunas_contencao_de_danos,index=[1])
            base_contencao_de_danos = pd.concat([base_contencao_de_danos,values_concat])



        # Remover muito destoantes da média
        if tipo_de_forecast != 'Average' and passagem == 0:


          posi_delta_alto = np.where(endog_projetada > limite_maximo)[0]
          posi_delta_baixo = np.where(endog_projetada < limite_minimo)[0]


          subs_min = limite_minimo
          subs_max = limite_maximo

          if endogenous == '%__Volume Aberta' or endogenous == '%__Coincident':
            if subs_min < valor_minimo:
              subs_min = valor_minimo
            if subs_max > valor_maximo:
              subs_max = valor_maximo


          if len(posi_delta_alto) > 0:
            endog_projetada[posi_delta_alto] = subs_max
          if len(posi_delta_baixo) > 0:
            endog_projetada[posi_delta_baixo] = subs_min

          if len(posi_delta_alto) > 0 or len(posi_delta_baixo) > 0:
            mensagem = mensagem+' - Delta Alto Vs. Média Removidos'

            values_concat = pd.DataFrame([[endogenous,passagem+1,mensagem.split(' - ')[-1],endog_media[0],valor_medio_projetado,limite_minimo,limite_maximo,limite_delta_max]],columns=colunas_contencao_de_danos,index=[1])
            base_contencao_de_danos = pd.concat([base_contencao_de_danos,values_concat])


          # Remover variações WoW extremas:
          limite_delta_max = wow_abs_medio+limite_delta_media*wow_abs_std

          hist_e_projetado = np.append(np.array(endog_hist[-1]),endog_projetada)
          wow_hist_e_projetado = hist_e_projetado[1:]/hist_e_projetado[:-1]-1
          wow_hist_e_projetado[wow_hist_e_projetado == np.inf] = 0
          wow_hist_e_projetado[wow_hist_e_projetado == -np.inf] = 0

          subs_max = hist_e_projetado[:-1]*(1+limite_delta_max)
          subs_min = hist_e_projetado[:-1]*(1-limite_delta_max)

          posi_delta_alto = np.where(wow_hist_e_projetado > (limite_delta_max))[0]
          posi_delta_baixo = np.where(wow_hist_e_projetado < (-limite_delta_max))[0]
          if len(posi_delta_alto) > 0:
            endog_projetada[posi_delta_alto] = subs_max[posi_delta_alto]
          if len(posi_delta_baixo) > 0:
            endog_projetada[posi_delta_baixo] = subs_min[posi_delta_baixo]

          if len(posi_delta_alto) > 0 or len(posi_delta_baixo) > 0:
            mensagem = mensagem+' - WoW Alto Vs. Média Removidos'

            values_concat = pd.DataFrame([[endogenous,passagem+1,mensagem.split(' - ')[-1],endog_media[0],valor_medio_projetado,limite_minimo,limite_maximo,limite_delta_max]],columns=colunas_contencao_de_danos,index=[1])
            base_contencao_de_danos = pd.concat([base_contencao_de_danos,values_concat])



        # Remover valores negativos

        # No caso do s__Coincident' permitimos valores negativos
        # de no máximo -100% do share que sobrou da cohort
        if endogenous != 's__Coincident' and endogenous != '%__Coincident':
          endog_projetada = np.where(endog_projetada < 0,0,endog_projetada)
          posi_danos = np.where(endog_projetada < 0)[0]
          if len(posi_danos) > 0:
            mensagem = mensagem+' - Negativos Removidos'

            values_concat = pd.DataFrame([[endogenous,passagem+1,mensagem.split(' - ')[-1],endog_media[0],valor_medio_projetado,limite_minimo,limite_maximo,limite_delta_max]],columns=colunas_contencao_de_danos,index=[1])
            base_contencao_de_danos = pd.concat([base_contencao_de_danos,values_concat])


        # Remover valores acima de 100%
        if endogenous != 'Volume' and endogenous != 's__Coincident' and endogenous != '%__Coincident':
          endog_projetada = np.where(endog_projetada > limite_proj,limite_proj,endog_projetada)
          posi_danos = np.where(endog_projetada > limite_proj)[0]
          if len(posi_danos) > 0:
            mensagem = mensagem+' - > 100% Removidos'

            values_concat = pd.DataFrame([[endogenous,passagem+1,mensagem.split(' - ')[-1],endog_media[0],valor_medio_projetado,limite_minimo,limite_maximo,limite_delta_max]],columns=colunas_contencao_de_danos,index=[1])
            base_contencao_de_danos = pd.concat([base_contencao_de_danos,values_concat])


        # Remover Cohort aberta, share W0 e volume inteiramente zerados
        if sum(endog_projetada) == 0 and (endogenous == '%__Volume Aberta' or endogenous == 's__0' or endogenous == 'Volume'):
          if endogenous == 'Volume':
            limite_delta_media = 0
          elif endogenous == '%__Volume Aberta':
            limite_delta_media = limite_delta_media_aberta
          else:
            limite_delta_media = limite_delta_media_share

          if valor_medio-limite_delta_media*valor_std > 0:
            subs = valor_medio-limite_delta_media*valor_std
          elif valor_minimo > 0:
            subs = valor_minimo
          else:
            if endogenous == '%__Volume Aberta':
              subs = subs_aberta_zerada
            if endogenous == 's__0':
              subs = subs_share_w0_zerado
            if endogenous == 's__Coincident':
              subs = 1
            if endogenous == 'Volume':
              subs = 0

          endog_projetada[:] = subs

          mensagem = mensagem+' - Totalmente Zerados Removidos'

          values_concat = pd.DataFrame([[endogenous,passagem+1,mensagem.split(' - ')[-1],endog_media[0],valor_medio_projetado,limite_minimo,limite_maximo,limite_delta_max]],columns=colunas_contencao_de_danos,index=[1])
          base_contencao_de_danos = pd.concat([base_contencao_de_danos,values_concat])


        # Remover Cohort aberta e share W0 parcialmente zerados
        if (endogenous == '%__Volume Aberta' or endogenous == 's__0' or endogenous == 's__Coincident'):
          if endogenous == '%__Volume Aberta':
            limite_delta_media = limite_delta_media_aberta
          else:
            limite_delta_media = limite_delta_media_share

          if valor_medio-limite_delta_media*valor_std > 0:
            subs = valor_medio-limite_delta_media*valor_std
          elif valor_minimo > 0:
            subs = valor_minimo
          else:
            if endogenous == '%__Volume Aberta':
              subs = subs_aberta_zerada
            elif endogenous == 's__0':
              subs = subs_share_w0_zerado
            elif endogenous == 's__Coincident':
              subs = 1
            else:
              subs = subs_share_w0_zerado

          endog_projetada[:] = np.where(endog_projetada <= 0, subs, endog_projetada)

          mensagem = mensagem+' - Zerados Removidos'

          values_concat = pd.DataFrame([[endogenous,passagem+1,mensagem.split(' - ')[-1],endog_media[0],valor_medio_projetado,limite_minimo,limite_maximo,limite_delta_max]],columns=colunas_contencao_de_danos,index=[1])
          base_contencao_de_danos = pd.concat([base_contencao_de_danos,values_concat])

        # Remover shares muito baixos
        if endogenous[0:3] == 's__':

          endog_projetada = np.where((endog_projetada > 0) & (endog_projetada < limite_inferior_share), limite_inferior_share, endog_projetada)

          if len(np.where(endog_projetada == limite_inferior_share)[0]) > 0:
            mensagem = mensagem+' - Shares Muito Baixos Removidos'

            values_concat = pd.DataFrame([[endogenous,passagem+1,mensagem.split(' - ')[-1],endog_media[0],valor_medio_projetado,limite_minimo,limite_maximo,limite_delta_max]],columns=colunas_contencao_de_danos,index=[1])
            base_contencao_de_danos = pd.concat([base_contencao_de_danos,values_concat])


        # Substituir cohorts que são sempre 100% (FSS)
        if endogenous == '%__Volume Aberta' or endogenous == 's__0':

          if np.average(endog_hist_sem_zero) > 0.99 and np.average(endog_hist_sem_zero) < 1.01:
            endog_projetada[:] = 1

          elif valor_std == 0:
            endog_projetada = endog_media
            mensagem = mensagem+' - Sem Variação do Histórico'

            values_concat = pd.DataFrame([[endogenous,passagem+1,mensagem.split(' - ')[-1],endog_media[0],valor_medio_projetado,limite_minimo,limite_maximo,limite_delta_max]],columns=colunas_contencao_de_danos,index=[1])
            base_contencao_de_danos = pd.concat([base_contencao_de_danos,values_concat])

        passagem = passagem+1
        '''
        print(passagem)
        print(endog_projetada)
        print('____________________________________')
        '''

    # Não achamos interessante manter a mensagem "Zerados Removidos" da base de contenção de danos.
    # Também achamos pouco útil as métricas além de volume, cohort aberta e share da W0:
    if len(base_contencao_de_danos) > 0:
      base_contencao_de_danos = base_contencao_de_danos.loc[(base_contencao_de_danos['Mensagem'] != 'Zerados Removidos') & (base_contencao_de_danos['Métrica'].isin(['s__0','%__Volume Aberta','Volume']))]
      base_contencao_de_danos = base_contencao_de_danos[colunas_contencao_de_danos]



    return endog_projetada,mensagem,base_contencao_de_danos

  elif len(endog_hist) > 0:
    return endog_projetada,mensagem,base_contencao_de_danos
  else:
    endog_projetada = 0
    mensagem = 'Sem Histórico'
    return endog_projetada,mensagem,base_contencao_de_danos
