#@title Def funil_dinamico_forecast (fdf 0)
import pandas as pd
import numpy as np
from aplica_forecast import aplica_forecast
from projeta_tof_externo import projeta_tof_externo
from formata_base_para_funil import formata_base_para_funil
from auxiliary_functions import *
from base_de_inputs import base_de_inputs
from progressao_funil import progressao_funil
from data_export import exportar_base
from formatacao_output_forecast import formatacao_output_forecast
from clear___output import *
from colored import colored
'''
Descrição Geral:

# Define o nome das aberturas da base
# Definimos as etapas do funil
# Ordenamos corretamente as etapas do funil

# Para cada etapa do funil:

  Filtra a etapa na base geral
  ________________________
  (fdf 1) aplica_forecast
  ________________________
  aplica os inputs
  realiza a progressão do funil (calcula volumes da etapa seguinte)

# Formata base final

retorna:
  df_funil,
  etapas_vol,
  etapas_conv,
  out_parametros,
  funil_cru



'''

def funil_dinamico_forecast(df_completo,      # DataFrame completo do histórico + séries exógenas históricas e projetadas
                            df_ToF_externo,   # DataFrame contendo uma base projetada do ToF
                            df_parametros,    # DataFrame dos parâmetros do medelo já treinados (se houver)
                            df_pareto_aberturas,
                            df_aberturas_clusterizadas,
                            colunas_pareto,
                            df_inputs,        # DataFrame com os inputs manuais de ganhos de conversão
                            df_backup_cohort,
                            df_backup_parametros,
                            nome_e_arquivo_de_destino_backup_cohort,
                            nome_e_arquivo_de_destino_backup_parametros,
                            usar_backup,
                            Nome_do_arquivo_sheets,
                            data_corte,       # Data referente à última data do histórico
                            primeira_data_forecast,  # Data referente à primeira data de forecast (não necessariamente é a seguinte da última de histórico)
                            topo,                    # Lista com o nome das etapas de topo de funil
                            etapas_vol,
                            etapas_conv,
                            max_origin,              # inteiro indicando a cohort máxima
                            conversoes,              # Lista com o nome das conversões que vão ser projetadas
                            qtd_semanas_media,
                            df_inputs_exogenas,
                            tipo_de_projecao,
                            remover_outliers,
                            estimativa_outliers,
                            aplicar_ajuste_w0,
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
                            nome_do_painel_de_controle,
                            fit_intercept):

  # Define o cabeçalho da base
  cb_df = list(df_completo.columns.values)

  # Define o nome das aberturas da base
  chaves = cb_df[:cb_df.index('Etapa')]

  # Definimos o nome da coluna que contém as datas
  col_data = chaves[0]

  # Definimos o nome das colunas que contém as aberturas:
  aberturas = list(df_completo.columns.values)
  aberturas = aberturas[1:aberturas.index('Etapa')]

  # Vamos ordenar a base completa de acordo a soma do volume total de todas as etapas:
  if len(df_pareto_aberturas) > 0:
    df_completo = pd.merge(df_completo,df_pareto_aberturas,how='left',on=aberturas)
    df_completo = df_completo.sort_values(by='%_total_final', ascending=False)
    df_completo = df_completo.drop(columns=colunas_pareto)

  # Definindo a base de inputs:
  base_inputs = []

  # definindo a base de parâmetros
  out_parametros = pd.DataFrame()
  matriz_forecast = pd.DataFrame()
  base_outliers = pd.DataFrame()
  base_contencao_de_danos = pd.DataFrame()

  if len(df_parametros) > 0 and tipo_de_projecao == 'Treinado':
    df_parametros = df_parametros.rename(columns={'etapa':'Etapa','endógena':'Endógena','exógena':'Exógena','maturação':'Maturação'})


  #-------------------------------------------------------------------------------------------------
  # Vamos definir a lista de etapas do funil:
  # Definimos as etapas do funil
  '''
  etapas = list(np.unique(df_completo['Etapa'].values))
  etapas = [e.lower() for e in etapas]
  topo = [t.lower() for t in topo]
  etapas_vol = etapas.copy()
  etapas_conv = etapas.copy()
  ordem = []
  for i in range(len(etapas)):
    etapas_vol[i] = etapas[i].split('2')[0]

    if etapas[i].find('2') != -1:
      segunda_etapa = etapas[i].split('2')[1]
    else:
      segunda_etapa = 'sem_etapa'

    if etapas_vol[i] in topo:
      ordem = ordem + [etapas_vol[i]]

    if segunda_etapa in topo:
      ordem.remove(segunda_etapa)

    if etapas[i].find('2') == -1:
      etapas_conv.remove(etapas[i])


  # Ordenamos corretamente as etapas do funil
  etapas_vol_ordenado = ordem.copy()
  etapas_conv_ordenado = []
  for o in range(len(etapas)-1):
    for i in range(len(etapas_conv)):
      if etapas_conv[i].split('2')[0] in ordem and etapas_conv[i].split('2')[-1] not in ordem:
        ordem = ordem + [etapas_conv[i].split('2')[-1]]
        etapas_vol_ordenado = etapas_vol_ordenado + [etapas_conv[i].split('2')[-1]]
        etapas_conv_ordenado = etapas_conv_ordenado + [etapas_conv[i]]


  etapas_vol = etapas_vol_ordenado
  etapas_conv = etapas_conv_ordenado
  '''
  #-------------------------------------------------------------------------------------------------

  # Primeiramente, caso a projeção do ToF seja externa, vamos substituir de antemão na base completa
  # os dados futuros do volume de ToF:
  if tipo_de_tof == 'Input Externo' and len(df_ToF_externo) > 0:

    df_completo,base_contencao_de_danos = projeta_tof_externo(df_completo = df_completo, # base completa sem estar filtrada
                                                              df_ToF_externo = df_ToF_externo,
                                                              col_data = col_data,
                                                              aberturas = aberturas,
                                                              primeira_data_forecast = primeira_data_forecast,#data_corte,
                                                              lista_de_etapas = etapas_conv,
                                                              topos = topo,
                                                              qtd_semanas_media = qtd_semanas_media,
                                                              limite_delta_media_vol = limite_delta_vol)



  # para cada etapa de conversão que identificamos, vamos calcular os volumes das cohorts e o volume
  # coincident da etapa seguinte utilizando a função auxiliar de progerssão do funil
  #_________________________________________________________________________________________________
  i = 0
  for e in etapas_conv:


    # Vamos verificar se essa etapa do funil já foi projetada. Se já foi projetada, vamos abrir os
    # outputs_cohort e prosseguir com a projeção da etapa seguinte. Sabemos se a base cohort é parcial
    # se ela foi salva com acoluna "Shifted":
    teste_ja_projetou = False
    if len(df_backup_cohort)>0 and usar_backup:
      df_backup_cohort = df_backup_cohort.rename(columns={'shifted':'Shifted','week origin':'Week Origin'})
      if e.split("2")[-1] in list(df_backup_cohort.columns.values)  and 'Shifted' in list(df_backup_cohort.columns.values):

        out_parametros = df_backup_parametros
        out_parametros = out_parametros.rename(columns={'etapa':'Etapa','endógena':'Endógena','exógena':'Exógena','maturação':'Maturação'})

        valores = list(set(list(df_backup_cohort.columns.values)) - set(chaves+[col_data,'Week Origin','Shifted']))

        df_backup_cohort[col_data] =  pd.to_datetime(df_backup_cohort[col_data], infer_datetime_format=True)
        df_backup_cohort['Week Origin'] =  df_backup_cohort['Week Origin'].replace('0.00E+00','0')
        df_backup_cohort['Week Origin'] =  df_backup_cohort['Week Origin'].replace('1.00E+00','1')
        df_backup_cohort['Week Origin'] =  df_backup_cohort['Week Origin'].replace('2.00E+00','2')
        df_backup_cohort['Week Origin'] =  df_backup_cohort['Week Origin'].replace('3.00E+00','3')
        df_backup_cohort['Week Origin'] =  df_backup_cohort['Week Origin'].replace('4.00E+00','4')
        df_backup_cohort['Week Origin'] =  df_backup_cohort['Week Origin'].replace('5.00E+00','5')
        df_backup_cohort[valores] = df_backup_cohort[valores].astype(float)

        df_funil = df_backup_cohort.copy()
        df_funil_etapa_anterior = df_backup_cohort.copy()
        etapa_anterior = e

        teste_ja_projetou = True

    if not teste_ja_projetou:

      # Primeiro, selecionamos do dataframe histórico e futuro exógeno completo somente a etapa
      # do funil que estamos trabalhando:
      df_completo_etapa = df_completo.loc[df_completo['Etapa'] == e]

      # Selecionamos os parametros da etapa também
      if len(df_parametros) > 0:
        df_parametros_etapa = df_parametros.loc[df_parametros['Etapa'] == e]
      else:
        df_parametros_etapa = df_parametros

      # Precisamos selecionar também a base com a etapa anterior.
      # Precisamos dessa base para verificar se vai haver projeção de ToF com base na existência de
      # volume na etapa anterior na mesma abertura:
      if i != 0:
        df_completo_etapa_anterior = df_completo.loc[df_completo['Etapa'] == etapas_conv[i-1]]
      else:
        df_completo_etapa_anterior = pd.DataFrame(columns=list(df_completo.columns.values))
      i=i+1


      # Selecionamos a etapa na base de clusters:
      if len(df_aberturas_clusterizadas)>0:
        df_aberturas_clusterizadas_etapa = df_aberturas_clusterizadas.loc[df_aberturas_clusterizadas['Etapa'] == e]
      else:
        df_aberturas_clusterizadas_etapa = df_aberturas_clusterizadas


      # Antes de realizar a projeção, precisamos verificar se a etapa a ser projetada não é a primeira.
      # Se não for a primeira etapa do funil, precisamos incluir na base completa o volume resultante
      # da progressão do funil referente à projeção do funil da etapa anterior:
      #-----------------------------------------------------------------------------------------------
      if e != etapas_conv[0]:

        # Definimos qual etapa de volume precisamos selecionar
        etapa_vol = e.split('2')[0]

        # Selecionamos apenas as colunas que importam do df_funil
        df_funil_volume = df_funil.groupby(chaves, as_index=False)[[etapa_vol]].first()

        # Selecionamos apenas as semanas futuras do df_completo_etapa:
        df_completo_etapa_futuro = df_completo_etapa.loc[df_completo_etapa[col_data] >= primeira_data_forecast]


        # Mergeamos o volume do funil com o df_completo_etapa_futuro:
        df_completo_etapa_futuro = pd.merge(df_completo_etapa_futuro,df_funil_volume,how='left',on=chaves)

        # Somamos os valores de volumes no df_completo_etapa:
        df_completo_etapa.loc[df_completo_etapa[col_data] >= primeira_data_forecast,['Volume']] = df_completo_etapa.loc[df_completo_etapa[col_data] >= primeira_data_forecast]['Volume'].values + df_completo_etapa_futuro[etapa_vol].values


      # Aqui vamos projetar todas as conversões cohorts fechadas da etapa, além do ToF, se existir, na
      # etapa do funil que estamos trabalhando:
      #-----------------------------------------------------------------------------------------------
      forecast_df,out_parametros_f,matriz_forecast_f,base_outliers_local,base_contencao_de_danos_local = aplica_forecast(df_completo = df_completo_etapa,      # DataFrame completo filtrado na etapa
                                                                                                                        df_anterior = df_completo_etapa_anterior,           # DataFrame completo filtrado na etapa anterior. Usado para saber se vai ter projeção de ToF
                                                                                                                          df_parametros = df_parametros_etapa,
                                                                                                                        df_aberturas_clusterizadas = df_aberturas_clusterizadas_etapa,
                                                                                                                        df_pareto_aberturas = df_pareto_aberturas,
                                                                                                                          col_data = col_data,
                                                                                                                          data_corte = data_corte,
                                                                                                                          primeira_data_forecast = primeira_data_forecast,
                                                                                                                          etapa = e,
                                                                                                                          lista_de_etapas = etapas_conv,
                                                                                                                          topo = topo,
                                                                                                                          max_origin = max_origin,
                                                                                                                          conversoes = conversoes,
                                                                                                                          qtd_semanas_media = qtd_semanas_media,
                                                                                                                          df_inputs_exogenas = df_inputs_exogenas,
                                                                                                                          tipo_de_projecao = tipo_de_projecao,
                                                                                                                          remover_outliers = remover_outliers,
                                                                                                                          estimativa_outliers = estimativa_outliers,
                                                                                                                          retorna_parametros = retorna_parametros,
                                                                                                                          tipo_de_tof = tipo_de_tof,
                                                                                                                            premissa_dummy = premissa_dummy,
                                                                                                                            aberta_zerada = aberta_zerada,
                                                                                                                            share_w0_zerado = share_w0_zerado,
                                                                                                                            limite_share = limite_share,
                                                                                                                            limite_delta_vol = limite_delta_vol,
                                                                                                                            limite_delta_share = limite_delta_share,
                                                                                                                            limite_delta_aberta = limite_delta_aberta,
                                                                                                                            limite_proj = limite_proj,
                                                                                                                         fit_intercept = fit_intercept)

      # Adicionando a base de parâmetros com a da etapa anterior já calculada:
      out_parametros = pd.concat([out_parametros,out_parametros_f])
      matriz_forecast = pd.concat([matriz_forecast,matriz_forecast_f])
      base_outliers = pd.concat([base_outliers,base_outliers_local])
      base_contencao_de_danos = pd.concat([base_contencao_de_danos,base_contencao_de_danos_local])


      #print(out_parametros.columns.values)
      # Antes de aplicar a projeção das cohorts, precisamos transformar o formato do forecast_df
      # no formato do df_funil. Caso essa seja a primerira etapa do funil, vamos transformar
      # a base como um todo e definir o df_funil inicial. Caso contrário, vamos somente adicionar
      # as novas etapas projetadas ao df_funil já existente:
      #-----------------------------------------------------------------------------------------------
      if e == etapas_conv[0]:
        df_funil = formata_base_para_funil(df = forecast_df,
                                          max_origin = max_origin,
                                          etapa = e)

        # Selecionamos apenas as datas necessárias para realizar a progreção do funil. Não queremos
        # progredir a base histórica inteira:
        # A data mínima onde começa a progressão do funil é igual a 2 vezes a maturação máxima
        data_min = primeira_data_forecast - pd.Timedelta(2*max_origin*7, unit='D')
        data_max = np.max(df_funil[col_data].values) # Usada para aplicar inputs mais tarde
        df_funil = df_funil.loc[df_funil[col_data] >= data_min]



      else:
        df_funil_etapa = formata_base_para_funil(df = forecast_df,
                                                max_origin = max_origin,
                                                etapa = e)


        # Adicionamos as conversões em % projetadas à base df_funil.
        # Caso a etapa projetada (que aqui não é mais a primeira) também tenha a projeção do volume
        # de um ToF, precisamos somar este volume ao volume anterior calculado via progressão do funil.
        etapa_vol_anterior = e.split('2')[0]
        if etapa_vol_anterior in topo:

          df_funil = pd.merge(df_funil,df_funil_etapa[chaves+['Week Origin',e,etapa_vol_anterior]],how='left',on=chaves+['Week Origin'])
          df_funil[etapa_vol_anterior+'_x'] = df_funil[etapa_vol_anterior+'_x'].astype(float)
          df_funil[etapa_vol_anterior+'_y'] = df_funil[etapa_vol_anterior+'_y'].astype(float)

          # Porém, devemos nos atentar para o fato de que mesmo se a etapa anterior continha um ToF, não necessariamente
          # esse ToF entrou em todas as aberturas. Assim, só podemos somar ToF calculado diretamente com
          # o ToF calculado via conversão nos casos onde os valores são diferentes, pois nos casos onde
          # os valores são iguais significa que a abertura não recalculou aquela etapa por não considerá-la
          # um ToF.

          # Como no df_funil já existia a etapa_vol_anterior calculada via cohort, a base ficou com duas
          # colunas de volume dessa etapa. Vamos somá-las e remover a duplicação:
          df_funil[etapa_vol_anterior] = df_funil[etapa_vol_anterior+'_x'] + df_funil[etapa_vol_anterior+'_y']

          df_funil.loc[df_funil[etapa_vol_anterior+'_x'] != 0,[etapa_vol_anterior]] =\
          df_funil.loc[df_funil[etapa_vol_anterior+'_x'] != 0,[etapa_vol_anterior+'_x']].values

          #print(df_funil[['week_start',etapa_vol_anterior,etapa_vol_anterior+'_x',etapa_vol_anterior+'_y']])

          df_funil = df_funil.drop(columns=[etapa_vol_anterior+'_x',etapa_vol_anterior+'_y'])



        # Caso a etapa anterior não tenha sido um ToF, vamos simplesmente adicionar as conversões em % projetadas à base df_funil.
        else:
          #print("---------------------------")
          #print(e,etapa_vol_anterior)
          #print(df_funil.loc[df_funil['week_start'] == '2022-07-04'])
          #print(df_funil_etapa.loc[df_funil_etapa['week_start'] == '2022-07-04'])
          df_funil = pd.merge(df_funil,df_funil_etapa[chaves+['Week Origin',e]],how='left',on=chaves+['Week Origin'])

          # Para que o volume do passado não seja resultado do cálculo da cohort,
          # vamos substituir os valores da base funil pelos valores que estão na df_funil_etapa para datas de histórico:



        '''
        # Para garantir que o resultado final não vai conter erros, vamos remover conversões >100% e <0 que podem ter sido geradas aqui:
        valores = df_funil.loc[df_funil['Week Origin'] != 'Coincident'][e].astype(float).values
        valores = np.where(valores<0,0,valores)
        valores = np.where(valores>1,1,valores)
        df_funil.loc[df_funil['Week Origin'] != 'Coincident'][e] = valores
        '''
        #---------------------------------------------------------------------------------------------
        # Agora precisamos adicionar os volumes realizados corretos do histórico não maturado
        # da etapa anterior, pois não queremos realizar a progressão do funil com base nos volumes
        # progredidos desse período:

        # Porém, não queremos substituir pelo volume da etapa anterior caso essa seja um ToF, pois nesse
        # caso não existe volume na base anterior calculado via cohort.
        etapa_anterior_vol = etapa_anterior.split('2')[1]
        if etapa_vol_anterior not in topo:
          df_funil_realizado = df_funil_etapa_anterior.loc[df_funil_etapa_anterior[col_data] < primeira_data_forecast, [etapa_anterior_vol,"coh_vol_"+etapa_anterior]].values

          # Vamos substituir os valores no funil atual:
          df_funil.loc[df_funil[col_data] < primeira_data_forecast, [etapa_anterior_vol,"coh_vol_"+etapa_anterior]] = df_funil_realizado




      # Adicionamos as datas deslocadas para realizar a conta da cohort:
      #-----------------------------------------------------------------------------------------------
      df_funil['Shifted'] = shift_datas(datas = df_funil[col_data],     # vetor contendo as datas das cohorts em formato datetime
                                          conv = df_funil['Week Origin'],      # vetor contendo as cohorts correspondentes
                                          intervalo = 7, # interio que define o intervalo da cohort em dias (no caso semanal = 7 dias)
                                          max_conv = max_origin, # inteiro que define qual é a cohort máxima (no caso = 5)
                                          aplicar_ajuste_w0 = aplicar_ajuste_w0)



      # Aqui, se existir uma base de inputs manuais, vamos aplicá-los em cima da projeção antes de
      # enviar para a progerssão do funil:
      #-----------------------------------------------------------------------------------------------
      #print(df_funil.loc[df_funil['week_start'] == '2022-06-27'])
      if len(df_inputs) > 0:

        if len(base_inputs) == 0:
          base_inputs = base_de_inputs(df_inputs = df_inputs,
                                      modelo = df_funil,
                                      aberturas = chaves[1:],
                                      max_origin = max_origin,
                                      col_data = col_data,
                                      dia_inicio = data_min,
                                      dia_fim = data_max)

        inputs_etapa = base_inputs.loc[base_inputs['Etapa'] == e]
        colunas_inputs = list(inputs_etapa.columns.values)
        colunas_inputs = colunas_inputs[colunas_inputs.index('Etapa')+1:]
        inputs_etapa = inputs_etapa.drop(columns=['Etapa'])

        if len(inputs_etapa) > 0:
          df_funil['Week Origin'] = df_funil['Week Origin'].astype(str)
          inputs_etapa['Week Origin'] = inputs_etapa['Week Origin'].astype(str)
          df_funil = pd.merge(df_funil,inputs_etapa,how='left',on=['Week Origin']+chaves)
          df_funil = df_funil.fillna(0)
          for c in colunas_inputs:
            metrica = c
            metrica = metrica.replace("i_","")
            if metrica in list(df_funil.columns.values):
              df_funil[metrica] = (df_funil[c].values+1)*df_funil[metrica].astype(float).values
              # Redistrubuir cohort aberta que ficou > 100%
          df_funil = df_funil.drop(columns=colunas_inputs)



      # Agora vamos finalmente calcular a progressão do funil para a próxima etapa:
      #-----------------------------------------------------------------------------------------------
      df_funil = df_funil.fillna(0)
      etapas_split = [["sem split"]]

      # Vamos substituir os volumes calculados pela função progressão no passado pelos volumes realizados:
      try:
        df_funil = pd.merge(df_funil,df_funil_etapa[chaves+['Week Origin',etapa_anterior_vol]],how='left',on=chaves+['Week Origin'])
        df_funil.loc[df_funil['week_start'] <= data_corte,etapa_anterior_vol+'_x'] = df_funil.loc[df_funil['week_start'] <= data_corte,etapa_anterior_vol+'_y'].values
        df_funil[etapa_anterior_vol] = df_funil[etapa_anterior_vol+'_x'].values
        df_funil = df_funil.drop(columns=[etapa_anterior_vol+'_x',etapa_anterior_vol+'_y'])
      except:
        #print(df_funil_etapa.head(3))
        df_funil = df_funil


      df_funil = progressao_funil(df_funil,
                                    e,
                                    etapas_split,
                                    chaves,
                                    max_origin) # @função_auxiliar


      # Vamos remover possíveis valores negativos:
      df_funil.loc[df_funil[e.split('2')[1]] < 0,[e.split('2')[1]]] = 0

      # Agora podemos atualizar o funil anterior e a etapa anterior:
      df_funil_etapa_anterior = df_funil.copy()
      etapa_anterior = e


      # Vamos salvar o resultado parcial da projeção:
      #---------------------------------------------------------------------------------------------
      exportar_base(base_df = df_funil,                     # DataFrame
                  nome_do_painel_de_controle = nome_do_painel_de_controle,  # String com o nome do arquivo sheets com o painel de controle
                  arquivo_de_destino = nome_e_arquivo_de_destino_backup_cohort[1],          # String com o nome do arquivo sheets de destino ou o caminho do Google Drive
                  nome_da_aba = nome_e_arquivo_de_destino_backup_cohort[0],                 # String com o nome da aba no arquivo sheets de destino ou nome final do CSV que será salvo
                  substituir = True)                 # Booleano indicando se o arquivo CSV será salvo por cima de um existente com mesmo nome ou se será salva uma cópia com nome diferente

      exportar_base(base_df = out_parametros,                     # DataFrame
                  nome_do_painel_de_controle = nome_do_painel_de_controle,  # String com o nome do arquivo sheets com o painel de controle
                  arquivo_de_destino = nome_e_arquivo_de_destino_backup_parametros[1],          # String com o nome do arquivo sheets de destino ou o caminho do Google Drive
                  nome_da_aba = nome_e_arquivo_de_destino_backup_parametros[0],                 # String com o nome da aba no arquivo sheets de destino ou nome final do CSV que será salvo
                  substituir = True)                 # Booleano indicando se o arquivo CSV será salvo por cima de um existente com mesmo nome ou se será salva uma cópia com nome diferente



  #_________________________________________________________________________________________________
  # Após calcular a base completa, formatamos as bases finais separando a base de volumes cohort dos
  # volumes coincident, utilizando uma função auxiliar
  # @função_auxiliar

  funil_cru = df_funil.copy()

  df_funil = formatacao_output_forecast(base = df_funil,
                                   etapas_coh = etapas_conv,
                                   etapas_vol = etapas_vol,
                                   chaves = chaves,
                                   col_data = col_data)

  # Vamos remover possíveis valores negativos:
  for e in etapas_conv:
    df_funil.loc[(df_funil['Week Origin'] != 'Coincident') & (df_funil['Week Origin'] != 'Não Convertido') & (df_funil[e] < 0),[e] ] = 0

  clear___output(flag_clear) # Comentar para manutenção
  print(colored("Funil Dinâmico Finalizado",'green'))


  # Formatamos os outputs de parâmetros:
  cb_out = list(out_parametros.columns.values)
  if 'Exógena' in cb_out:
    aberturas = cb_out[cb_out.index('Exógena')+1:cb_out.index('Endógena')]
    out_parametros = out_parametros[['Etapa']+aberturas+['Maturação','Endógena','Exógena','slope','intercept','error']]
    out_parametros['slope'] = out_parametros['slope'].astype(float)
    out_parametros = out_parametros.loc[out_parametros['slope'] != 0]

  return df_funil,etapas_vol,etapas_conv,out_parametros,funil_cru,matriz_forecast,base_outliers,base_contencao_de_danos

#---------------------------------------------------------------------------------------------------
