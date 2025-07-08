#@title Def aplica_forecast (fdf 1)
import timeit
import pandas as pd
import numpy as np
from clear___output import *
from colored import colored
from remove_historico_zerado import remove_historico_zerado
from tempo_maturacao import *
from forecast_2 import *
'''
Descrição Geral:

# Recebe DF completo filtrado na etapa
# Definimos o nome das colunas que contém as aberturas
# Criamos uma matriz contendo apenas as combinações de aberturas da base.

# Para cada abertura da lista_das_aberturas:

  Filtramos somente a abertura a ser projetada
  remove_historico_zerado
  tempo_maturacao
  Só prosseguimos com a projeção se existirem pontos suficientes no histórico
  Caso o tipo de projeção seja baseado num modelo treinado mas não houver parâmetros nessa abertura, alteramos o tipo de projeção para média
  ______________
  (fdf 2) forecast_2
  ______________

# return:
  df_completo_final,
  out_parametros


'''

def aplica_forecast(df_completo,      # DataFrame completo filtrado na etapa
                    df_anterior,      # DataFrame completo filtrado na etapa anterior. Usado para saber se vai ter projeção de ToF
                    df_parametros,
                    df_pareto_aberturas,
                    df_aberturas_clusterizadas, # DataFrame completo filtrado na etapa
                    col_data,
                    data_corte,
                    primeira_data_forecast,
                    etapa,
                    lista_de_etapas,
                    topo,
                    max_origin,
                    conversoes,
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

  # Vamos criar um DataFrame vazio que vai ir recebendo a projeção de cada uma das aberturas individuais:
  df_completo_final = pd.DataFrame(columns=list(df_completo.columns.values))

  df_completo_etapa = df_completo.copy()


  # Abaixo vamos definir as aberturas a serem filtradas:
  #_________________________________________________________________________________________________

  # Definimos o nome das colunas que contém as aberturas:
  aberturas = list(df_completo_etapa.columns.values)
  aberturas = aberturas[1:aberturas.index('Etapa')]

  # Criamos uma matriz contendo apenas as combinações de aberturas da base.
  # Serão essas combinações de aberturas únicas que vamos percorrer e projetar
  lista_das_aberturas = df_completo_etapa.groupby(aberturas, as_index=False)['Volume'].sum()

  lista_das_aberturas = lista_das_aberturas[aberturas].to_numpy()


  # Definindo a base de parâmetros:
  out_parametros = pd.DataFrame()
  matriz_forecast = pd.DataFrame()
  base_outliers = pd.DataFrame()
  base_contencao_de_danos = pd.DataFrame()
  base_contencao_de_danos_local = pd.DataFrame()

  #-------------------------------------------------------------------------------------------------
  # Para cada abertura da lista_das_aberturas
  tempo_do_loop = 0
  posi_etapa_na_lista = lista_de_etapas.index(etapa)
  n_etapas = len(lista_de_etapas)
  lista_de_etapas_upper = [l.upper() for l in lista_de_etapas]
  for a in range(len(lista_das_aberturas[:,0])):

    tipo_de_projecao_aplicada = tipo_de_projecao

    clear___output(flag_clear) # Comentar para manutenção
    print("Calculando...")
    if posi_etapa_na_lista == n_etapas-1:
      print("Etapa do Funil:",colored(' '.join(lista_de_etapas_upper[:posi_etapa_na_lista]), 'green'),colored(etapa.upper(), 'yellow'))
    else:
      print("Etapa do Funil:",colored(' '.join(lista_de_etapas_upper[:posi_etapa_na_lista]), 'green'),colored(etapa.upper(), 'yellow'),colored(' '.join(lista_de_etapas_upper[posi_etapa_na_lista+1:]), 'red'))
    print("Abertura:",lista_das_aberturas[a])
    print("_______________________________________________________________________________________")
    print(round(100*((a+1)/len(lista_das_aberturas[:,0])),1),"%",colored("("+str(a+1)+")",'grey'),"de",len(lista_das_aberturas[:,0]),"aberturas")
    tempo_restante = (tempo_do_loop / (a+1)) * (len(lista_das_aberturas[:,0]) - (a+1))
    tempo_restante_total = (tempo_do_loop / (a+1)) * (len(lista_das_aberturas[:,0])*(n_etapas-posi_etapa_na_lista) - (a+1))
    if a == 0:
      print("Tempo restante na etapa do funil: Calculando...")
    else:
      print("Tempo restante na etapa do funil:",colored(str(timedelta(seconds=round(tempo_restante,0))),'yellow'))
      print("Tempo restante total:            ",colored(str(timedelta(seconds=round(tempo_restante_total,0))),'red'))
    print("_______________________________________________________________________________________")

    time_start = timeit.default_timer() # registramos o início da execução

    # Vamos criar uma base contendo somente a etapa e a abertura investigados
    df_completo_abertura = df_completo_etapa.copy()
    df_anterior_abertura = df_anterior.copy()
    df_pareto_aberturas_abertura = df_pareto_aberturas.copy()
    df_aberturas_clusterizadas_abertura = df_aberturas_clusterizadas.copy()

    # Caso exista uma base de parâmetros já treinados nessa etapa, vamos criar uma cópia que será filtrada nas aberturas
    df_parametros_abertura = df_parametros.copy()

    # Filtramos somente a abertura a ser projetada
    for i in range(len(aberturas)):
      # Selecionamos o histórico
      df_completo_abertura = df_completo_abertura.loc[df_completo_abertura[aberturas[i]] == lista_das_aberturas[a][i]]
      df_anterior_abertura = df_anterior_abertura.loc[df_anterior_abertura[aberturas[i]] == lista_das_aberturas[a][i]]
      # Selecionamos o a categoria da abertura de acordo com a base de pareto:
      if len(df_pareto_aberturas)>0:
        df_pareto_aberturas_abertura = df_pareto_aberturas_abertura.loc[df_pareto_aberturas_abertura[aberturas[i]] == lista_das_aberturas[a][i]]
      # Selecionamos o cluster da abertura e a classificação se é uma abertura muito ruim para ser projetada:
      if len(df_aberturas_clusterizadas) > 0:
        df_aberturas_clusterizadas_abertura = df_aberturas_clusterizadas_abertura.loc[df_aberturas_clusterizadas_abertura[aberturas[i]] == lista_das_aberturas[a][i]]

      # Caso exista uma base de parâmetros já treinados nessa etapa, vamos filtrar as aberturas
      if len(df_parametros_abertura) > 0 and tipo_de_projecao == 'Treinado':
        df_parametros_abertura = df_parametros_abertura.loc[df_parametros_abertura[aberturas[i]] == lista_das_aberturas[a][i]]
      # Caso o modelo não seja treinado, vamos zerar a base de parâmetros para ela não ser considerada:
      else:
        df_parametros_abertura = pd.DataFrame()


    #-----------------------------------------------------------------------------------------------
    # Removemos o histórico zerado passado da base. Fazemos isso desconsiderando todos os dados de volume
    # que cumulativamente somam zero a partir da data mais antiga para a mais recente:

    # Atenção: @ essa função reordena pelas datas
    df_completo_abertura = remove_historico_zerado(train_data = df_completo_abertura,
                                                   endogenous = 'Volume',
                                                   col_data = col_data,
                                                   abertura = lista_das_aberturas[a])

    # Determinar a quantidade de semanas de maturação (chegar em 99% da cohort aberta)
    # Caso existam parâmetros já calculados, abrimos qual foi a maturação calculada:
    if len(df_parametros_abertura) > 0:
      maturacao = df_parametros_abertura['Maturação'].astype(float).max()
      maturacao = max_origin
    else:
      maturacao = tempo_maturacao(historico_df=df_completo_abertura,
                                  max_origin=max_origin,
                                  col_data = col_data,
                                  data_corte = data_corte,
                                  qtd_semanas_media=qtd_semanas_media)

    data_maturacao = primeira_data_forecast-pd.Timedelta((1+maturacao)*7, unit='D')


    #-----------------------------------------------------------------------------------------------
    # Só prosseguimos com a projeção se existirem pontos suficientes no histórico:

    # Definimos o tamanho do histórico
    len_hist_maturado = len(df_completo_abertura.loc[df_completo_abertura[col_data] <= data_maturacao])
    vol_hist_maturado_total = np.sum(df_completo_abertura.loc[df_completo_abertura[col_data] <= data_maturacao,['Volume']].fillna(0).values)

    # definimos se a abertura é ruim de projetar ou não
    if len(df_aberturas_clusterizadas_abertura) > 0:
      abertura_outlier = df_aberturas_clusterizadas_abertura['outlier'].values[0]
    elif len(df_aberturas_clusterizadas) > 0:
      abertura_outlier = 1
    else:
      abertura_outlier = 0

    # Caso a abertura tenha histórico e não seja ruim:
    if len_hist_maturado > 0 and vol_hist_maturado_total > len_hist_maturado/3 and abertura_outlier == 0:

      # Caso o histórico exista, mas ainda assim tiver poucos pontos, mudamos o modelo para média simples:
      if len(df_completo_abertura.loc[df_completo_abertura[col_data] <= data_maturacao]) < qtd_semanas_media:
        tipo_de_projecao_aplicada = 'Average'

      # Caso o tipo de projeção seja baseado num modelo treinado mas não houver parâmetros nessa abertura,
      # alteramos o tipo de projeção para média:
      if tipo_de_projecao == 'Treinado' and len(df_parametros_abertura) == 0:
        tipo_de_projecao_aplicada = 'Average'

    # Caso não haja histórico suficiente ou a abertura seja muito ruim para ser projetada, vamos
    # projetar com base na média do cluster
    elif len(df_aberturas_clusterizadas_abertura) > 0:
      tipo_de_projecao_aplicada = 'Cluster'


    # Caso não exista histórico para a abertura, vamos projetar a abertura zerada para garantir que
    # não haverá erros de base vazia.
    else:
      tipo_de_projecao_aplicada = 'Sem Projeção'
      if df_completo_abertura.loc[(df_completo_abertura[col_data] >= primeira_data_forecast),['Volume']].values.sum() == 0:
        df_completo_abertura.loc[(df_completo_abertura[col_data] >= primeira_data_forecast),['Volume','%__Volume Aberta']+conversoes] = 0
      out_parametros_f = pd.DataFrame()
      matriz_forecast_f = pd.DataFrame()
      #-------------------------------------------------------------------------------------------



    # Realizamos a projeção de acordo com as características da abertura:
    if tipo_de_projecao_aplicada != 'Sem Projeção':


      # Precisamos garantir que vamos projetar o volume de ToF na abertura correta.
      # Caso a etapa anterior à etapa com o ToF tenha volume, vamos excluir a etapa da lista de topos.
      topo_abertura = topo.copy()
      if etapa.split('2')[0] in topo and len(df_anterior_abertura) != 0:
        if sum(df_anterior_abertura['Volume'].values) > 0:
          topo_abertura.remove(etapa.split('2')[0])
      #---------------------------------------------------------------------------------------------

      # Realizamos a projeção de todas as variáveis da abertura (Volumes de topo, shares de cohorts, cohort aberta e cohorts fechadas)
      #print(df_completo_abertura[[col_data,'Volume Aberta','%__Volume Aberta','1','s__1','%__1']])

      df_completo_abertura,out_parametros_f,matriz_forecast_f,base_outliers_local,base_contencao_de_danos_local = forecast_2(df_completo = df_completo_abertura,      # DataFrame filtrado, somente datas e valores
                                                                                                                            df_parametros = df_parametros_abertura,
                                                                                                                            df_pareto_aberturas = df_pareto_aberturas_abertura,
                                                                                                                            df_aberturas_clusterizadas = df_aberturas_clusterizadas_abertura,
                                                                                                                            col_data = col_data,
                                                                                                                            data_corte = data_corte,
                                                                                                                            primeira_data_forecast = primeira_data_forecast,
                                                                                                                            maturacao = maturacao,
                                                                                                                            data_maturacao = data_maturacao,
                                                                                                                            abertura = lista_das_aberturas[a],
                                                                                                                            etapa = etapa,                    # String com o nome da etapa do funil
                                                                                                                            topo = topo_abertura,                      # Lista com o nome das etapas do ToF.
                                                                                                                            max_origin = max_origin,               # int indicando qual a maior cohort do histórico
                                                                                                                            conversoes = conversoes,               # Lista com os nomes das conversões sem '%__Volume Aberta' ('s__0','s__1',...,'s__Coincident')
                                                                                                                            qtd_semanas_media = qtd_semanas_media,
                                                                                                                            df_inputs_exogenas = df_inputs_exogenas,
                                                                                                                            tipo_de_projecao = tipo_de_projecao_aplicada,
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

      # Salvamos os outliers encontrados numa base:
      base_outliers = pd.concat([base_outliers,base_outliers_local])


      # Adicionamos a maturação:
      out_parametros_f['Maturação'] = maturacao
      out_parametros_f['Etapa'] = etapa

      # Adicionamos a informação da abertura e etapa na matriz:
      matriz_forecast_f[aberturas] = lista_das_aberturas[a]
      matriz_forecast_f['Etapa'] = etapa

      # Adicionamos a informação da abertura e etapa na base de contenção de danos:
      if len(base_contencao_de_danos_local)>0:
        base_contencao_de_danos_local[aberturas] = lista_das_aberturas[a]
        base_contencao_de_danos_local['Etapa'] = etapa


    # Adicionamos a projeção da abertura na base final
    df_completo_final = pd.concat([df_completo_final,df_completo_abertura])

    # Concatenamos os parâmetros da abertura e as matrizes de exógenas
    out_parametros = pd.concat([out_parametros,out_parametros_f])
    matriz_forecast = pd.concat([matriz_forecast,matriz_forecast_f])
    base_contencao_de_danos = pd.concat([base_contencao_de_danos,base_contencao_de_danos_local])

    time_end = timeit.default_timer()
    tempo_do_loop = tempo_do_loop + (time_end - time_start)


  if len(base_contencao_de_danos)>0:
    base_contencao_de_danos = base_contencao_de_danos[aberturas+['Etapa','Métrica','Passagem','Mensagem','Valor Histórico Médio','Valor Projetado Médio','Valor Mínimo','Valor Máximo','Delta Máximo']]
  # Retornamos a base final, que contém a projeção de todas as aberturas de uma etapa do funil específica
  return df_completo_final,out_parametros,matriz_forecast,base_outliers,base_contencao_de_danos
