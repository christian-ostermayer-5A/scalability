#@title Def treina_melhores_exogs_e_lags (AG 0)
import timeit
from datetime import timedelta
import pandas as pd
import numpy as np
from remove_historico_zerado import remove_historico_zerado
from tempo_maturacao import tempo_maturacao
from aplica_teste import aplica_teste
from data_export import *
from clear___output import *

def treina_melhores_exogs_e_lags(df_completo,      # DataFrame completo
                                df_inputs_exogenas,
                                df_sanity_check,
                                df_pareto_aberturas,
                                df_aberturas_clusterizadas,
                                data_corte,
                                topo,
                                 tipo_de_tof,
                                 etapas_vol,
                                 etapas_conv,
                                max_origin,
                                conversoes,
                                qtd_semanas_media,  # Também será o lag fixo
                                max_lag_test,       # Qual a quantidade maxima de lags testar nas series exogenas
                                max_lags, # Inteiro definindo o número máximo de lags pré-estabelecido
                                remover_outliers,
                                retorna_parametros,
                                 nome_do_painel_de_controle,
                                 nome_e_arquivo_de_destino_backup_parametros,
                                 trained_model_df,
                                 usar_backup,
                                  estabilizar_cohorts,
                                  incluir_exogs_sazonais,
                                  premissa_dummy,
                                  num_generations, # Number of generations (quanto maior mais tempo a evolução prossegue)
                                  num_parents_mating, # Number of solutions to be selected as parents
                                  sol_per_pop, # Number of solutions (i.e. chromosomes) within the population.  PyGAD creates an initial population using the sol_per_pop and num_genes parameters.
                                  parent_selection_type, # The parent selection type. Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection).
                                  keep_parents, # Number of parents to keep in the current population. -1 (default) means to keep all parents in the next population. 0 means keep no parents in the next population. A value greater than 0 means keeps the specified number of parents in the next population. Note that the value assigned to keep_parents cannot be < - 1 or greater than the number of solutions within the population sol_per_pop.
                                  crossover_type, # Type of the crossover operation. Supported types are single_point (for single-point crossover), two_points (for two points crossover), uniform (for uniform crossover), and scattered (for scattered crossover)
                                  mutation_type, # Type of the mutation operation. Supported types are random (for random mutation), swap (for swap mutation), inversion (for inversion mutation), scramble (for scramble mutation), and adaptive (for adaptive mutation).
                                  mutation_percent_genes, #  Percentage of genes to mutate. It defaults to the string "default" which is later translated into the integer 10 which means 10% of the genes will be mutated. It must be >0 and <=100. Out of this percentage, the number of genes to mutate is deduced which is assigned to the mutation_num_genes parameter. The mutation_percent_genes parameter has no action if mutation_probability or mutation_num_genes exist.
                                  fitness_func,   # Nome da função que vai calcular a qualidade da regressão
                                 overfitting_vs_underfitting,
                                 fit_intercept):



  # Definimos o cabeçalho da base
  cb_df = list(df_completo.columns.values)

  # Definimos os nomes das aberturas
  aberturas = cb_df[1:cb_df.index('Etapa')]

  # Definimos o nome da coluna que contém as datas
  col_data = cb_df[0]

  # Definindo a base de parâmetros final
  if len(trained_model_df) > 0 and usar_backup:
    trained_model_df = trained_model_df.rename(columns={'etapa':'Etapa','endógena':'Endógena','exógena':'Exógena','maturação':'Maturação'})

  if len(trained_model_df) != 0:
    out_parametros = trained_model_df
  else:
    out_parametros =  pd.DataFrame()


  # Vamos definir quais são as séries exógenas
  #-------------------------------------------------------------------------------------------
  lista_conversoes_vol = list(range(max_origin+1)) + ['Coincident']
  lista_conversoes_vol = [str(c) for c in lista_conversoes_vol]
  lista_conversoes_p = ["%__"+c for c in lista_conversoes_vol] + ['%__Volume Aberta']
  lista_conversoes_s = ["s__"+c for c in lista_conversoes_vol]
  lista_volumes = ['Volume Aberta','Volume']
  lista_sazonais = ['ordem_semana','Tempo Numérico','Year','Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
  lista_nao_exogenas = lista_conversoes_vol+lista_conversoes_p+lista_conversoes_s+lista_volumes+aberturas+[col_data,'Etapa']

  # As exógenas iniciais são todas as colunas da base que não contenham alguma variável endógena ou abertura
  exog_list_inicial = [e for e in cb_df if e not in lista_nao_exogenas]

  # Separamos as exógenas que serão transformadas em dummy
  lista_exogenas_dummy = [e for e in exog_list_inicial if '(dummy)' in e]

  # Criamos uma lista de exógenas que não contém as dummy para serem transformadas
  exog_list_inicial_sem_dummy = [e for e in exog_list_inicial if e not in lista_exogenas_dummy]
  # Vamos remover as séries sazonais dessa lista também, pois não faz sentido usar o lag ou diferencial
  # de um marcador de sazonalidade:
  exog_list_inicial_sem_dummy = [e for e in exog_list_inicial_sem_dummy if e not in lista_sazonais]

  exog_list = exog_list_inicial_sem_dummy.copy()

  # Adicionando os lags:
  for lag in range(1,int(max_lag_test)+1):
    exog_list_lag = [l+"___l___"+str(lag) for l in exog_list_inicial_sem_dummy]
    exog_list = exog_list + exog_list_lag

  # Adicionando os diferenciais:
  exog_list_d = [l+"___d" for l in exog_list_inicial_sem_dummy]

  # Adicionando os logs
  exog_list_log = [l+"___log" for l in exog_list_inicial_sem_dummy]

  # Definimos se queremos incluir as séries sazonais ou se mantemos apenas as exógenas definidas na base ou nos inputs.
  exog_list = exog_list + exog_list_d + exog_list_log + lista_exogenas_dummy #+ lista_sazonais
  if incluir_exogs_sazonais:
    exog_list = exog_list + lista_sazonais
  else:
    #exog_list = list(set(exog_list) - set(['Feriado (dummy)']))
    exog_list = exog_list

  # para cada etapa de conversão que identificamos, vamos calcular os volumes das cohorts e o volume
  # coincident da etapa seguinte utilizando a função auxiliar de progerssão do funil
  #_________________________________________________________________________________________________
  for e in etapas_conv:

    # Primeiro, selecionamos do dataframe histórico e futuro exógeno completo somente a etapa
    # do funil que estamos trabalhando:
    df_completo_etapa = df_completo.loc[df_completo['Etapa'] == e]

    # Definimos a base de parâmetros da etapa
    out_parametros_e =  pd.DataFrame()

    # Abaixo vamos definir as aberturas a serem filtradas:

    # Criamos uma matriz contendo apenas as combinações de aberturas da base.
    # Serão essas combinações de aberturas únicas que vamos percorrer e projetar
    lista_das_aberturas = df_completo_etapa.groupby(aberturas, as_index=False)['Volume'].sum()

    lista_das_aberturas = lista_das_aberturas[aberturas].to_numpy()


    #-------------------------------------------------------------------------------------------------
    # Para cada abertura da lista_das_aberturas
    lista_de_etapas = etapas_conv
    tempo_do_loop = 0
    posi_etapa_na_lista = lista_de_etapas.index(e)
    n_etapas = len(lista_de_etapas)
    lista_de_etapas_upper = [l.upper() for l in lista_de_etapas]
    for a in range(len(lista_das_aberturas[:,0])):

      clear___output(flag_clear)
      print("Calculando...")
      if posi_etapa_na_lista == n_etapas-1:
        print("Etapa do Funil:",colored(' '.join(lista_de_etapas_upper[:posi_etapa_na_lista]), 'green'),colored(e.upper(), 'yellow'))
      else:
        print("Etapa do Funil:",colored(' '.join(lista_de_etapas_upper[:posi_etapa_na_lista]), 'green'),colored(e.upper(), 'yellow'),colored(' '.join(lista_de_etapas_upper[posi_etapa_na_lista+1:]), 'red'))
      print("Abertura:",lista_das_aberturas[a])
      print("_______________________________________________________________________________________")
      print(round(100*((a+1)/len(lista_das_aberturas[:,0])),1),"%",colored("("+str(a+1)+")",'grey'),"de",len(lista_das_aberturas[:,0]),"aberturas")
      tempo_restante = (tempo_do_loop / (a+1)) * (len(lista_das_aberturas[:,0]) - (a+1))
      tempo_restante_total = (tempo_do_loop / (a+1)) * (len(lista_das_aberturas[:,0])*(n_etapas-posi_etapa_na_lista) - (a+1))


      # Caso a abertura já tenha sido treinada, vamos pular para a próxima:
      teste_abertura = True
      if len(out_parametros) != 0 and usar_backup:
        ab=0
        if e.lower() in out_parametros['Etapa'].values:
          out_etapa = out_parametros.loc[out_parametros['Etapa'] == e]
          aa = 0
          for abertura_treinada in aberturas:
            if str(lista_das_aberturas[a][aa]) in out_etapa[abertura_treinada].astype(str).values:
              out_etapa = out_etapa.loc[out_etapa[abertura_treinada] == str(lista_das_aberturas[a][aa])]
              ab=ab+1

            else:
              ab=-1
            aa=aa+1

          if ab == len(aberturas):
            print(e,'###### Já Treinou #######')
            print("_______________________________________________________________________________________")
            teste_abertura = False


      # Caso a abertura já tenha sido treinada, vamos pular para a próxima:
      if teste_abertura:


        if a == 0:
          print("Tempo restante na etapa do funil: Calculando...")
        else:
          print("Tempo restante na etapa do funil:",colored(str(timedelta(seconds=round(tempo_restante,0))),'yellow'))
          print("Tempo restante total:            ",colored(str(timedelta(seconds=round(tempo_restante_total,0))),'red'))
          print("_______________________________________________________________________________________")

        #-------------------------------------------------------------------------------------------





        time_start = timeit.default_timer() # registramos o início da execução

        # Vamos criar uma base contendo somente a etapa e a abertura investigados
        df_completo_abertura = df_completo_etapa.copy()
        df_pareto_aberturas_abertura = df_pareto_aberturas.copy()
        df_aberturas_clusterizadas_abertura = df_aberturas_clusterizadas.copy()

        # Filtramos somente a abertura a ser projetada
        for i in range(len(aberturas)):
          # Selecionamos o histórico
          df_completo_abertura = df_completo_abertura.loc[df_completo_abertura[aberturas[i]] == lista_das_aberturas[a][i]]

          # Selecionamos o pareto na base de pareto
          if len(df_pareto_aberturas)>0:
            df_pareto_aberturas_abertura = df_pareto_aberturas_abertura.loc[df_pareto_aberturas_abertura[aberturas[i]] == lista_das_aberturas[a][i]]

          # Selecionamos a abertura na base clusterizada
          if len(df_aberturas_clusterizadas) > 0:
            df_aberturas_clusterizadas_abertura = df_aberturas_clusterizadas_abertura.loc[df_aberturas_clusterizadas_abertura[aberturas[i]] == lista_das_aberturas[a][i]]


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

        maturacao = tempo_maturacao(historico_df=df_completo_abertura,
                                    max_origin=max_origin,
                                    col_data = col_data,
                                    data_corte = data_corte,
                                    qtd_semanas_media=qtd_semanas_media)

        data_maturacao = data_corte-pd.Timedelta(maturacao*7, unit='D')

        # Definimos se vamos realizar o AG de acordo com a categoria da abertura na análise de pareto
        if len(df_pareto_aberturas) == 0:
          categoria_pareto = 0
        elif len(df_pareto_aberturas_abertura) > 0:
          categoria_pareto = df_pareto_aberturas_abertura['pareto_'+etapas_conv[0]].values[0]
        else:
          categoria_pareto = 3

        # definimos se a abertura é ruim de projetar ou não
        if len(df_aberturas_clusterizadas_abertura) > 0:
          abertura_outlier = df_aberturas_clusterizadas_abertura['outlier'].values[0]
        elif len(df_aberturas_clusterizadas) > 0:
          abertura_outlier = 1
        else:
          abertura_outlier = 0

        len_hist_maturado = len(df_completo_abertura.loc[df_completo_abertura[col_data] <= data_maturacao])
        vol_hist_maturado_total = np.sum(df_completo_abertura.loc[df_completo_abertura[col_data] <= data_maturacao,['Volume']].fillna(0).values)


        #_______________________________________________________________________________________________
        # Só prosseguimos com a projeção se existirem pontos suficientes no histórico:
        if len_hist_maturado > 0 and vol_hist_maturado_total > len_hist_maturado/3 and abertura_outlier == 0 and categoria_pareto < 3:
          # Caso o histórico exista, mas ainda assim tiver poucos pontos não realizamos a otimização
          if len(df_completo_abertura.loc[df_completo_abertura[col_data] <= data_maturacao]) > qtd_semanas_media:

            # Realizamos a projeção de todas as variáveis da abertura (Volumes de topo, shares de cohorts, cohort aberta e cohorts fechadas)
            out_parametros_a = aplica_teste(df_completo = df_completo_abertura,      # DataFrame filtrado na etapa e abertura
                                            df_inputs_exogenas = df_inputs_exogenas,
                                            df_sanity_check = df_sanity_check,
                                            col_data = col_data,   # DataFrame filtrado, somente datas e valores
                                            data_corte = data_corte,
                                            data_maturacao = data_maturacao,
                                            abertura = lista_das_aberturas[a],
                                            aberturas = aberturas,
                                            etapa = e,                    # String com o nome da etapa do funil
                                            topo = topo,                      # Lista com o nome das etapas do ToF.
                                            tipo_de_tof = tipo_de_tof,
                                            max_origin = max_origin,               # int indicando qual a maior cohort do histórico
                                            conversoes = conversoes,               # Lista com os nomes das conversões sem '%__Volume Aberta' ('s__0','s__1',...,'s__Coincident')
                                            qtd_semanas_media = qtd_semanas_media,
                                            remover_outliers = remover_outliers,
                                            retorna_parametros = retorna_parametros,
                                            exog_list = exog_list,
                                            max_lag_test = max_lag_test,       # Qual a quantidade maxima de lags testar nas series exogenas
                                            estabilizar_cohorts = estabilizar_cohorts,
                                              premissa_dummy = premissa_dummy,
                                              max_lags = max_lags, # Inteiro definindo o número máximo de lags pré-estabelecido
                                              num_generations = num_generations, # Number of generations (quanto maior mais tempo a evolução prossegue)
                                              num_parents_mating = num_parents_mating, # Number of solutions to be selected as parents
                                              sol_per_pop = sol_per_pop, # Number of solutions (i.e. chromosomes) within the population.  PyGAD creates an initial population using the sol_per_pop and num_genes parameters.
                                              parent_selection_type = parent_selection_type, # The parent selection type. Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection).
                                              keep_parents = keep_parents, # Number of parents to keep in the current population. -1 (default) means to keep all parents in the next population. 0 means keep no parents in the next population. A value greater than 0 means keeps the specified number of parents in the next population. Note that the value assigned to keep_parents cannot be < - 1 or greater than the number of solutions within the population sol_per_pop.
                                              crossover_type = crossover_type, # Type of the crossover operation. Supported types are single_point (for single-point crossover), two_points (for two points crossover), uniform (for uniform crossover), and scattered (for scattered crossover)
                                              mutation_type = mutation_type, # Type of the mutation operation. Supported types are random (for random mutation), swap (for swap mutation), inversion (for inversion mutation), scramble (for scramble mutation), and adaptive (for adaptive mutation).
                                              mutation_percent_genes = mutation_percent_genes, #  Percentage of genes to mutate. It defaults to the string "default" which is later translated into the integer 10 which means 10% of the genes will be mutated. It must be >0 and <=100. Out of this percentage, the number of genes to mutate is deduced which is assigned to the mutation_num_genes parameter. The mutation_percent_genes parameter has no action if mutation_probability or mutation_num_genes exist.
                                              fitness_func = fitness_func, # Nome da função que vai calcular a qualidade da regressão
                                            overfitting_vs_underfitting = overfitting_vs_underfitting,
                                            fit_intercept = fit_intercept)





            # Concatenamos os parâmetros da abertura e adicionamos a maturação:
            out_parametros_e = pd.concat([out_parametros_e,out_parametros_a])
            out_parametros_e['Maturação'] = maturacao
            out_parametros_e['Etapa'] = e


            # Vamos salvar o progresso a cada 10 aberturas calculadas
            if a%10 == 0:
              print('Salvando modelos parciais...')
              out_parcial = pd.concat([out_parametros,out_parametros_e])

              exportar_base(base_df = out_parcial,                     # DataFrame
                          nome_do_painel_de_controle = nome_do_painel_de_controle,  # String com o nome do arquivo sheets com o painel de controle
                          arquivo_de_destino = nome_e_arquivo_de_destino_backup_parametros[1],          # String com o nome do arquivo sheets de destino ou o caminho do Google Drive
                          nome_da_aba = nome_e_arquivo_de_destino_backup_parametros[0],                 # String com o nome da aba no arquivo sheets de destino ou nome final do CSV que será salvo
                          substituir = True)                 # Booleano indicando se o arquivo CSV será salvo por cima de um existente com mesmo nome ou se será salva uma cópia com nome diferente



        time_end = timeit.default_timer()
        tempo_do_loop = tempo_do_loop + (time_end - time_start)



    # Adicionando a base de parâmetros com a da etapa anterior já calculada:
    out_parametros = pd.concat([out_parametros,out_parametros_e])



  cb_out = list(out_parametros.columns.values)
  if len(cb_out) > 0:
    out_parametros = out_parametros[['Etapa']+aberturas+['Maturação','Endógena','Exógena','slope','intercept','error']]
    out_parametros['slope'] = out_parametros['slope'].astype(float)
    out_parametros = out_parametros.loc[out_parametros['slope'] != 0]



  # Retornamos a base final, que contém a projeção de todas as aberturas de uma etapa do funil específica
  return out_parametros
