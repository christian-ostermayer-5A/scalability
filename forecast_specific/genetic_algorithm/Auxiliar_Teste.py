#@title Def Auxiliar_Teste (AG 2)
import pandas as pd
from inputs_exogenas import *
from transforma_dummy import *
from transforma_exogs_2 import *
from  algoritmo_genetico import *
from acf_pac import *


def Auxiliar_Teste(df_completo, # DF filtrado etapa abertura endog e data, ordenado com a data mais antiga no topo
                   df_inputs_exogenas, # DF contendo os inputs de quais exógenas devem ser usadas obrigatoriamente
                   df_sanity_check,
                   aberturas,   # Nome das colunas com as aberturas
                      abertura,
                      etapa,
                      col_data,
                      endogenous,   # String com o nome da variável endog
                      qtd_semanas_media,
                      max_origin,
                      remover_outliers,
                      retorna_parametros,
                      exog_list,
                      max_lag_test,
                      estabilizar_cohorts,
                        premissa_dummy,
                        max_lags, # Inteiro definindo o número máximo de lags pré-estabelecido
                        num_generations, # Number of generations (quanto maior mais tempo a evolução prossegue)
                        num_parents_mating, # Number of solutions to be selected as parents
                        sol_per_pop, # Number of solutions (i.e. chromosomes) within the population.  PyGAD creates an initial population using the sol_per_pop and num_genes parameters.
                        parent_selection_type, # The parent selection type. Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection).
                        keep_parents, # Number of parents to keep in the current population. -1 (default) means to keep all parents in the next population. 0 means keep no parents in the next population. A value greater than 0 means keeps the specified number of parents in the next population. Note that the value assigned to keep_parents cannot be < - 1 or greater than the number of solutions within the population sol_per_pop.
                        crossover_type, # Type of the crossover operation. Supported types are single_point (for single-point crossover), two_points (for two points crossover), uniform (for uniform crossover), and scattered (for scattered crossover)
                        mutation_type, # Type of the mutation operation. Supported types are random (for random mutation), swap (for swap mutation), inversion (for inversion mutation), scramble (for scramble mutation), and adaptive (for adaptive mutation).
                        mutation_percent_genes, #  Percentage of genes to mutate. It defaults to the string "default" which is later translated into the integer 10 which means 10% of the genes will be mutated. It must be >0 and <=100. Out of this percentage, the number of genes to mutate is deduced which is assigned to the mutation_num_genes parameter. The mutation_percent_genes parameter has no action if mutation_probability or mutation_num_genes exist.
                        fitness_func, # Nome da função que vai calcular a qualidade da regressão
                   overfitting_vs_underfitting,
                   fit_intercept):




  out_parametros = pd.DataFrame(columns = aberturas+['Endógena'])



  # Vamos determinar o número de lags de acordo com a função de auto-correlação:
  #-------------------------------------------------------------------------------------------------
  ar,ma = acf_pac(df_completo,endogenous)

  if ar > qtd_semanas_media:
    ar = qtd_semanas_media

  # Caso seja a projeção da cohort aberta, queremos introduzir mais estabilidade, então vamos forçar o modelo
  # a testar mais parâmetros de lag do que a função e autocorrelação indicaria:
  if endogenous == '%__Volume Aberta' and ar < qtd_semanas_media and estabilizar_cohorts == 'Sim':
    ar = qtd_semanas_media

  pcf_lag = ar

  # Vamos checar se o número máximo de lags não é superior ao tamanho da base histórica mais o número mínimo de pontos para treinar o modelo:
  tamanho_historico = len(df_completo[endogenous].values)

  if tamanho_historico <= pcf_lag + qtd_semanas_media and tamanho_historico > qtd_semanas_media:
    pcf_lag = tamanho_historico-qtd_semanas_media-1
  elif tamanho_historico <= pcf_lag + qtd_semanas_media:
    pcf_lag = 1


  # Selecionar as séries exógenas que devem estar obrigatoriamente presentes em qualquer solução do AG:
  #-------------------------------------------------------------------------------------------------
  exogs_obrigatorias = inputs_exogenas(df_inputs_exogenas = df_inputs_exogenas,
                                      abertura = abertura,
                                      etapa = etapa,
                                      endogenous = endogenous,
                                      exog_list = [])


  # Caso seja a projeção da cohort aberta ou das fechadas, verificamos se o objetivo da projeção
  # é simplificar essas métricas para otimizar a estabilidade da projeção dos volumes. Se for esse
  # o caso, vamos utilizar como séries exógenas apenas aquelas indicadas nos inputs de exógenas (exogs_obrigatorias):
  if endogenous != 'Volume' and estabilizar_cohorts == 'Sim':
    exog_list = exogs_obrigatorias
    # Caso seja o share da W1, vamos adicionar o share da W0 nas séries obrigatórias:
    if endogenous == 's__1':
      exog_list = exog_list + ['s__0']
      exogs_obrigatorias = exogs_obrigatorias + ['s__0']
      exog_list = list(dict.fromkeys(exog_list)) # remove duplicadas
      exogs_obrigatorias = list(dict.fromkeys(exogs_obrigatorias)) # remove duplicadas
    # Caso não tenha exogs obrigatórias, adicionamos unicamente a de volume:
    if len(exog_list) == 0:
      exog_list = ['Volume']
      if endogenous == 's__1':
        exog_list = exog_list + ['s__0']


  # Calculando as séries exógenas e endógenas transformadas:
  #-------------------------------------------------------------------------------------------------

  # Vamos separar as exógenas dummy
  col_exogs_dummy = [l for l in exog_list if '(dummy)' in l]
  col_exogs_dummy_t = [l+"_t" for l in col_exogs_dummy]

  col_exogs_dummy_obrigatorias = [l for l in exogs_obrigatorias if '(dummy)' in l]
  col_exogs_dummy_obrigatorias_t = [l+"_t" for l in col_exogs_dummy_obrigatorias]

  exogs_totais_dummy = col_exogs_dummy_obrigatorias+col_exogs_dummy
  exogs_totais_dummy_t = col_exogs_dummy_obrigatorias_t+col_exogs_dummy_t
  exogs_totais_dummy = list(dict.fromkeys(exogs_totais_dummy)) # remove duplicadas
  exogs_totais_dummy_t = list(dict.fromkeys(exogs_totais_dummy_t)) # remove duplicadas

  # Transforma dummy:
  df_completo = transforma_dummy(df = df_completo, # Dataframe filtrado (somente uma etapa e uma abertura especifica, e sem as colunas das mesmas. Somente data e valores)
                                  exogenous_dummy = exogs_totais_dummy, # Lista com o nome das variaveis exogenas que são dummy
                                  endogenous = endogenous, # variável endógena
                                  n_avg = premissa_dummy, # número de pontos passados usados na média móvel da endógena
                                  col_data = col_data) # String com o nome da coluna de datas

  exog_list = exog_list + col_exogs_dummy_t
  exogs_obrigatorias = exogs_obrigatorias + col_exogs_dummy_obrigatorias_t
  # Vamos remover da lista de exogs as dummy não transformadas:
  exog_list = [l for l in exog_list if l not in col_exogs_dummy]
  exogs_obrigatorias = [l for l in exogs_obrigatorias if l not in col_exogs_dummy_obrigatorias]

  # Vamos adicionar séries exógenas trasnformadas em relação à série dummy:
  for l in range(1,int(max_lag_test)+1):
    col_exogs_dummy_lag = [e+"___l___"+str(l) for e in col_exogs_dummy_t]
    exog_list = exog_list + col_exogs_dummy_lag

  col_exogs_dummy_d = [l+"___d" for l in col_exogs_dummy_t]

  exog_list = exog_list + col_exogs_dummy_d




  # Vamos separar as exógenas que devem ser transformadas
  #-------------------------------------------------------------------------------------------------
  exogs_totais = exogs_obrigatorias+exog_list
  exogs_totais = list(dict.fromkeys(exogs_totais)) # remove duplicadas
  if len(exogs_totais) == 0:
    exogs_totais = exogs_obrigatorias

  col_exogs_l_d = [l for l in exogs_totais if '___' in l]


  # Vamos separar as séries endógenas transformadas:
  endog_trans = []
  maior_lag = max(pcf_lag,max_lags)
  for l in range(1,maior_lag+1):
    endog_trans = endog_trans + [endogenous+"___l___"+str(l)]

  col_exogs_l_d = col_exogs_l_d + endog_trans

  df_completo =  transforma_exogs_2(df = df_completo, # Dataframe filtrado de dados futuros
                                              exogenous_l_d = col_exogs_l_d, # Lista com o nome das variaveis exogenas (com os suffixos "l" e "d")
                                              col_data = col_data) # String com o nome da coluna de datas


  # Primeiro vamos remover as séries obrigatórias da lista total de séries, pois não queremos treinar
  # um modelo com séries repetidas:
  #-------------------------------------------------------------------------------------------------
  exog_list = list(dict.fromkeys(exog_list)) # remove duplicadas
  exogs_obrigatorias = list(dict.fromkeys(exogs_obrigatorias)) # remove duplicadas
  if exog_list == exogs_obrigatorias:
    exogs_obrigatorias = []
  exog_list = list(set(exog_list) - set(exogs_obrigatorias))


  '''
  # Remover outliers somente com base na endógena:
  #-------------------------------------------------------------------------------------------------
  if remover_outliers != 'Não':
    df_completo,flag,base_outliers =  aplica_removedor_outliers(df = df_completo,
                                                  endogenous = endogenous,
                                                  exog_list = [],
                                                  threshold = 0.01,
                                                  col_data = col_data,
                                                  method = remover_outliers)
    if not flag:
      mensagem = 'Não Removeu Outliers'
      #print('Não Removeu Outliers')
  '''


  # Aplicamos o algoritmo genético para encontrar as melhores séries exógenas
  #-------------------------------------------------------------------------------------------------
  out_parametros =  algoritmo_genetico(df_completo = df_completo, # DF filtrado somente etapa e abertura e endógena e data, já com todas as combinações possíveis de exógenas (inclusive as endog transformadas)
                                      col_exogs = exog_list,   # Lista com o nome das colunas de exogs
                                      df_sanity_check = df_sanity_check,
                                      exogs_obrigatorias = exogs_obrigatorias, # Lista com as exógenas que devem estar obrigatoriamente no modelo de AG
                                      col_endog = endogenous,   # String com o nome da variável endog
                                      retorna_parametros = retorna_parametros, # Bool que define se os parâmetros serão retornados com ou sem estimativa de erro
                                      pcf_lag = pcf_lag, # Inteiro indicando qual o número de lags de acordo com a função de auto-correlação
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




  # Adicionar nome da abertura e endógena no out_parametros:
  out_parametros[aberturas] = abertura
  out_parametros['Endógena'] = endogenous



  return out_parametros
