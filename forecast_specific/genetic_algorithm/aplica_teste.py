#@title Def aplica_teste (AG 1)
import pandas as pd
from Auxiliar_Teste import *

#@title Def aplica_teste (AG 1)


def aplica_teste(df_completo,      # DataFrame filtrado na etapa e abertura
            df_inputs_exogenas,
            df_sanity_check,
            col_data,   # DataFrame filtrado, somente datas e valores
            data_corte,
            data_maturacao,
            abertura,
            aberturas,
            etapa,                    # String com o nome da etapa do funil
            topo,                      # Lista com o nome das etapas do ToF.
            tipo_de_tof,
            max_origin,               # int indicando qual a maior cohort do histórico
            conversoes,               # Lista com os nomes das conversões sem '%__Volume Aberta' ('s__0','s__1',...,'s__Coincident')
            qtd_semanas_media,
            remover_outliers,
            retorna_parametros,
            exog_list,
            max_lag_test,
            max_lags, # Inteiro definindo o número máximo de lags pré-estabelecido
            estabilizar_cohorts,
              premissa_dummy,
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


  out_parametros = pd.DataFrame()



  #-------------------------------------------------------------------------------------------------

  # Abaixo vamos projetar o volume de ToF e também os shares das cohorts fechadas.
  # Com base nos shares das cohorts fechadas vamos projetar a cohort aberta no histórico não maturado
  # Depois vamos usar o histórico completo para projetar a cohort aberta e calcular a projeção das fechadas
  #_________________________________________________________________________________________________
  # Caso a etapa sendo projetada inclui o ToF, faremos uma projeção específica:
  if etapa.split('2')[0] in topo and tipo_de_tof != 'Input Externo' and tipo_de_tof != 'AG - Somente Cohort Aberta':

    endogenous = 'Volume'

    # Selecionamos apenas a base histórica de volume
    df_completo_datas = df_completo.loc[df_completo[col_data] <= data_corte]

    # Realizamos a projeção
    out_parametros =  Auxiliar_Teste(df_completo = df_completo_datas, # DF filtrado etapa abertura endog e data, ordenado com a data mais antiga no topo
                                    df_inputs_exogenas = df_inputs_exogenas,
                                    df_sanity_check = df_sanity_check,
                                    aberturas = aberturas,   # Nome das colunas com as aberturas
                                    abertura = abertura,
                                    etapa = etapa,
                                    col_data = col_data,
                                    endogenous = endogenous,   # String com o nome da variável endog
                                    qtd_semanas_media = qtd_semanas_media,
                                    max_origin = max_origin,
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



  if tipo_de_tof != 'AG - Somente ToF':
    #-------------------------------------------------------------------------------------------------
    # Para cada cohort fechada, vamos projetar o share a partir do histórico maturado:
    lista_out_parametros = [out_parametros]
    if tipo_de_tof != 'AG - Somente Cohort Aberta':

      for c in conversoes:

        endogenous = c

        # Selecionamos apenas a base histórica de volume
        df_completo_datas = df_completo.loc[df_completo[col_data] <= data_maturacao]


        # Realizamos a projeção
        out_parametros_c =  Auxiliar_Teste(df_completo = df_completo_datas, # DF filtrado etapa abertura endog e data, ordenado com a data mais antiga no topo
                                        df_inputs_exogenas = df_inputs_exogenas,
                                        df_sanity_check = df_sanity_check,
                                        aberturas = aberturas,   # Nome das colunas com as aberturas
                                        abertura = abertura,
                                        etapa = etapa,
                                        col_data = col_data,
                                        endogenous = endogenous,   # String com o nome da variável endog
                                        qtd_semanas_media = qtd_semanas_media,
                                        max_origin = max_origin,
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



        # Vamos adicionar os parâmetros calculados dos shares de conversão aos de volume:
        lista_out_parametros = lista_out_parametros + [out_parametros_c.copy()]

    out_parametros = pd.concat(lista_out_parametros)


    # Vamos projetar a cohort Aberta a partir do histórico maturado
    #_________________________________________________________________________________________________

    endogenous = '%__Volume Aberta'

    # Selecionamos apenas a base histórica de volume
    df_completo_datas = df_completo.loc[df_completo[col_data] <= data_maturacao]

    # Realizamos a projeção
    out_parametros_a =  Auxiliar_Teste(df_completo = df_completo_datas, # DF filtrado etapa abertura endog e data, ordenado com a data mais antiga no topo
                                      df_inputs_exogenas = df_inputs_exogenas,
                                      df_sanity_check = df_sanity_check,
                                      aberturas = aberturas,   # Nome das colunas com as aberturas
                                      abertura = abertura,
                                      etapa = etapa,
                                      col_data = col_data,
                                      endogenous = endogenous,   # String com o nome da variável endog
                                      qtd_semanas_media = qtd_semanas_media,
                                      max_origin = max_origin,
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




    # Vamos adicionar os parâmetros calculados da cohort aberta aos de volume e shares:
    out_parametros = pd.concat([out_parametros,out_parametros_a])

  #_________________________________________________________________________________________________



  return out_parametros
