#@title Def algoritmo_genetico (AG 3)
from sklearn.linear_model import LinearRegression
import numpy as np
#from fitness_func_qualidade_do_modelo import *
from parametros_modelo import *
#import warnings
#warnings.simplefilter("ignore", UserWarning)
#warnings.simplefilter("error", category=Warning)
import pygad
        
#@title Def algoritmo_genetico (AG 3)

def algoritmo_genetico(df_completo, # DF filtrado somente etapa e abertura e endógena e data, já com todas as combinações possíveis de exógenas (inclusive as endog transformadas)
                      col_exogs,   # Lista com o nome das colunas de exogs
                      exogs_obrigatorias, # Lista com as exógenas que devem estar obrigatoriamente no modelo de AG
                      df_sanity_check,
                      col_endog,   # String com o nome da variável endog
                      retorna_parametros, # Bool que define se os parâmetros serão retornados com ou sem estimativa de erro
                       pcf_lag, # Inteiro indicando qual o número de lags de acordo com a função de auto-correlação
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

  # Vamos definir o baseline de da qualidade do modelo medindo o R2 ajustado para o caso de um modelo
  # simples de média móvel ponderada para um número de termos da endógena dada pela pcf_lag:
  #_________________________________________________________________________________________________

  exogs_media_movel = []
  if pcf_lag == 0:
    exogs_media_movel = [col_endog+"___l___1"]
  else:
    for l in range(1,pcf_lag+1):
      exogs_media_movel = exogs_media_movel + [col_endog+"___l___"+str(l)]

  # Limpamos os nan's da base:
  df_limpa = df_completo.copy()
  df_limpa.dropna(axis=0, how='any', inplace=True)

  # Definimos as bases para treinar o modelo:
  X_train = df_limpa[exogs_media_movel]
  y_train = df_limpa[[col_endog]]

  # Definimos o modelo linear
  model = LinearRegression(fit_intercept=fit_intercept)

  # Treinamos o modelo

  model.fit(X_train, y_train)

  # Vamos definir o valor a ser otimizado como sendo o R2 ajustado do modelo:
  #fitness_baseline = 1 - (1-model.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
  fitness_baseline = model.score(X_train, y_train)

  # Algoritmo Genético:
  #_________________________________________________________________________________________________

  # Vamos tentar otimizar apenas as séries exógenas para projetar Volume, %_Volume Aberta, s__0 e s__1.
  # O restante das conversões possui impacto muito pequeno que não vale a pena ser otimizado
  if col_endog in ['Volume','%__Volume Aberta','s__0','s__1']:


    # Vamos iniciar definindo os genes dos cromossomos com base na lista de séries exógenas:
    # A primeira posição dos genes sempre vai conter a informação sobre os lags da endógena
    num_genes =  1+len(col_exogs)

    gene_type=int # Controls the gene type. It can be assigned to a single data type that is applied to all genes or can specify the data type of each individual gene.

    # Os genes podem ter os valores 0, 1 ou 2.
    # Para a primeira posição do cromossomo, se o gene for 0, não teremos lags da variável endógena.
    # Se for 1, teremos os lags de acordo com a pcf_lag. Se for 2 teremos a max_lags.
    # Para o restante das posições, 0 e 1 não ativamos o gene (não consideramos a série exógena daquela posição). 2 consideramos.
    init_range_low = 0 # The lower value of the random range from which the gene values in the initial population are selected
    init_range_high = 2 # The upper value of the random range from which the gene values in the initial population are selected.

    # Agora definimos as variáveis externas que serão usadas pela "fitness_function", que vai classificar os indivíduos
    # de acordo com o R2 ajustado do modelo multilinear determinado pelos seus genes.
    global endog_fitness
    global df_fitness
    global vetor_exogs_fitness
    global exogs_obrigatorias_fitness
    global pcf_fitness
    global max_lags_fitness
    global last_fitness
    global overfitting_vs_underfitting_fitness
    global df_sanity_check_fitness
    global fit_intercept_fitness

    endog_fitness = col_endog
    df_fitness = df_completo.copy()
    vetor_exogs_fitness = np.array(col_exogs)
    exogs_obrigatorias_fitness = exogs_obrigatorias
    pcf_fitness = pcf_lag
    max_lags_fitness = max_lags
    fitness_function = fitness_func_qualidade_do_modelo
    overfitting_vs_underfitting_fitness = overfitting_vs_underfitting
    df_sanity_check_fitness = df_sanity_check
    fit_intercept_fitness = fit_intercept


    saturacao = "saturate_" + str(int(num_generations/10))


    last_fitness = 0
    def callback_generation(ga_instance):
        global last_fitness

        generation=ga_instance.generations_completed
        solution,fitness,solution_idx=ga_instance.best_solution()

        if solution[0] == 0:
          endog_lag = 0
        elif solution[0] == 1:
          endog_lag = pcf_fitness
        else:
          endog_lag = max_lags_fitness

        n_series = endog_lag+len(np.where(np.array(solution[1:]) != 0)[0])

        fitness_r2 = np.exp((fitness + n_series * overfitting_vs_underfitting_fitness)/2)

        change=fitness_r2 - last_fitness
        last_fitness = fitness_r2

        print("\r", end="")
        print(col_endog,"| Baseline MA:",round(fitness_baseline,2),"| Generation:",generation,"| Fitness:",round(fitness_r2,2),"| n Series:",n_series,"| n lags:",endog_lag,"| Change:",round(change*100,2),"%","| Best Generation Exogs: ",list(vetor_exogs_fitness[np.where(np.array(solution[1:]) != 0)[0]]),end="")


    # Aqui definimos o modelo do algoritmo genético
    ga_instance = pygad.GA(num_generations=num_generations,
                          num_parents_mating=num_parents_mating,
                          fitness_func=fitness_function,
                          sol_per_pop=sol_per_pop,
                          num_genes=num_genes,
                          gene_type=gene_type,
                          init_range_low=init_range_low,
                          init_range_high=init_range_high,
                          parent_selection_type=parent_selection_type,
                          keep_parents=keep_parents,
                          crossover_type=crossover_type,
                          mutation_type=mutation_type,
                          mutation_percent_genes=mutation_percent_genes,
                           stop_criteria =  saturacao,
                           on_generation=callback_generation)

    # Executamos o GA:
    ga_instance.run()

    # Retornamos o melhor cromossomo (melhor indivíduo que resultou no maior R2 do ajuste multilienar)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    try:
      ga_instance.plot_fitness()
    except:
      print("Sem Gráfico")


    # Caso a melhor solução tenha sido a de fitness == -10000, significa que o modelo falhou:
    if solution_fitness <= -10000:
      if solution_fitness == -10000:  
              print("\n!!!!!!!!!!!!!!!!!!! Model Fail - len(exogs) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      if solution_fitness == -20000:  
              print("\n!!!!!!!!!!!!!!!!!!! Model Fail - sanity check !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")      
      if solution_fitness == -30000:  
              print("\n!!!!!!!!!!!!!!!!!!! Model Fail - fitness = nan !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")      
      solution = []
      endog_trans = []
      for l in range(1,pcf_lag+1):
        endog_trans = endog_trans + [col_endog+'___l___'+str(l)]
      exogs_finais = endog_trans
      if len(exogs_finais) == 0:
        exogs_finais = [col_endog+'___l___1']

    else:

      # Definimos as melhores séries exogs de acordo com o melhor cromossomo:
      posi_series = np.where(np.array(solution[1:]) != 0)[0]
      exogs_finais = list(vetor_exogs_fitness[posi_series]) + exogs_obrigatorias_fitness
      exogs_finais = list(dict.fromkeys(exogs_finais)) # remove duplicadas
      if len(exogs_finais) == 0:
        exogs_finais = exogs_obrigatorias_fitness
        if len(exogs_finais) == 0:
          exogs_finais = ['Volume']
          if col_endog == 's__1':
            exogs_finais = exogs_finais + ['s__0']
      print("Final Exogs: ",exogs_finais)

  else:

    solution = []

    # Caso não seja alguma das endógenas principais, vamos treinar o modelo com base nos lags da
    # função de auto correlação e com base no share da W0:
    endog_trans = []
    for l in range(1,pcf_lag+1):
      endog_trans = endog_trans + [col_endog+'___l___'+str(l)]

    exogs_finais = endog_trans + ['s__0']


  # Retorna parâmetros do melhor modelo treinado:
  #_________________________________________________________________________________________________


  # Vamos definir os lags da série endógena:
  if len(solution) != 0:
    endog_lag = []
    if solution[0] == 0:
      endog_lag = []
    elif solution[0] == 1:
      for l in range(1,pcf_fitness+1):
        endog_lag = endog_lag + [endog_fitness+"___l___"+str(l)]
    else:
      for l in range(1,max_lags_fitness+1):
        endog_lag = endog_lag + [endog_fitness+"___l___"+str(l)]


    # A lista final de séries exógenas é a soma das endógenas transformadas com as exógenas:
    exogs_finais = exogs_finais + endog_lag
    exogs_finais = list(dict.fromkeys(exogs_finais)) # remove duplicadas
    if len(exogs_finais) == 0:
      exogs_finais = endog_lag

  df_completo[exogs_finais] = df_completo[exogs_finais].replace(np.inf, 0, inplace=False).values
  df_completo[exogs_finais] = df_completo[exogs_finais].replace(-np.inf, 0, inplace=False).values
  df_completo[exogs_finais] = df_completo[exogs_finais].replace(np.nan, 0, inplace=False).values

  # Definimos as bases para treinar o modelo:
  X_train = df_completo[exogs_finais]
  y_train = df_completo[[col_endog]]

  # Definimos o modelo linear
  model = LinearRegression(fit_intercept=fit_intercept)

  # Treinamos o modelo
  model.fit(X_train, y_train)

  # Retornamos os parâmetros treinados do modelo:
  parametros = parametros_modelo(modelo = model,
                                  X = X_train,
                                  y = y_train,
                                  pivot = False,
                                  erro = True)

  parametros['Exógena'] = parametros.index.values

  return parametros


#_____________________________________________________________________________________________________________________________________________


'''
Essa função serve para retornar um número relacionado à qualidade
do ajuste multilinear dado um grupo de séries exógenas e lags da
endógena na forma de genes para o algoritmo genético otimizar.

"endog_fitness" é uma string com o nome da variável endógena, definido externamente antes de chamar essa função.

"df_fitness" é um DataFrame definido externamente antes de chamar essa função.
ele deve ser o histórico maturado da variável endógena a ser testada, juntamente
com todas as séries exógenas possíveis já transformadas além das endógenas possíveis já transformadas,
limpa, sem qualquer valor inf ou NaN.

"vetor_exogs_fitness" vetor com o nome de todas as séries exógenas possíveis e transformadas, definido externamente antes de chamar essa função

"pcf_fitness" é o número de lags referente ao resultado da função de auto correlação, definido externamente antes de chamar essa função

"max_lags_fitness" é o número de lags máximo pré-estabelecido, definido externamente antes de chamar essa função
'''

def fitness_func_qualidade_do_modelo(ga_instance,
                                      solution, # Lista de genes com inteiros que vairiam entre 0 e 2.
                                      solution_idx): # índice das soluções (não é utilizado)


    # Primeiramente, vamos definir quais são as séries exógenas a serem usadas com base no vetor
    # da solução proposta pelos genes (a primeira posição é sobre o lag da endógena):
    posi_series = np.where(np.array(solution[1:]) != 0)[0]
    exogs_fitness = list(vetor_exogs_fitness[posi_series])
    fitness = 0

    # Pode acontecer do cromossomo retornar nenhuma série exógena. Nesse caso o modelo vai
    # retornar um erro. Para evitar esse problema, checamos se as bases não estão vazias
    # antes de treinar o modelo. Se estiverem, retornamos "fitness" = 0:
    if len(exogs_fitness) == 0:
      #fitness = 0
      fitness = -10000

    else:


      # Vamos definir os lags da série endógena:
      endog_lag = []
      if solution[0] == 0:
        endog_lag = []
      elif solution[0] == 1:
        for l in range(1,pcf_fitness+1):
          endog_lag = endog_lag + [endog_fitness+"___l___"+str(l)]
      else:
        for l in range(1,max_lags_fitness+1):
          endog_lag = endog_lag + [endog_fitness+"___l___"+str(l)]

      # A lista final de séries exógenas é a soma das endógenas transformadas com as exógenas:
      exogs_fitness = exogs_fitness + endog_lag

      # Vamos adicionar as exógenas obrigatórias e removê-las caso estejam duplicadas:
      exogs_fitness = exogs_fitness + exogs_obrigatorias_fitness
      exogs_fitness = list(dict.fromkeys(exogs_fitness)) # remove duplicadas
      if len(exogs_fitness) == 0:
        exogs_fitness = exogs_obrigatorias_fitness

      df_fitness[exogs_fitness] = df_fitness[exogs_fitness].replace(np.inf, 0, inplace=False)
      df_fitness[exogs_fitness] = df_fitness[exogs_fitness].replace(-np.inf, 0, inplace=False)
      df_fitness[exogs_fitness] = df_fitness[exogs_fitness].replace(np.nan, 0, inplace=False)

      # Definimos o modelo linear
      model = LinearRegression(fit_intercept=fit_intercept_fitness)

      X_train = df_fitness[exogs_fitness]

      y_train = df_fitness[[endog_fitness]]

      model.fit(X_train, y_train)

      # Verificamos o sinal dos coeficientes e excluímos os modelos que não fazem sentido para o negócio:
      # Retornamos os parâmetros treinados do modelo:
      if len(df_sanity_check_fitness) > 0:
        parametros = parametros_modelo(modelo = model,
                                        X = X_train,
                                        y = y_train,
                                        pivot = False,
                                        erro = False)
        parametros['exógenas_originais'] = parametros.index
        parametros = parametros.reset_index()
        parametros['exógena'] = parametros['exógenas_originais'].apply(lambda x: x.split('___')[0])
        parametros['exógena'] = parametros['exógena'].apply(lambda x: x.split('_t')[0])
        parametros['slope'] = np.where(parametros['slope'] >= 0, 1, -1)
        sanity_check_merged = pd.merge(parametros,df_sanity_check_fitness,how='left',on='exógena')
        sanity_check_merged['sanity check'] = sanity_check_merged['slope'].values + sanity_check_merged['slope sanity check'].values
        sanity_check_merged['sanity check'] = sanity_check_merged['sanity check'].fillna(1)
        sanity_check_merged = sanity_check_merged.loc[sanity_check_merged['sanity check'] == 0]
        if len(sanity_check_merged) > 0:
          fitness = -10000

      if fitness > -10000:

        # Vamos definir o valor a ser otimizado como sendo o R2 ajustado do modelo:
        #r2_ajustado = 1 - (1-model.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
        r2_ajustado = model.score(X_train, y_train)
        n_series = len(exogs_fitness)
        fitness = 2*np.log(r2_ajustado) - overfitting_vs_underfitting_fitness * n_series
        if np.isnan(fitness):
          fitness = -10000

      #fitness = 1 - (1-model.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)

    return fitness
