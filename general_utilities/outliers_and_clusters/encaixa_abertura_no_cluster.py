#@title Def encaixa_abertura_no_cluster_por_relevancia
import numpy as np
import pandas as pd

def encaixa_abertura_no_cluster_por_relevancia(df_aberturas_ruins,
                                              df_base_clusters,
                                              df_base_chaves_classificadas,
                                              classifica_outlier,
                                              col_cluster,
                                              col_valores,
                                              aberturas):

  if len(df_aberturas_ruins) > 0:

    df_clusters_final = df_base_clusters.copy()
    df_base_clusters_aux = df_base_clusters.copy()
    df_base_clusters_aux['aux'] = 1
    df_base_clusters_aux = df_base_clusters_aux.groupby(col_cluster, as_index=False)[['aux']].sum()
    maior_cluster = df_base_clusters_aux.loc[df_base_clusters_aux['aux'] == df_base_clusters_aux['aux'].max()][col_cluster].values[0]

    base_qtd_aberturas_clusters = df_base_clusters.copy()
    base_qtd_aberturas_clusters['count'] = 1
    base_qtd_aberturas_clusters = base_qtd_aberturas_clusters.groupby([col_cluster],as_index=False)['count'].sum()

    l = 0
    for l in range(len(df_aberturas_ruins)):

      df_base_clusters_aux = df_base_clusters.copy()

      base_clusters = pd.DataFrame()
      abertura = df_aberturas_ruins[aberturas].iloc[l,:].values


      menor_relevancia = np.max(df_base_chaves_classificadas['score'].values)

      # Colunas auxiliares
      relevancia_aberturas = [x+"_relevancia" for x in aberturas]

      # Vamos encontrar a relevância de cada chave da abertura ruim para determinar o cluster:
      relevancia = []
      for i in range(len(aberturas)):
        score = df_base_chaves_classificadas.loc[(df_base_chaves_classificadas['abertura'] == aberturas[i]) & (df_base_chaves_classificadas['chaves'] == abertura[i])]['score'].values
        if len(score) == 0:
          relevancia = relevancia+[menor_relevancia]
        else:
          relevancia = relevancia+[score[0]]

      # Caso a combinação das chaves seja totalmente única, vamos alocar
      # a abertura ruim dentro do maior cluster:
      if np.sum(relevancia) == 3*menor_relevancia:
        cluster_mais_parecido = maior_cluster

      else:

        # Criar colunas na base clusterizada contendo a relevancia das aberturas:
        df_base_clusters_aux[relevancia_aberturas] = menor_relevancia
        for i in range(len(relevancia_aberturas)):
          df_base_clusters_aux.loc[df_base_clusters_aux[aberturas[i]] == abertura[i],[relevancia_aberturas[i]]] = relevancia[i]

        # Criamos uma coluna com a soma das relevancias de cada chave:
        df_base_clusters_aux['relevancia_somada'] = df_base_clusters_aux[relevancia_aberturas].sum(axis=1)
        df_teste = df_base_clusters_aux
        # Agrupamos os clusters pela relevancia somada média:
        df_base_clusters_aux = df_base_clusters_aux.groupby(col_cluster, as_index=False)[['relevancia_somada']].mean()

        # Adicionamos a informação da quantidade de aberturas por cluster
        df_base_clusters_aux  = pd.merge(df_base_clusters_aux,base_qtd_aberturas_clusters,how='left',on=col_cluster)

        # Calculamos o log + 1 da quantidade de aberturas por cluster
        df_base_clusters_aux['log'] = np.log2(df_base_clusters_aux['count'])+1

        # Calculamos a relevância final como sendo a relevancia somada média dividida pelo log da quantidade de aberturas naquele cluster.
        # Assim, se um cluster que contém poucas aberturas não terá tanta vantagem na relevancia média:
        df_base_clusters_aux['relevancia_final'] = df_base_clusters_aux['relevancia_somada'].values#/df_base_clusters_aux['log'].values

        # O cluster escolhido é aquele com a maior relevancia média:
        cluster_mais_parecido = df_base_clusters_aux.loc[df_base_clusters_aux['relevancia_final'] == df_base_clusters_aux['relevancia_final'].min(),[col_cluster]].values[0][0]

        if abertura[0] == '0':
          print(3*menor_relevancia,maior_cluster)
          print(abertura)
          print(relevancia)
          print(df_teste.loc[df_teste[col_cluster] == cluster_mais_parecido])
          print("------------------------------------------------------------------------")


      if classifica_outlier == 0:
        media_do_cluster = pd.DataFrame(np.array([df_aberturas_ruins.iloc[l,:].values]),
                                        columns = df_aberturas_ruins.columns.values,
                                        index = [1])



      else:
        media_do_cluster = df_base_clusters.loc[df_base_clusters[col_cluster] == cluster_mais_parecido,aberturas+col_valores+[col_cluster]]
        media_do_cluster = media_do_cluster.groupby([col_cluster],as_index=False)[col_valores].mean()
        media_do_cluster[aberturas] = list(abertura)

      media_do_cluster[col_cluster] = cluster_mais_parecido
      base_media_abertura_ruim = media_do_cluster
      base_media_abertura_ruim['outlier'] = classifica_outlier

      df_clusters_final = pd.concat([df_clusters_final,base_media_abertura_ruim])

  else:
    df_clusters_final = df_base_clusters


  return df_clusters_final

#____________________________________________________________________________________________
#@title Def encaixa_abertura_no_cluster_por_frequencia

def encaixa_abertura_no_cluster_por_frequencia(df_aberturas_ruins,
                                              df_base_clusters,
                                              aberturas,
                                              classifica_outlier,
                                              col_valores,
                                              col_cluster,
                                              abertura_principal = ''):

  etapas = col_valores

  df_clusters_final = df_base_clusters.copy()

  # Para cada abertura ruim, vamos encontrar qual cluster possui a maior compatibilidade de chaves das aberturas:
  base_qtd_aberturas_clusters = df_base_clusters.copy()
  base_qtd_aberturas_clusters['count'] = 1
  base_qtd_aberturas_clusters = base_qtd_aberturas_clusters.groupby([col_cluster],as_index=False)['count'].sum()

  l = 0
  for l in range(len(df_aberturas_ruins)):


    base_clusters = pd.DataFrame()
    abertura = df_aberturas_ruins[aberturas].iloc[l,:].values

    # Primeira tentativa será encontrando conjuntos de aberturas menos 1:
    #---------------------------------------------------------------------------------------------

    # Para cada combinação de n chaves das aberturas, onde n é o número de aberturas -1, vamos
    # identificar quais são os clusteres que mais possuem aberturas com essa combinação:
    for i in range(len(aberturas)):
      comb_aberturas = list(aberturas)
      comb_chaves = list(abertura)
      del comb_aberturas[i]
      del comb_chaves[i]
      #print("_________________________________")
      #print(comb_aberturas)
      #print(comb_chaves)

      df_merged_valores_originais_filtrado = df_base_clusters.copy()

      for a in range(len(comb_aberturas)):
        df_merged_valores_originais_filtrado = df_merged_valores_originais_filtrado.loc[df_merged_valores_originais_filtrado[comb_aberturas[a]] == comb_chaves[a]]

      #print(df_merged_valores_originais_filtrado)
      if len(df_merged_valores_originais_filtrado) > 0:
        # Se temos uma abertura principal, não queremos considerar nenhum cluster que não contenha a chave da abertura principal:
        if abertura_principal != '':
          if len(df_merged_valores_originais_filtrado.loc[df_merged_valores_originais_filtrado[abertura_principal] == abertura[aberturas.index(abertura_principal)]]) == 0:
            df_merged_valores_originais_filtrado['count'] = 0
          else:
            df_merged_valores_originais_filtrado['count'] = 1
        else:
          df_merged_valores_originais_filtrado['count'] = 1
        df_merged_valores_originais_filtrado = df_merged_valores_originais_filtrado.groupby([col_cluster],as_index=False)['count'].sum()
        base_clusters = pd.concat([base_clusters,df_merged_valores_originais_filtrado])


    # Se foi possível encontrar clusters com as combinações de aberturas:
    if len(base_clusters) > 0:
      # Calculamos quantos % da quantidade de aberturas de cada cluster as combinações
      # de chaves da abertura ruim representam:
      base_clusters = base_clusters.groupby([col_cluster],as_index=False)['count'].sum()
      base_clusters = pd.merge(base_clusters,base_qtd_aberturas_clusters,how='left',on=col_cluster)
      base_clusters['percent'] = base_clusters['count_x'].values/base_clusters['count_y'].values

      # Calculamos um score para que não seja selecionado o cluster com 1 abertura que represente
      # 100% da abertura ruim:
      base_clusters['log'] = np.log2(base_clusters['count_y']) # log da quantidade total de aberturas de cada cluster
      base_clusters['score'] = base_clusters['percent'].values*base_clusters['log'].values


      cluster_mais_parecido = base_clusters.loc[base_clusters['score'] == base_clusters['score'].max(),[col_cluster]].values[0][0]

      if classifica_outlier == 0:
        media_do_cluster = pd.DataFrame(np.array([df_aberturas_ruins.iloc[l,:].values]),
                                        columns = df_aberturas_ruins.columns.values,
                                        index = [1])



      else:
        media_do_cluster = df_base_clusters.loc[df_base_clusters[col_cluster] == cluster_mais_parecido,aberturas+etapas+[col_cluster]]
        media_do_cluster = media_do_cluster.groupby([col_cluster],as_index=False)[etapas].mean()
        media_do_cluster[aberturas] = list(abertura)

      media_do_cluster[col_cluster] = cluster_mais_parecido
      base_media_abertura_ruim = media_do_cluster
      base_media_abertura_ruim['outlier'] = classifica_outlier

      df_clusters_final = pd.concat([df_clusters_final,base_media_abertura_ruim])



    else:
      # Caso nenhuma combinação de aberturas da abertura ruim esteja presente na base realizada:
      # Segunda tentativa será encontrando somente 1 abertura de cada vez:
      #-------------------------------------------------------------------------------------------

      base_clusters = pd.DataFrame()
      abertura = df_aberturas_ruins[aberturas].iloc[l,:].values
      # Para cada combinação de n chaves das aberturas, onde n é o número de aberturas -1, vamos
      # identificar quais são os clusteres que mais possuem aberturas com essa combinação:
      for i in range(len(aberturas)):
        comb_aberturas = aberturas[i]
        comb_chaves = abertura[i]
        # Se temos uma abertura principal, quremos checar apenas o cluster com maior proporção dessa abertura:
        if abertura_principal != '':
          comb_aberturas = abertura_principal
          comb_chaves = abertura[aberturas.index(abertura_principal)]

        df_merged_valores_originais_filtrado = df_base_clusters.copy()

        df_merged_valores_originais_filtrado = df_merged_valores_originais_filtrado.loc[df_merged_valores_originais_filtrado[comb_aberturas] == comb_chaves]

        if len(df_merged_valores_originais_filtrado) > 0:
          df_merged_valores_originais_filtrado['count'] = 1
          df_merged_valores_originais_filtrado = df_merged_valores_originais_filtrado.groupby([col_cluster],as_index=False)['count'].sum()
          base_clusters = pd.concat([base_clusters,df_merged_valores_originais_filtrado])


      # Se foi possível encontrar clusters com as combinações de aberturas:
      if len(base_clusters) > 0:
        # Calculamos quantos % da quantidade de aberturas de cada cluster as combinações
        # de chaves da abertura ruim representam:
        base_clusters = base_clusters.groupby([col_cluster],as_index=False)['count'].sum()
        base_clusters = pd.merge(base_clusters,base_qtd_aberturas_clusters,how='left',on=col_cluster)
        base_clusters['percent'] = base_clusters['count_x'].values/base_clusters['count_y'].values

        # Calculamos um score para que não seja selecionado o cluster com 1 abertura que represente
        # 100% da abertura ruim:
        base_clusters['log'] = np.log2(base_clusters['count_y']) # log da quantidade total de aberturas de cada cluster
        base_clusters['score'] = base_clusters['percent'].values*base_clusters['log'].values


        cluster_mais_parecido = base_clusters.loc[base_clusters['score'] == base_clusters['score'].max(),[col_cluster]].values[0][0]
        if classifica_outlier == 0:
          media_do_cluster = pd.DataFrame(np.array([df_aberturas_ruins.iloc[l,:].values]),
                                          columns = df_aberturas_ruins.columns.values,
                                          index = [1])
        else:
          media_do_cluster = df_base_clusters.loc[df_base_clusters[col_cluster] == cluster_mais_parecido,aberturas+etapas+[col_cluster]]
          media_do_cluster = media_do_cluster.groupby([col_cluster],as_index=False)[etapas].mean()
          media_do_cluster[aberturas] = list(abertura)

        media_do_cluster[col_cluster] = cluster_mais_parecido
        base_media_abertura_ruim = media_do_cluster
        base_media_abertura_ruim['outlier'] = classifica_outlier

        df_clusters_final = pd.concat([df_clusters_final,base_media_abertura_ruim])

      # Caso exista uma abertura principal, mas a chave da abertura ruim não existe nas aberturas boas,
      # vamos ignorar a obrigatoriedade de encaixar a abertura ruim somente onde a chave é compatível
      # com a abertura obrigatória:
      elif abertura_principal != '':
        # Para cada combinação de n chaves das aberturas, onde n é o número de aberturas -1, vamos
        # identificar quais são os clusteres que mais possuem aberturas com essa combinação:
        for i in range(len(aberturas)):
          comb_aberturas = list(aberturas)
          comb_chaves = list(abertura)
          del comb_aberturas[i]
          del comb_chaves[i]
          #print("_________________________________")
          #print(comb_aberturas)
          #print(comb_chaves)

          df_merged_valores_originais_filtrado = df_base_clusters.copy()

          for a in range(len(comb_aberturas)):
            df_merged_valores_originais_filtrado = df_merged_valores_originais_filtrado.loc[df_merged_valores_originais_filtrado[comb_aberturas[a]] == comb_chaves[a]]

          #print(df_merged_valores_originais_filtrado)
          if len(df_merged_valores_originais_filtrado) > 0:
            df_merged_valores_originais_filtrado['count'] = 1
            df_merged_valores_originais_filtrado = df_merged_valores_originais_filtrado.groupby([col_cluster],as_index=False)['count'].sum()
            base_clusters = pd.concat([base_clusters,df_merged_valores_originais_filtrado])


        # Se foi possível encontrar clusters com as combinações de aberturas:
        if len(base_clusters) > 0:
          # Calculamos quantos % da quantidade de aberturas de cada cluster as combinações
          # de chaves da abertura ruim representam:
          base_clusters = base_clusters.groupby([col_cluster],as_index=False)['count'].sum()
          base_clusters = pd.merge(base_clusters,base_qtd_aberturas_clusters,how='left',on=col_cluster)
          base_clusters['percent'] = base_clusters['count_x'].values/base_clusters['count_y'].values

          # Calculamos um score para que não seja selecionado o cluster com 1 abertura que represente
          # 100% da abertura ruim:
          base_clusters['log'] = np.log2(base_clusters['count_y']) # log da quantidade total de aberturas de cada cluster
          base_clusters['score'] = base_clusters['percent'].values*base_clusters['log'].values


          cluster_mais_parecido = base_clusters.loc[base_clusters['score'] == base_clusters['score'].max(),[col_cluster]].values[0][0]
          if classifica_outlier == 0:
            media_do_cluster = pd.DataFrame(np.array([df_aberturas_ruins.iloc[l,:].values]),
                                            columns = df_aberturas_ruins.columns.values,
                                            index = [1])
          else:
            media_do_cluster = df_base_clusters.loc[df_base_clusters[col_cluster] == cluster_mais_parecido,aberturas+etapas+[col_cluster]]
            media_do_cluster = media_do_cluster.groupby([col_cluster],as_index=False)[etapas].mean()
            media_do_cluster[aberturas] = list(abertura)

          media_do_cluster[col_cluster] = cluster_mais_parecido
          base_media_abertura_ruim = media_do_cluster
          base_media_abertura_ruim['outlier'] = classifica_outlier

          df_clusters_final = pd.concat([df_clusters_final,base_media_abertura_ruim])

        else:
          # Caso nenhuma combinação de aberturas da abertura ruim esteja presente na base realizada:
          # Segunda tentativa será encontrando somente 1 abertura de cada vez:
          #-------------------------------------------------------------------------------------------

          base_clusters = pd.DataFrame()
          abertura = df_aberturas_ruins[aberturas].iloc[l,:].values
          # Para cada combinação de n chaves das aberturas, onde n é o número de aberturas -1, vamos
          # identificar quais são os clusteres que mais possuem aberturas com essa combinação:
          for i in range(len(aberturas)):
            comb_aberturas = aberturas[i]
            comb_chaves = abertura[i]

            df_merged_valores_originais_filtrado = df_base_clusters.copy()

            df_merged_valores_originais_filtrado = df_merged_valores_originais_filtrado.loc[df_merged_valores_originais_filtrado[comb_aberturas] == comb_chaves]

            if len(df_merged_valores_originais_filtrado) > 0:
              df_merged_valores_originais_filtrado['count'] = 1
              df_merged_valores_originais_filtrado = df_merged_valores_originais_filtrado.groupby([col_cluster],as_index=False)['count'].sum()
              base_clusters = pd.concat([base_clusters,df_merged_valores_originais_filtrado])


          # Se foi possível encontrar clusters com as combinações de aberturas:
          if len(base_clusters) > 0:
            # Calculamos quantos % da quantidade de aberturas de cada cluster as combinações
            # de chaves da abertura ruim representam:
            base_clusters = base_clusters.groupby([col_cluster],as_index=False)['count'].sum()
            base_clusters = pd.merge(base_clusters,base_qtd_aberturas_clusters,how='left',on=col_cluster)
            base_clusters['percent'] = base_clusters['count_x'].values/base_clusters['count_y'].values

            # Calculamos um score para que não seja selecionado o cluster com 1 abertura que represente
            # 100% da abertura ruim:
            base_clusters['log'] = np.log2(base_clusters['count_y']) # log da quantidade total de aberturas de cada cluster
            base_clusters['score'] = base_clusters['percent'].values*base_clusters['log'].values


            cluster_mais_parecido = base_clusters.loc[base_clusters['score'] == base_clusters['score'].max(),[col_cluster]].values[0][0]
            if classifica_outlier == 0:
              media_do_cluster = pd.DataFrame(np.array([df_aberturas_ruins.iloc[l,:].values]),
                                              columns = df_aberturas_ruins.columns.values,
                                              index = [1])
            else:
              media_do_cluster = df_base_clusters.loc[df_base_clusters[col_cluster] == cluster_mais_parecido,aberturas+etapas+[col_cluster]]
              media_do_cluster = media_do_cluster.groupby([col_cluster],as_index=False)[etapas].mean()
              media_do_cluster[aberturas] = list(abertura)

            media_do_cluster[col_cluster] = cluster_mais_parecido
            base_media_abertura_ruim = media_do_cluster
            base_media_abertura_ruim['outlier'] = classifica_outlier

            df_clusters_final = pd.concat([df_clusters_final,base_media_abertura_ruim])

          else:
            cluster_mais_parecido = df_base_clusters[col_cluster].unique()
            if len(cluster_mais_parecido) == 0:
              cluster_mais_parecido = ''
            else:
              cluster_mais_parecido = cluster_mais_parecido[0]
            media_do_cluster = df_base_clusters.loc[df_base_clusters[col_cluster] == cluster_mais_parecido,aberturas+etapas+[col_cluster]]
            media_do_cluster = media_do_cluster.groupby([col_cluster],as_index=False)[etapas].mean()
            base_media_abertura_ruim = media_do_cluster
            base_media_abertura_ruim['outlier'] = -3

            df_clusters_final = pd.concat([df_clusters_final,base_media_abertura_ruim])
            print("Erro: nenhuma abertura está no histórico: "+str(abertura))



      else:
        cluster_mais_parecido = df_base_clusters[col_cluster].unique()
        if len(cluster_mais_parecido) == 0:
          cluster_mais_parecido = ''
        else:
          cluster_mais_parecido = cluster_mais_parecido[0]
        media_do_cluster = df_base_clusters.loc[df_base_clusters[col_cluster] == cluster_mais_parecido,aberturas+etapas+[col_cluster]]
        media_do_cluster = media_do_cluster.groupby([col_cluster],as_index=False)[etapas].mean()
        base_media_abertura_ruim = media_do_cluster
        base_media_abertura_ruim['outlier'] = -4

        df_clusters_final = pd.concat([df_clusters_final,base_media_abertura_ruim])
        print("Erro: nenhuma abertura está no histórico: "+str(abertura))

  return df_clusters_final

#_______________________________________________________________________________________________

#@title Def escolhe_modelo_de_encaixe_de_outliers

'''
Serve para determinar qual algoritmo de encaixe performa melhor.
Selecionamos uma amostra aletaória de aberturas de uma base
clusterizada sem outliers, supomos que estas sejam outliers e tentamos encaixar
de volta nos clusters com os algoritmos de encaixe. É escolhido aquele que
melhor aproximar da configuração inicial.
'''

def escolhe_modelo_de_encaixe_de_outliers(df_base_clusters,
                                          df_base_chaves_classificadas,
                                          col_cluster,
                                          col_valores,
                                          aberturas):
  # Iniciamos os valores:
  taxa_eficiencia_frequencia_abs_final = 0
  taxa_eficiencia_relevancia_abs_final = 0
  taxa_eficiencia_frequencia_valor_final = 0
  taxa_eficiencia_relevancia_valor_final = 0
  col_valores_teste = [x+'_teste' for x in col_valores]
  col_valores_erro = [x+'_erro' for x in col_valores]

  # Selecionamos uma amostra aleatória de 20% das aberturas de cada cluster,
  # sendo selecionado no mínimo 1 abertura de cada cluster que possui mais de 1 abertura.
  #-------------------------------------------------------------------------------------------------

  # Selecionamos a base sem outliers
  df_base_clusters['outlier'] = df_base_clusters['outlier'].astype(int)
  df_base_clusters[col_valores] = df_base_clusters[col_valores].astype(float)
  df_base_clusters_aux = df_base_clusters.loc[df_base_clusters['outlier'] == 0]

  # Selecionamos apenas os clusters que possuem mais de 1 abertura
  base_qtd_aberturas_clusters = df_base_clusters_aux.copy()
  base_qtd_aberturas_clusters['count'] = 1
  base_qtd_aberturas_clusters = base_qtd_aberturas_clusters.groupby([col_cluster],as_index=False)['count'].sum()
  df_base_clusters_aux = pd.merge(df_base_clusters_aux,base_qtd_aberturas_clusters,how='left',on=col_cluster)

  df_base_clusters_aux = df_base_clusters_aux.loc[df_base_clusters_aux['count'] > 1]

  # Para cada cluster, selecionamos 10% das aberturas
  df_base_clusters_amostra = pd.DataFrame()
  clusters = df_base_clusters_aux[col_cluster].unique()

  # Repetimos o processo 10 vezes:
  for i in range(10):
    lista_clusters_amostra = [df_base_clusters_amostra]
    for cluster in clusters:
      df_base_cluster = df_base_clusters_aux.loc[df_base_clusters_aux[col_cluster] == cluster]
      samples = int(np.round(0.1 * len(df_base_cluster),0))
      if samples == 0:
        samples = 1
      amostra = df_base_cluster.sample(n=samples)
      lista_clusters_amostra = lista_clusters_amostra+[amostra.copy()]
    df_base_clusters_amostra = pd.concat(lista_clusters_amostra)


    # Removemos a amostra da base original
    df_base_clusters_sem_amostra = df_base_clusters_aux.merge(df_base_clusters_amostra, how='left', indicator=True)
    df_base_clusters_sem_amostra = df_base_clusters_sem_amostra[df_base_clusters_sem_amostra['_merge'] == 'left_only']
    df_base_clusters_sem_amostra = df_base_clusters_sem_amostra.drop(columns='_merge')


    # Agora vamos performar o encaixe da amostra dentro da base sem a amostra e medir a precisão
    # de cada algoritmo de encaixe:
    #-------------------------------------------------------------------------------------------------
    #print(df_base_clusters_amostra)
    # Encaixamos na base sem a amostra utilizando o algoritmo por frequencia:
    df_base_clusters_teste_frequencia = encaixa_abertura_no_cluster_por_frequencia(df_aberturas_ruins = df_base_clusters_amostra[aberturas],
                                                                                  df_base_clusters = df_base_clusters_sem_amostra,
                                                                                  aberturas = aberturas,
                                                                                  classifica_outlier = 1,
                                                                                  col_valores = col_valores,
                                                                                  col_cluster = col_cluster)

    # Encaixamos na base sem a amostra utilizando o algoritmo por relevancia
    df_base_clusters_teste_relevancia = encaixa_abertura_no_cluster_por_relevancia(df_aberturas_ruins = df_base_clusters_amostra[aberturas],
                                                                                  df_base_clusters = df_base_clusters_sem_amostra,
                                                                                  df_base_chaves_classificadas = df_base_chaves_classificadas,
                                                                                  classifica_outlier = 1,
                                                                                  col_cluster = col_cluster,
                                                                                  col_valores = col_valores,
                                                                                  aberturas = aberturas)

    # Vamos medir a eficiência do algoritmo de frequencia:
    amostra_realocada = df_base_clusters_teste_frequencia.loc[df_base_clusters_teste_frequencia['outlier'] == 1][aberturas+[col_cluster]+col_valores+['outlier']]
    amostra_realocada = amostra_realocada.rename(columns=dict(zip(col_valores, col_valores_teste)))
    #print("-----------------------------------")
    #print(amostra_realocada)
    teste_merge_erro_abs = pd.merge(df_base_clusters_amostra[aberturas+[col_cluster]],amostra_realocada[aberturas+[col_cluster,'outlier']],how='left',on=aberturas+[col_cluster])
    teste_merge_erro_abs = teste_merge_erro_abs.fillna(0)
    taxa_eficiencia_frequencia_erro_abs = teste_merge_erro_abs['outlier'].sum()/len(teste_merge_erro_abs)

    teste_merge_erro_valores = pd.merge(df_base_clusters_amostra[aberturas+col_valores],amostra_realocada[aberturas+col_valores_teste],how='left',on=aberturas)
    teste_merge_erro_valores[col_valores_erro] = abs(teste_merge_erro_valores[col_valores_teste].values/teste_merge_erro_valores[col_valores].values - 1)
    taxa_eficiencia_frequencia_erro_valor = teste_merge_erro_valores[col_valores_erro].mean(axis=0).mean()


    # Vamos medir a eficiência do algoritmo de relevancia:
    amostra_realocada = df_base_clusters_teste_relevancia.loc[df_base_clusters_teste_relevancia['outlier'] == 1][aberturas+[col_cluster]+col_valores+['outlier']]
    amostra_realocada = amostra_realocada.rename(columns=dict(zip(col_valores, col_valores_teste)))
    #print("-----------------------------------")
    #print(amostra_realocada)
    teste_merge_erro_abs = pd.merge(df_base_clusters_amostra[aberturas+[col_cluster]],amostra_realocada[aberturas+[col_cluster,'outlier']],how='left',on=aberturas+[col_cluster])
    teste_merge_erro_abs = teste_merge_erro_abs.fillna(0)
    taxa_eficiencia_relevancia_erro_abs = teste_merge_erro_abs['outlier'].sum()/len(teste_merge_erro_abs)

    teste_merge_erro_valores = pd.merge(df_base_clusters_amostra[aberturas+col_valores],amostra_realocada[aberturas+col_valores_teste],how='left',on=aberturas)
    teste_merge_erro_valores[col_valores_erro] = abs(teste_merge_erro_valores[col_valores_teste].values/teste_merge_erro_valores[col_valores].values - 1)
    taxa_eficiencia_relevancia_erro_valor = teste_merge_erro_valores[col_valores_erro].mean(axis=0).mean()



    taxa_eficiencia_frequencia_abs_final = taxa_eficiencia_frequencia_abs_final+taxa_eficiencia_frequencia_erro_abs
    taxa_eficiencia_relevancia_abs_final = taxa_eficiencia_relevancia_abs_final+taxa_eficiencia_relevancia_erro_abs

    taxa_eficiencia_frequencia_valor_final = taxa_eficiencia_frequencia_valor_final+taxa_eficiencia_frequencia_erro_valor
    taxa_eficiencia_relevancia_valor_final = taxa_eficiencia_relevancia_valor_final+taxa_eficiencia_relevancia_erro_valor

    #print([taxa_eficiencia_frequencia,taxa_eficiencia_relevancia])
  # Retornamos a eficiência estimada do algoritmo de freqência e do de relevância:
  return [taxa_eficiencia_frequencia_abs_final/10,taxa_eficiencia_relevancia_abs_final/10,taxa_eficiencia_frequencia_valor_final/10,taxa_eficiencia_relevancia_valor_final/10]

#___________________________________________________________________________________________________________________

#@title Def classifica_chaves_significativas
from itertools import combinations
"""
Classificamos cada abertura com base no número de clusters que ela aparece
e também na distância média entre os centros dos clusters onde ela aparece.
"""

def classifica_chaves_significativas(base_clusterizada,
                                    aberturas,
                                    coordenadas,
                                    col_cluster):

  base_clusterizada[coordenadas] = base_clusterizada[coordenadas].astype('float')

  # Calculamos os centros dos clusters
  centros = base_clusterizada.groupby(col_cluster, as_index=False)[coordenadas].mean()

  # Calculamos a matriz de distâncias entre os clusters
  matriz_distancias = pd.DataFrame(squareform(pdist(centros.iloc[:, 1:])), columns=centros[col_cluster].unique(), index=centros[col_cluster].unique())

  # Iniciamos o DatafRame final
  chaves_classificadas = pd.DataFrame()

  # Para cada abertura, vamos classificar as chaves:
  for i in range(len(aberturas)):

    # Criamos uma base auxiliar somente com as chaves da abertura e seus clusters
    base_clusterizada['aux'] = 1
    aux = base_clusterizada.groupby([aberturas[i],col_cluster], as_index=False)[['aux']].sum()
    base_clusterizada = base_clusterizada.drop(columns='aux')
    aux = aux[[aberturas[i],col_cluster]]

    # Calculamos a quantidade de clusters nos quais cada chave da abertura aparece
    freq = base_clusterizada.groupby(aberturas[i], as_index=False)[[col_cluster]].nunique()
    freq['dist_media'] = 0

    # Para cada chave, vamos calcular a distância média dos centros dos clusters
    chaves_unicas = aux[aberturas[i]].unique()

    for chave in chaves_unicas:
      dist = []
      clusters = aux.loc[aux[aberturas[i]] == chave][col_cluster].unique()
      combinations = list(itertools.combinations(clusters, 2))
      for comb in combinations:
        dist = dist + [matriz_distancias.loc[matriz_distancias.index == comb[0],[comb[1]]].values[0][0]]

      if len(dist) > 0:
        freq.loc[freq[aberturas[i]] == chave,['dist_media']] = np.average(dist)
      else:
        freq.loc[freq[aberturas[i]] == chave,['dist_media']] = 0

    # Calculamos o score de relevância multiplicando a quantidade de clusters pela distância média
    # quanto menor o score, mais relevante é a chave.
    freq['score'] = freq['dist_media']*freq[col_cluster]

    freq['abertura'] = aberturas[i]

    freq = freq.rename(columns={aberturas[i]:'chaves'})

    # Empilhamos as chaves para ter uma base única
    if len(chaves_classificadas) == 0:
      chaves_classificadas = freq
    else:
      chaves_classificadas = pd.concat([chaves_classificadas,freq])


  return chaves_classificadas
