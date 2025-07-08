#@title Def clusterizacao_aberturas
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from colored import *
from encaixa_abertura_no_cluster import *
from checks import check_etapas_do_funil

def clusterizacao_aberturas(df_completo,  # data_frame formatado completo
                            df_classificacao_aberturas_ruins,
                               ultima_data_hist,
                               qtd_semanas_media,
                               col_data,
                               topos,
                               aberturas,
                               abertura_principal,
                              max_origin,
                            eps_parameter,
                              flag_clusterizacao):

    if flag_clusterizacao:

      etapas = df_completo['Etapa'].unique()
      etapas = [x for x in etapas if '2' in x]
      topo = [x for x in etapas if x.split('2')[0] in topos]

      # Valos filtrar a base completa somente nas datas de interesse:
      df_filtrado = df_completo.copy()
      df_filtrado_maior = df_filtrado.loc[(df_filtrado[col_data] <= ultima_data_hist-pd.Timedelta(max_origin*7, unit='D')) & (df_filtrado[col_data] >= ultima_data_hist-pd.Timedelta((max_origin+qtd_semanas_media)*7*3, unit='D'))]
      df_filtrado = df_filtrado.loc[(df_filtrado[col_data] <= ultima_data_hist-pd.Timedelta(max_origin*7, unit='D')) & (df_filtrado[col_data] >= ultima_data_hist-pd.Timedelta((max_origin+qtd_semanas_media)*7, unit='D'))]
      df_filtrado_original = df_filtrado.copy()
      if len(df_filtrado[col_data].unique()) < 2:
        print(colored("Não existem pontos de histórico suficientes para realizar clusterização.","red"))
        df_clusters_final = pd.DataFrame()
        df_clusters_final_formatado = pd.DataFrame()
      else:

        # Vamos gerar uma base sem as aberturas ruins para realizar a clusterização:
        df_filtrado = pd.merge(df_filtrado,df_classificacao_aberturas_ruins,how='left',on=aberturas)
        df_filtrado['outlier'] = df_filtrado['outlier'].fillna(1)
        df_filtrado = df_filtrado.loc[df_filtrado['outlier'] == 0]



        # Vamos criar uma base com a media das cohorts abertas de todas as etapas e todas as aberturas
        #df_media = df_filtrado.groupby(aberturas+['Etapa'],as_index=False)['%__Volume Aberta'].mean()
        # vamos melhorar a média fazendo a cohort aberta média ao invés da média das cohorts abertas:
        df_media = df_filtrado.groupby(aberturas+['Etapa'],as_index=False)['Volume Aberta','Volume'].sum()
        df_media['%__Volume Aberta'] = df_media['Volume Aberta'].values / df_media['Volume'].values
        df_media = df_media.drop(columns=['Volume Aberta','Volume'])
        df_media = pd.pivot_table(df_media, values='%__Volume Aberta', index=aberturas, columns='Etapa').reset_index(inplace=False)
        df_media = df_media[aberturas+etapas]

        # Vamos criar uma base com o volume máximo do primeiro ToF da base de todas as aberturas
        df_vol = df_filtrado.loc[df_filtrado['Etapa'].isin(topo)]
        df_vol = pd.pivot_table(df_vol, values='Volume', index=aberturas, columns='Etapa').reset_index(inplace=False)
        df_vol = df_vol.groupby(aberturas,as_index=False)[topo].max()
        df_vol = df_vol.rename(columns=dict(zip(topo, topos)))

        df_vol_original = df_filtrado_original.loc[df_filtrado_original['Etapa'].isin(topo)]
        df_vol_original = pd.pivot_table(df_vol_original, values='Volume', index=aberturas, columns='Etapa').reset_index(inplace=False)
        df_vol_original = df_vol_original.groupby(aberturas,as_index=False)[topo].max()
        df_vol_original = df_vol_original.rename(columns=dict(zip(topo, topos)))


        # Vamos unir as bases
        df_merged = pd.merge(df_media,df_vol,how='left',on=aberturas)

        #-----------------------------------------------------------------------------------------
        # Caso a abertura principal indique "Contornar via coincident de maior período", não faremos a clusterização.
        # Ao invés disso, somente calculamos a média coincident das aberturas ruins num período 3 vezes maior
        # do que o indicado pela qdt de semanas na média e atribuímos este valor à cohort aberta.
        if abertura_principal == "Contornar via coincident de maior período":
          # vamos calcular a conversão coincident do período maior:

          # Definindo e ordenando corretamente as etapas do funi:
          etapas_conversao,etapas_volume,etapas_extra,mensagem_local,erro_local = check_etapas_do_funil(lista_etapas_conversao = etapas, # lista com todas as etapas de conversão definidas pelo usuário no painel de controle
                                                                                                        lista_topos_de_funil = topos,  # lista com os ToF's definida pelo usuário no painel de controle
                                                                                                        df_on_top_ratio = pd.DataFrame(),
                                                                                                        racional_on_top_ratio = '',
                                                                                                        flag_gerar_on_top_ratio = False,
                                                                                                        nome_do_arquivo = 'clusterizacao_aberturas.py')
          etapas_shifted = etapas_conversao[1:]+[etapas_volume[-1]]
          
          df_media = df_filtrado_maior.groupby(aberturas+['Etapa'],as_index=False)['Volume'].sum()
          df_media = pd.pivot_table(df_media, values='Volume', index=aberturas, columns='Etapa').reset_index(inplace=False)
          df_media[etapas_conversao] = df_media[etapas_shifted].values / df_media[etapas_conversao].values
          df_media['clusters'] = '-1'
          df_media = pd.merge(df_media,df_classificacao_aberturas_ruins,how='left',on=aberturas)
          df_media['outlier'] = df_media['outlier'].fillna(1)
          df_clusters_final = df_media

        
        # Caso contrário, realizamos a clusterização:
        #------------------------------------------------------------------------------------------
        else:
          
          # Vamos criar colunas com índices numéricos para cada chave de abertura. Assim, aberturas
          # com chaves semelhantes podem ser clusterizadas juntas:
          df_merged = df_merged.sort_values(by=etapas)
          index_chaves = []
          for abertura in aberturas:
            chaves = list(df_merged[abertura].unique())
            indices = list(range(len(chaves)))
            dic = dict(zip(chaves,indices))
            df_merged['index_'+abertura] = df_merged[abertura].values
            df_merged = df_merged.replace({'index_'+abertura: dic})
            index_chaves = index_chaves+['index_'+abertura]
  
  
          # Vamos normalizar todos os valores para facilitar a clusterização:
          df_merged_valores_originais = df_merged.copy()
          for e in etapas+topos+index_chaves:
            df_merged[e]=(df_merged[e]-df_merged[e].min())/(df_merged[e].max()-df_merged[e].min())
            df_merged[e] = df_merged[e].fillna(0)
  
  
          # Vamos clusterizar as aberturas via DBSCAN primariamente somente com basenas médias das cohorts
          train = df_merged[etapas]
          train = train.astype(float).to_numpy(dtype=float)
  
          try:
            clustering = DBSCAN(eps=eps_parameter, min_samples=0).fit(train) # eps=0.11, min_samples=0
          except:
            clustering = DBSCAN(eps=eps_parameter, min_samples=1).fit(train) # eps=0.11, min_samples=0
  
          cluster_labels = clustering.labels_
  
          n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
  
          df_merged['clusters_primarios'] = cluster_labels
          df_merged['clusters_secundarios'] = cluster_labels
          df_merged['clusters_terciarios'] = cluster_labels
  
          df_merged_final = pd.DataFrame()
  
  
          # Em seguida, para cada cluster primario, vamos subclusterizar com base na semelhança
          # das chaves das aberturas e das cohorts abertas, caso o cluster primario tenha mais do que 10 aberturas:
          clusters = list(dict.fromkeys(list(cluster_labels))) # remover duplicados
          for cluster in clusters:
            df_merged_secundario_final = pd.DataFrame()
  
            df_merged_secundario = df_merged.loc[df_merged['clusters_primarios'] == cluster]
  
            train = df_merged_secundario[etapas]
            if len(train) >= 10:
  
              train = train.astype(float).to_numpy(dtype=float)
  
              try:
                clustering = DBSCAN(eps=eps_parameter/2, min_samples=0).fit(train) # eps=0.11, min_samples=0
              except:
                clustering = DBSCAN(eps=eps_parameter/2, min_samples=1).fit(train) # eps=0.11, min_samples=0
  
              cluster_labels = clustering.labels_.astype(int)
  
  
              n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
  
              if n_clusters_ < (len(train)-2):
                df_merged_secundario['clusters_secundarios'] = cluster_labels
              else:
                df_merged_secundario['clusters_secundarios'] = ''
            else:
              df_merged_secundario['clusters_secundarios'] = ''
  
            # Vamos realocar os clusters secundários que ficaram com somente uma abertura dentro do cluster
            # primario:
            base_qtd_aberturas_clusters = df_merged_secundario.copy()
            base_qtd_aberturas_clusters['count'] = 1
            base_qtd_aberturas_clusters = base_qtd_aberturas_clusters.groupby(['clusters_secundarios'],as_index=False)['count'].sum()
  
            df_merged_secundario = pd.merge(df_merged_secundario,base_qtd_aberturas_clusters,how='left',on='clusters_secundarios')
  
            df_aberturas_ruins = df_merged_secundario.loc[df_merged_secundario['count'] <= 1]
            df_merged_secundario_filtrado = df_merged_secundario.loc[df_merged_secundario['count'] > 1]
  
            # Com base na clusterização até o momento, gerar uma classificação das chaves que são
            # mais relevantes para a clusterização para ajudar a realocar as aberturas outliers:
            '''
            df_base_chaves_classificadas = classifica_chaves_significativas(base_clusterizada = df_merged,
                                                                            aberturas = aberturas,
                                                                            coordenadas = etapas,
                                                                            col_cluster = 'clusters_primarios')
            '''
  
            if len(df_merged_secundario_filtrado) > 0:
              df_merged_secundario = encaixa_abertura_no_cluster_por_frequencia(df_aberturas_ruins = df_aberturas_ruins,
                                                                                df_base_clusters = df_merged_secundario_filtrado,
                                                                                aberturas = aberturas,
                                                                                abertura_principal = '',
                                                                                classifica_outlier = 0,
                                                                                col_valores = etapas+index_chaves,
                                                                                col_cluster = 'clusters_secundarios')
  
  
  
            # Se houver uma abertura principal, vamos subdividir os clusters
            if abertura_principal != "":
  
              clusters_secundarios = list(df_merged_secundario['clusters_secundarios'].unique())
              lista_merged_secundario_final = [df_merged_secundario_final]
              for cluster_secundario in clusters_secundarios:
                df_merged_terciario = df_merged_secundario.loc[df_merged_secundario['clusters_secundarios'] == cluster_secundario]
  
                aberturas_principais = list(df_merged_terciario[abertura_principal].unique())
                if len(aberturas_principais) == 0:
                  df_merged_terciario['clusters_terciarios'] = ''
                else:
                  indices = list(range(len(aberturas_principais)))
                  dic = dict(zip(aberturas_principais,indices))
                  df_merged_terciario['clusters_terciarios'] = df_merged_terciario[abertura_principal].values
                  df_merged_terciario = df_merged_terciario.replace({'clusters_terciarios': dic})
  
                df_merged_terciario['clusters_secundarios'] = cluster_secundario
  
                df_merged_terciario['outlier'] = 0
  
                lista_merged_secundario_final = lista_merged_secundario_final+[df_merged_terciario.copy()]
  
              df_merged_secundario_final = pd.concat(lista_merged_secundario_final)
  
            else:

              df_merged_secundario_final = df_merged_secundario
              #df_merged_secundario_final['clusters_terciarios'] = df_merged_secundario_final['clusters_terciarios'].astype(int)
              df_merged_secundario_final['clusters_terciarios'] = ''
  
  
  
  
  
  
            df_merged_secundario_final['clusters_primarios'] = cluster
  
            df_merged_secundario_final['outlier'] = 0
  
            df_merged_final = pd.concat([df_merged_final,df_merged_secundario_final])
  
  
  
          df_merged_final['clusters'] = df_merged_final[['clusters_primarios','clusters_secundarios','clusters_terciarios']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
  
          df_merged_valores_originais = pd.merge(df_merged_valores_originais,df_merged_final[aberturas+['clusters']],how='left',on=aberturas)
  
          df_merged_valores_originais = df_merged_valores_originais[aberturas+etapas+['clusters']]
  
  
          # Classificação das aberturas ruins dentro dos clusters
          #-----------------------------------------------------------------------------------------------
          df_clusters_final = df_merged_valores_originais.copy()
  
          # Vamos incluir a informação de abertura ruim na clusterização:
          # Inicialmente a clusterização só considerou aberturas boas:
          df_clusters_final['outlier'] = 0
          df_clusters_final.loc[df_clusters_final['clusters'].isna(),['outlier']] = 1
          df_clusters_final.loc[df_clusters_final['clusters'].isna(),['clusters']] = -1
  
  
          # aberturas ruins:
          df_aberturas_ruins_classificadas = df_classificacao_aberturas_ruins.loc[df_classificacao_aberturas_ruins['outlier'] != 0]
          df_aberturas_ruins_clusters = df_clusters_final.loc[df_clusters_final['outlier'] != 0]
          df_aberturas_ruins = pd.concat([df_aberturas_ruins_classificadas,df_aberturas_ruins_clusters])
  
  
  
          df_clusters_final = df_clusters_final.loc[df_clusters_final['outlier'] == 0]
  
  
          # Com base na clusterização até o momento, gerar uma classificação das chaves que são
          # mais relevantes para a clusterização para ajudar a realocar as aberturas outliers:
          '''
          df_base_chaves_classificadas = classifica_chaves_significativas(base_clusterizada = df_clusters_final,
                                                                          aberturas = aberturas,
                                                                          coordenadas = etapas,
                                                                          col_cluster = 'clusters')
          '''
  
          df_clusters_final = encaixa_abertura_no_cluster_por_frequencia(df_aberturas_ruins = df_aberturas_ruins,
                                                                        df_base_clusters = df_clusters_final,
                                                                        aberturas = aberturas,
                                                                        abertura_principal = abertura_principal,
                                                                        classifica_outlier = 1,
                                                                        col_valores = etapas,
                                                                        col_cluster = 'clusters')

        #--------------------------------------------------------------------------------------------------------------

        
        # Precisamos gerar uma base final com a média dos shares das fechadas e da cohort de ajuste em cada cluster para serem
        # usados no forecast:

        df_share_fechada = df_filtrado_original.copy()
        colunas = df_share_fechada.columns.values
        col_shares = [x for x in colunas if "s__" in x]
        df_share_fechada = df_share_fechada.groupby(aberturas+['Etapa'],as_index=False)[col_shares+['%__Coincident']].mean()
        #df_share_fechada = pd.merge(df_share_fechada,df_clusters_final.loc[df_clusters_final['outlier'] == 0,aberturas+['clusters']],how='left',on=aberturas)
        df_share_fechada = pd.merge(df_share_fechada,df_clusters_final[aberturas+['clusters','outlier']],how='left',on=aberturas)
        # Calculando a média dos shares de Coincident dos clusters sem outliers.
        df_share_fechada_mean = df_share_fechada.loc[df_share_fechada['outlier'] == 0].groupby(['clusters','Etapa'],as_index=False)[col_shares+['%__Coincident']].mean()
        # Substituindo os outliers pelos valores médios dos clusters
        col_shares_x = [x+"_x" for x in col_shares]
        col_shares_y = [x+"_y" for x in col_shares]

        df_share_fechada = pd.merge(df_share_fechada,df_share_fechada_mean,how='left',on=['clusters','Etapa'])
        df_share_fechada.loc[df_share_fechada['outlier'] == 1,col_shares_x+['%__Coincident_x']] = df_share_fechada.loc[df_share_fechada['outlier'] == 1,col_shares_y+['%__Coincident_y']].values
        df_share_fechada = df_share_fechada.rename(columns=dict(zip(col_shares_x+['%__Coincident_x'], col_shares+['%__Coincident'])))
        df_share_fechada = df_share_fechada.drop(columns=col_shares_y+['%__Coincident_y'])

        df_clusters_final_formatado = df_clusters_final.copy()
        df_clusters_final_formatado = df_clusters_final_formatado.melt(id_vars=aberturas+['clusters','outlier'],var_name='Etapa', value_name='Cohort Aberta Média')
        df_clusters_final_formatado = pd.merge(df_clusters_final_formatado,df_share_fechada[aberturas+['Etapa','clusters']+col_shares+['%__Coincident']],how='left',on=aberturas+['Etapa','clusters'])


        df_aberturas = df_filtrado_original.copy()
        df_aberturas['aux'] = 1
        df_aberturas = df_aberturas.groupby(aberturas,as_index=False)['aux'].sum()
        df_aberturas['aux'] = 1
        #df_clusters_final = df_clusters_final.drop(columns='aux')
        df_clusters_final = pd.merge(df_aberturas,df_clusters_final,how='left',on=aberturas)
        media_df_clusters_final = df_clusters_final.loc[~df_clusters_final['clusters'].isna()]
        media_df_clusters_final = media_df_clusters_final.groupby(['aux'],as_index=False)[etapas].mean()
        df_clusters_final.loc[df_clusters_final['clusters'].isna(),etapas] = media_df_clusters_final[etapas].values
        df_clusters_final.loc[df_clusters_final['clusters'].isna(),['clusters','outlier']] = [-1,1]
        
                                
    else:
      df_clusters_final = pd.DataFrame()
      df_clusters_final_formatado = pd.DataFrame()



    return df_clusters_final,df_clusters_final_formatado
