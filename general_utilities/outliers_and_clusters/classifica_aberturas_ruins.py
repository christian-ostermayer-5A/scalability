#@title Def classifica_aberturas_ruins
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from colored import *

def classifica_aberturas_ruins(df_completo,  # data_frame formatado completo
                               ultima_data_hist,
                               qtd_semanas_media,
                               outlier_treshold,
                               col_data,
                               max_origin,
                               topos,
                               aberturas,
                               flag_clusterizacao):
     
    if flag_clusterizacao:

      etapas = df_completo['Etapa'].unique()
      etapas = [x for x in etapas if '2' in x]
      etapas_sum = [x+"_sum" for x in etapas]
      topo = [x for x in etapas if x.split('2')[0] == topos[0]]

      # definimos a última etapa do funil:
      primeiras_etapas_vol = [x.split('2')[0] for x in etapas]
      ultimas_etapas_vol = [x.split('2')[1] for x in etapas]
      ultima_etapa_vol = [x for x in ultimas_etapas_vol if x not in primeiras_etapas_vol][0]
      ultima_etapa = [x for x in etapas if x.split('2')[1] == ultima_etapa_vol][0]

      # Valos filtrar a base completa somente nas datas de interesse:
      df_filtrado = df_completo.copy()
      df_filtrado = df_filtrado.loc[(df_filtrado[col_data] <= ultima_data_hist-pd.Timedelta(max_origin*7, unit='D')) & (df_filtrado[col_data] >= ultima_data_hist-pd.Timedelta((max_origin+qtd_semanas_media)*7, unit='D'))]

      if len(df_filtrado[col_data].unique()) < 2:
        print(colored("Não existem pontos de histórico suficientes para realizar classificação de aberturas ruins.","red"))
        df_merged = pd.DataFrame()
      else:
            
        #print(df_filtrado.loc[(df_filtrado['city_group'] == 'Mogi das Cruzes') & (df_filtrado['lead'] == 'IS') & (df_filtrado['mkt_channel'] == 'Indica Aí - Agents'),['week_start','Etapa','%__Volume Aberta']])
        # Vamos criar uma base com o desvio padrão da cohort aberta de todas as etapas e todas as aberturas

        # Não queremos poluir o std das cohorts com cohorts zeradas por conta de não ter entrado nenhum volume
        df_std = df_filtrado.loc[df_filtrado['Volume'] > 0]
        df_std = df_filtrado.groupby(aberturas+['Etapa'],as_index=False)['%__Volume Aberta'].std()
        df_std = df_std.rename(columns={'%__Volume Aberta':'%__Volume Aberta_std'})

        # Porém as aberturas ruins são justamente aquelas que possuem muitas semanas zeradas, reduzindo o std, apesar de saltos altos de conversão.
        # Vamos recalcular o desvio padrão considerando apenas os valores únicos de cada abertura nas semanas selecionadas.
        #df_max = df_filtrado.groupby(aberturas+['Etapa'],as_index=False)['%__Volume Aberta'].max()
        #df_max = df_max.rename(columns={'%__Volume Aberta':'%__Volume Aberta_max'})
        #df_min = df_filtrado.groupby(aberturas+['Etapa'],as_index=False)['%__Volume Aberta'].min()
        #df_min = df_min.rename(columns={'%__Volume Aberta':'%__Volume Aberta_min'})
        #df_std = pd.merge(df_std,df_max,how='left',on=aberturas+['Etapa'])
        #df_std = pd.merge(df_std,df_min,how='left',on=aberturas+['Etapa'])
        df_std['%__Volume Aberta'] = df_std['%__Volume Aberta_std'].values# * (df_std['%__Volume Aberta_max'].values - df_std['%__Volume Aberta_min'].values)
        #df_std['%__Volume Aberta'] = df_std['%__Volume Aberta_max'].values - df_std['%__Volume Aberta_min'].values

        df_std = pd.pivot_table(df_std, values='%__Volume Aberta', index=aberturas, columns='Etapa').reset_index(inplace=False)

        df_std = df_std[aberturas+etapas]

        # Vamos criar uma base com a soma das cohorts para classificar mais tarde como ruins aquelas
        # aberturas que não possuem nenhum valor
        df_sum = df_filtrado.groupby(aberturas+['Etapa'],as_index=False)['%__Volume Aberta'].sum()
        df_sum = pd.pivot_table(df_sum, values='%__Volume Aberta', index=aberturas, columns='Etapa').reset_index(inplace=False)
        df_sum = df_sum[aberturas+etapas]
        df_sum = df_sum.rename(columns=dict(zip(etapas,etapas_sum)))

        # Vamos criar uma base com o volume máximo do primeiro ToF da base de todas as aberturas
        df_vol = df_filtrado.loc[df_filtrado['Etapa'] == topo[0]]
        df_vol = df_vol.groupby(aberturas,as_index=False)['Volume'].std()

        # Vamos unir as bases
        df_merged = pd.merge(df_sum,df_std,how='left',on=aberturas)
        df_merged = pd.merge(df_merged,df_vol,how='left',on=aberturas)
        df_merged[etapas] = df_merged[etapas].fillna(0)


        #_______________________________________________________________________________________________

        # Identificando os outliers através do algoritmo de clusterização GaussianMixture

        # Vamos criar um DataFRame contendo qual deveria ser o valor máximo de std de uma conversão de
        # cada etapa para que a abertura seja classificada como sendo ruim:
        df_limite_std = pd.DataFrame([list(range(len(etapas)+1))],
                                    index=[1],
                                    columns=etapas+['Volume'])

        for e in etapas+['Volume']:
          train = df_merged.loc[df_merged[e]>0,[e]] # Não queremos considerar aberturas sem histórico
          train = train.astype(float).to_numpy(dtype=float)
          gm = GaussianMixture(n_components=2, random_state=0).fit(train)
          means = gm.means_
          stds = [ np.sqrt(  np.trace(gm.covariances_[i])/2) for i in range(0,2) ]

          # Vamos ordenar corretamente as médias e desvios das gaussianas:
          if means[-1] < means[0]:
            means_copy = means.copy()
            means[-1] = means_copy[0]
            means[0] = means_copy[-1]
            stds_copy = stds.copy()
            stds[-1] = stds_copy[0]
            stds[0] = stds_copy[-1]

          means_treshold = means[:,0] + np.array([1.,-1.])*np.array([outlier_treshold,outlier_treshold])*stds

          if means_treshold[-1] < means_treshold[0]:
            means_treshold[-1] = means[-1] - ((means[-1] - means[0]) / sum(stds))*stds[-1]

          df_limite_std[e] = means_treshold[-1]


        # Vamos verificar cada abertura em quantas etapas ela seria considerada um outlier:
        df_merged['outlier'] = 0
        for e in etapas+['Volume']:
          df_merged['outlier'+str(e)] = 0
          df_merged.loc[df_merged[e] >= df_limite_std[e].values[0],['outlier'+str(e)]] = 1
          df_merged['outlier'] = df_merged['outlier'].values + df_merged['outlier'+str(e)].values
          df_merged = df_merged.drop(columns=['outlier'+str(e)])

        df_merged[etapas_sum+['Volume']] = np.where(df_merged[etapas_sum+['Volume']] > 0, 1, 0)
        df_merged['etapas_nao_zeradas'] = df_merged[etapas_sum+['Volume']].sum(axis=1)
        df_merged['%_etapas_outliers'] = df_merged['outlier'].values / df_merged['etapas_nao_zeradas'].values
        df_merged['outlier'] = np.where(df_merged['%_etapas_outliers'] >= 0.5, 1, 0)
        #Também vamos considerar aberturas ruins aquelas que tiverem qualquer uma das etapas comuns
        # (aquelas que independem do fluxo do topo) zeradas. Definimos como outlier quando > 50% das
        # etapas estão zeradas pois, para alguns funis e aberturas, certas etapas do funil realmente
        # são zeradas, como o vb2os no fluxo direct.

        for e in etapas:
          if e.split('2')[0] in topos:
            etapas_comuns = etapas[etapas.index(e):]

        df_merged['outlier'] = np.where(df_merged['etapas_nao_zeradas'] < len(etapas_comuns), 1, df_merged['outlier'])


        # Também vamos classificar como outlier a abertura cuja última etapa do funil esteja zerada:
        # Essa classificação em outlier está errada
        # df_merged['outlier'] = np.where(df_merged[ultima_etapa] == 0, 1, df_merged['outlier'])


        # Retornamos apenas as colunas que importam
        df_merged = df_merged[aberturas+['outlier']]


    else:
      df_merged = pd.DataFrame()


    return df_merged
