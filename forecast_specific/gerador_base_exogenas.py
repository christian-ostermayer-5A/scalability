#@title Def gerador_base_exogenas
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from data_import import *
from data_export import *
from checks import *
'''
alterados:

leitura painel de controle
abertura de bases
abertura da base
abertura do arquivo
checks
check valore negativos
'''

def gerador_base_exogenas(base_modelo,
                          dict_renames,
                          nome_do_arquivo,
                          coluna_de_conversoes,
                          aberturas_da_base_modelo,
                          col_data_base_modelo,
                          df_lista_bases_exo,
                          utilizar_anterior,
                          arquivo_base_exo_exportada,
                          nome_base_exo_exportada,
                          qtd_semanas,
                          data_inicio_historico,
                          data_fim_historico,
                          data_fim_forecast,
                         client = 'client'):

                       
  if not utilizar_anterior:

    mensagem = ''

    nome_base_modelo = base_modelo.name

    base_modelo_copy = base_modelo.copy()
    base_modelo_copy.name = nome_base_modelo

    base_modelo_final = pd.DataFrame()

    # Vamos abrir cada uma das bases exógenas:

    lista_de_bases = list(df_lista_bases_exo['Base Exógena'].values)
    lista_de_nomes_das_bases = list(df_lista_bases_exo['Nome'].values)
    lista_de_arquivos_das_bases = list(df_lista_bases_exo['Arquivo'].values)

    lista_de_col_valores = list(df_lista_bases_exo['Valores'].values)

    lista_de_lista_col_valores = []
    for col in lista_de_col_valores:
      lista_valores = col.split(',')
      if len(lista_valores) > 0:
        lista_valores = [l.strip() for l in lista_valores]
      else:
        lista_valores = [col]
      lista_de_lista_col_valores = lista_de_lista_col_valores + [lista_valores]

    lista_de_agrupamento = list(df_lista_bases_exo['Agrupamento'].values)
    lista_de_filtros = list(df_lista_bases_exo['Filtros'].values)
    lista_de_col_datas = list(df_lista_bases_exo['Data'].values)
    lista_de_modelos = list(df_lista_bases_exo['Modelo de Projeção'].values)

    lista_obrigatoria = [True for x in lista_de_bases]


    if len(lista_de_arquivos_das_bases) > 0:
      lista_de_bases_exo,flag_abriu = abertura_das_bases(lista_de_nomes_das_bases = lista_de_nomes_das_bases,  # lista de strings com os nomes das bases (nomes dos arquivos csv ou das abas no sheets)
                                                        lista_de_arquivos_das_bases = lista_de_arquivos_das_bases, # lista de strings com os nomes dos arquivos onde estão as bases (nome dos sheets ou o caminho onde se encontra o csv)
                                                        colunas_para_dropar = [],       # lista de listas de inteiros, indicando a posição das colunas a serem excluídas de cada base (lista vazia caso nenhuma coluna a ser excluída)
                                                        lista_obrigatoria = lista_obrigatoria,         # lista de booleanos indicando se a existência daquela base é obrigatória
                                                        nome_do_painel_de_controle = nome_do_arquivo,           # string com o nome do arquivo sheets onde se encontra o painel de controle (somente para imprimir mensagens de erro)
                                                        sheets_painel_de_controle = [],
                                                        client = client)           # arquivo sheets do painel de controle aberto
      
    else:
      flag_abriu = False

    if flag_abriu:

      # Realizar um pré-check geral na base modelo:
      #-----------------------------------------------------------------------------------------------

      # Geramos a lista de vases que serão verificadas
      lista_de_bases_check = lista_de_bases_exo.copy()#+[base_modelo_copy]

      # Registramos o nome das colunas originais para substituir depois,
      # pois os checks transformam tudo em minúsculo:
      colunas_originais = []
      for b in lista_de_bases_check:
        colunas_originais = colunas_originais + [list(b.columns.values)]


      # Iniciamos a lista de colunas obrigatórias das bases de exógenas.
      # Adicionamos as colunas de datas e de valores indicadas no paiel de controle com todas as colunas
      # encontradas nas bases e removemos duplicadas. Fazemos isso para garantir que não
      # vamos remover colunas de aberturas que podem ser compatíveis e também por não sabermos
      # como agrupar os valores restantes.
      lista_de_colunas_obrigatorias = lista_de_lista_col_valores.copy()
      l=0
      while l < len(lista_de_colunas_obrigatorias):
        col_filtros = []
        if lista_de_filtros[l] != '':
          lista_col_filtros = lista_de_filtros[l].split(",")
          if len(lista_col_filtros) > 0:
            lista_col_filtros = [c.strip() for c in lista_col_filtros]
          else:
            lista_col_filtros = [lista_de_filtros[l]]
          for e in lista_col_filtros:
            col_filtros = col_filtros + [e.split(":")[0]]


        lista_de_colunas_obrigatorias[l] = list(dict.fromkeys(lista_de_colunas_obrigatorias[l]+[lista_de_col_datas[l]]+col_filtros+list(lista_de_bases_exo[l].columns.values)))
        l += 1


      lista_de_colunas_obrigatorias = lista_de_colunas_obrigatorias#+[list(dict.fromkeys([col_data_base_modelo]+aberturas_da_base_modelo+list(base_modelo_copy.columns.values)))]

      lista_colunas_de_valores = lista_de_lista_col_valores#+[[]]


      lista_do_retorno_de_valores = [False for f in lista_de_bases]#+[False]

      lista_mantem_formatacao_original = [False for f in lista_de_bases]#+[False]

      lista_check_vazios = [True for f in lista_de_bases]#+[True]

      lista_verifica_valores = lista_mantem_formatacao_original

      lista_de_bases_checar_chaves = lista_mantem_formatacao_original

      lista_comparacao_parcial = lista_check_vazios

      lista_comparacao_a_mais = lista_check_vazios

      aberturas_especificas = []

      lista_lista_colunas_datas = [[f] for f in lista_de_col_datas]#+[[col_data_base_modelo]]

      lista_lista_frequencia = [['None'] for f in lista_de_col_datas]#+[['W-Mon']]

      chaves_ignoradas = []

      lista_de_bases_check, colunas_de_valores, mensagem_local, erro_local = check_geral(lista_de_bases = lista_de_bases_check,
                                                                                        lista_de_colunas_obrigatorias = lista_de_colunas_obrigatorias,
                                                                                        lista_colunas_de_valores = lista_colunas_de_valores,
                                                                                        lista_do_retorno_de_valores = lista_do_retorno_de_valores,
                                                                                        lista_mantem_formatacao_original = lista_mantem_formatacao_original,
                                                                                        lista_check_vazios = lista_check_vazios,
                                                                                        lista_verifica_valores = lista_verifica_valores,
                                                                                        lista_de_bases_checar_chaves = lista_de_bases_checar_chaves,
                                                                                        lista_comparacao_parcial = lista_comparacao_parcial,
                                                                                        lista_comparacao_a_mais = lista_comparacao_a_mais,
                                                                                        aberturas_especificas = aberturas_especificas,
                                                                                        lista_lista_colunas_datas = lista_lista_colunas_datas,
                                                                                        lista_lista_frequencia = lista_lista_frequencia,
                                                                                        chaves_ignoradas = chaves_ignoradas,
                                                                                        aberturas_das_bases = aberturas_da_base_modelo,
                                                                                        coluna_de_conversoes = coluna_de_conversoes,
                                                                                        dict_renames = dict_renames,
                                                                                        tipo_de_tof = '',
                                                                                        Nome_do_arquivo_sheets = nome_do_arquivo)

      mensagem = mensagem + mensagem_local
      if erro_local == 0:

        # Caso não tenha sido encontrado nenhum erro nas bases, vamos gerar a base final:
        ############################################################################################


        # Vamos criar uma base de exógenas modelo, com todas as chaves e todas as semanas de histórico
        # e forecast:
        base_modelo_copy['teste'] = 1
        aberturas_modelo = base_modelo_copy.groupby(aberturas_da_base_modelo, as_index=False)['teste'].sum()
        aberturas_modelo = aberturas_modelo[aberturas_da_base_modelo]
        # Para todas as combinações de aberturas encontradas, vamos repet-las em todas as semanas de historiuco e de forecast:

        lista_semanas_completa = list(pd.date_range(start=data_inicio_historico, end=(data_fim_forecast+pd.Timedelta(7, unit='D')), freq='W'))

        lista_base_modelo_final = [base_modelo_final]
        for semana in lista_semanas_completa:
          semana = semana - pd.Timedelta(6, unit='D') # date_range gera lista de domingos
          aux = aberturas_modelo.copy()
          aux[col_data_base_modelo] = semana
          lista_base_modelo_final = lista_base_modelo_final + [aux]
        base_modelo_final = pd.concat(lista_base_modelo_final)

        base_modelo_final[col_data_base_modelo] = pd.to_datetime(base_modelo_final[col_data_base_modelo], infer_datetime_format=True)


        # Agora, para cada base exogena, vamos tratá-la e adicioná-la à base modelo:
        #___________________________________________________________________________________________

        lista_de_bases_exo_semanais = []

        for b in range(len(lista_de_bases_check)):

          # Precisamos formatar novamente as colunas pois não utilizamos as bases que saem dos checks, pois elas estão com as colunas trocadas ou modificadas
          base_exo = lista_de_bases_exo[b]
          for d in lista_lista_colunas_datas[b]:
            base_exo[d] = pd.to_datetime(base_exo[d], infer_datetime_format=True)
          for v in lista_de_lista_col_valores[b]:
            base_exo[v] = base_exo[v].replace('','0')
            base_exo[v] = base_exo[v].astype(float)

          # Renomeamos as colunas das bases conforme os nomes originais:
          base_exo.columns = colunas_originais[b]
          colunas = list(base_exo.columns.values)

          # Caso existam filtros que devem ser aplicados na base, vamos aplicá-los aqui:
          #-----------------------------------------------------------------------------------------
          tamanho_base_inicial = len(base_exo)
          if lista_de_filtros[b] != '':
            lista_col_filtros = lista_de_filtros[b].split(",")
            if len(lista_col_filtros) > 0:
              lista_col_filtros = [c.strip() for c in lista_col_filtros]
            else:
              lista_col_filtros = [lista_de_filtros[l]]

            for e in lista_col_filtros:
              filtro = e.split(":")
              coluna_filtrada = filtro[0].strip()
              chave_filtrada = filtro[1].strip()
              base_exo = base_exo.loc[base_exo[coluna_filtrada] == chave_filtrada]

          if len(base_exo) == 0 and tamanho_base_inicial != 0:

            mensagem = mensagem + '/n/nOs filtros removeram todos os dados da base'
            erro_local += 1


          if erro_local == 0:

            # Para cada base de exogenas, vamos transformá-la numa base semanal:
            #-----------------------------------------------------------------------------------------

            # Criamos uma base de de-para com os dias e as datas correspondentes da base:
            data_max = base_exo[lista_lista_colunas_datas[b]].values.max()
            data_min = base_exo[lista_lista_colunas_datas[b]].values.min()

            primeira_segunda = data_min - pd.Timedelta(pd.to_datetime(str(data_min)).weekday(), unit='D')
            if pd.to_datetime(str(data_max)).weekday() != 6:
              ultimo_domingo = data_max + pd.Timedelta((6-pd.to_datetime(str(data_max)).weekday()), unit='D')
            else:
              ultimo_domingo = data_max

            lista_dias = list(pd.date_range(start=primeira_segunda, end=ultimo_domingo))
            lista_dias = [d.strftime("%m/%d/%Y") for d in lista_dias]

            lista_datas_unicas = list(base_exo[lista_lista_colunas_datas[b][0]].sort_values(ascending=True).unique())
            lista_datas_unicas = [pd.to_datetime(str(d)).strftime("%m/%d/%Y") for d in lista_datas_unicas]

            lista_de_para = []

            lista_semanas = []

            d_datas_unicas = 0
            segunda = primeira_segunda
            for d in lista_dias:
              if d_datas_unicas == len(lista_datas_unicas)-1:
                lista_de_para = lista_de_para + [lista_datas_unicas[d_datas_unicas]]
              else:
                if pd.to_datetime(d) >= pd.to_datetime(lista_datas_unicas[d_datas_unicas+1]):
                  lista_de_para = lista_de_para + [lista_datas_unicas[d_datas_unicas+1]]
                  d_datas_unicas += 1
                else:
                  lista_de_para = lista_de_para + [lista_datas_unicas[d_datas_unicas]]
              if pd.to_datetime(d).weekday() == 0:
                segunda = d
              lista_semanas = lista_semanas + [segunda]

            de_para = {'de_para_semanas':lista_semanas,'de_para_dias':lista_dias,lista_lista_colunas_datas[b][0]:lista_de_para}
            de_para = pd.DataFrame(de_para)
            de_para['de_para_semanas'] = pd.to_datetime(de_para['de_para_semanas'], infer_datetime_format=True)
            de_para['de_para_dias'] = pd.to_datetime(de_para['de_para_dias'], infer_datetime_format=True)
            de_para[lista_lista_colunas_datas[b][0]] = pd.to_datetime(de_para[lista_lista_colunas_datas[b][0]], infer_datetime_format=True)

            # Depois de criar a base de_para, vamos unir com a base original:
            base_exo = pd.merge(base_exo,de_para,how='outer',on=lista_lista_colunas_datas[b])


            # Agrupamos os valores na semana:
            #-----------------------------------------------------------------------------------------
            base_exo[lista_de_lista_col_valores[b]] = base_exo[lista_de_lista_col_valores[b]].astype(float)

            chaves = list(set(colunas) - set(lista_lista_colunas_datas[b]+lista_de_lista_col_valores[b])) + ['de_para_semanas']
            aberturas_exo = list(set(chaves) - set(['de_para_semanas']))




            # Caso a métrica não seja aditiva, vamos agrupar nas semanas o valores que aparecem nos
            # domingos de cada semana.
            if lista_de_agrupamento[b] == 'não aditivo':
              base_exo_domingo = base_exo.copy()
              base_exo_domingo['de_para_semanas'] = base_exo_domingo['de_para_dias'].apply(lambda x: x-pd.Timedelta(6, unit='D'))
              base_exo_domingo['day_of_week'] = base_exo_domingo['de_para_semanas'].dt.dayofweek
              base_exo_domingo = base_exo_domingo.loc[base_exo_domingo['day_of_week'] == 0.0]

              base_exo = base_exo_domingo
              base_exo = base_exo.groupby(chaves, as_index=False)[lista_de_lista_col_valores[b]].mean()

            # Caso as datas originais estavam numa frequencia menor do que o número de semanas únicas
            # dentro do intervalo de data mínima e máxima, repetimos os valores de maior granularidade
            # nas semanas correspondentes, agrupando a base unida via média:
            if len(list(dict.fromkeys(lista_semanas))) >= len(lista_datas_unicas) or lista_de_agrupamento[b] == 'média':
              base_exo = base_exo.groupby(chaves, as_index=False)[lista_de_lista_col_valores[b]].mean()

            # Caso contrário, vamos agrupar as os valores nas semanas de acordo com a regra no painel de controle:

            if lista_de_agrupamento[b] == 'max':
              base_exo = base_exo.groupby(chaves, as_index=False)[lista_de_lista_col_valores[b]].max()
            if lista_de_agrupamento[b] == 'min':
              base_exo = base_exo.groupby(chaves, as_index=False)[lista_de_lista_col_valores[b]].min()
            if lista_de_agrupamento[b] == 'soma':
              base_exo = base_exo.groupby(chaves, as_index=False)[lista_de_lista_col_valores[b]].sum()



            colunas = list(base_exo.columns.values)

            # Após agrupar nas semanas, vamos unir a base de exógenas com a base modelo:
            #-----------------------------------------------------------------------------------------

            # Vamos determinar quais colunas estão faltando e quais estão sobrando:
            colunas_faltantes = list(set(aberturas_da_base_modelo) - set(aberturas_exo))
            colunas_extra = list(set(aberturas_exo) - set(aberturas_da_base_modelo))
            '''
            print("_____________________________________________________________")
            print(base_exo.columns.values)
            print(colunas_faltantes)
            '''
            if len(colunas_faltantes) > 0:

              # Vamos tentar encontrar as colunas faltantes na base modelo, mas somente aquelas que sejam aberturas:
              colunas_substituidas = []
              for col_faltante in colunas_faltantes:
                df_modelo_copy = aberturas_modelo.copy()
                conteudo_modelo = list(df_modelo_copy[col_faltante].unique())
                nome_da_base_modelo = nome_base_modelo

                # Vamos procurar qual coluna da base sendo testada tem a maior compatibilidade com o conteúdo modelo:
                comparacao = []
                comparacao_faltantes = []
                for col in colunas:
                  conteudo_col = list(base_exo[col].unique())
                  faltantes = list(set(conteudo_modelo) - set(conteudo_col))
                  extras = list(set(conteudo_col) - set(conteudo_modelo))
                  diferenca = faltantes + extras
                  comparacao = comparacao+[len(diferenca)]
                  comparacao_faltantes = comparacao_faltantes+[len(faltantes)]

                try:

                  # Caso tenha sido encontrada somente uma coluna na base original com o máximo de compatibilidade, vamos renomer a coluna:
                  maior_compatibilidade = min(comparacao)
                  coluna_compativel = colunas[comparacao.index(maior_compatibilidade)]
                  numero_faltante = comparacao_faltantes[comparacao.index(maior_compatibilidade)]

                  # Não queremos substituir colunas onde estão faltando mais da metade das chaves
                  # da base modelo:
                  if numero_faltante < len(conteudo_modelo)/2:

                    base_exo = base_exo.rename(columns={coluna_compativel:col_faltante})
                    colunas = list(base_exo.columns.values)
                    colunas_substituidas = colunas_substituidas + [col_faltante]
                    mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored('nome_da_base','yellow')+', não foi encontrada a coluna: '+colored(str(col_faltante),'red')+'\nPorém, o conteúdo da coluna '+colored(str(coluna_compativel),'red')+' parece ser o mais compatível com o conteúdo da coluna '+colored(str(col_faltante),'red')+' encontrada na base modelo '+colored(nome_da_base_modelo,'yellow')+'\nVai ser assumido que estas colunas são equivalentes.'

                except:
                  colunas_substituidas = colunas_substituidas

            # Após renomear e identificar possíveis colunas compatíveis, vamos agrupar a base exógena
            # somente nas aberturas que existirem na base modelo:

            colunas_faltantes = list(set(aberturas_da_base_modelo) - set(colunas))

            chaves_finais = ['de_para_semanas'] + list(set(aberturas_da_base_modelo) - set(colunas_faltantes))

            if lista_de_agrupamento[b] == 'média':
              base_exo = base_exo.groupby(chaves_finais, as_index=False)[lista_de_lista_col_valores[b]].mean()
            if lista_de_agrupamento[b] == 'max':
              base_exo = base_exo.groupby(chaves_finais, as_index=False)[lista_de_lista_col_valores[b]].max()
            if lista_de_agrupamento[b] == 'min':
              base_exo = base_exo.groupby(chaves_finais, as_index=False)[lista_de_lista_col_valores[b]].min()
            if lista_de_agrupamento[b] == 'soma' or lista_de_agrupamento[b] == 'não aditivo':
              base_exo = base_exo.groupby(chaves_finais, as_index=False)[lista_de_lista_col_valores[b]].sum()

            base_exo = base_exo.rename(columns={'de_para_semanas':col_data_base_modelo})

            chaves_finais = [col_data_base_modelo] + list(set(aberturas_da_base_modelo) - set(colunas_faltantes))

            # Unir com a base modelo nas chaves compatíveis
            base_modelo_final = pd.merge(base_modelo_final,base_exo,how='left',on=chaves_finais)



        # Com todas as bases unidas, vamos agrupar as exógenas que estão separadas entre histórico e projeção:
        #___________________________________________________________________________________________

        lista_de_bases_unicas = list(dict.fromkeys(lista_de_bases))

        # Para cada conjunto de bases com as mesmas exógenas, vamos agrupar as colunas de valores
        # em listas que tratam do mesmo valor. Vamos assumir que a ordem que as colunas aparecem no painel de controle
        # é a mesma em todas as bases.
        lista_listas_exogenas_repetidas = []
        for b in lista_de_bases_unicas:

          df_lista_bases_exo_filtrada = df_lista_bases_exo.loc[df_lista_bases_exo['Base Exógena'] == b]

          if len(df_lista_bases_exo_filtrada['Base Exógena'].values)>1:

            lista_de_col_valores = list(df_lista_bases_exo_filtrada['Valores'].values)
            lista_de_lista_col_valores = []
            for col in lista_de_col_valores:
              lista_valores = col.split(',')
              if len(lista_valores) > 0:
                lista_valores = [l.strip() for l in lista_valores]
              else:
                lista_valores = [col]
              lista_de_lista_col_valores = lista_de_lista_col_valores + [lista_valores]

            lista_de_lista_col_valores_reagrupada = []
            for l in range(len(lista_de_lista_col_valores[0])):
              lista_de_lista_col_valores_reagrupada = lista_de_lista_col_valores_reagrupada + [[item[l] for item in lista_de_lista_col_valores]]


            # Com as listas de colunas exógenas reagrupadas, vamos renomear os nomes repetidos para ficarem
            # iguais aos nomes que estão na base final.
            for lista_exo in lista_de_lista_col_valores_reagrupada:
              lista_exo_original = lista_exo.copy()
              df_renomear_colunas = pd.DataFrame([[1]],
                                                  index=[1],
                                                  columns=['teste'])
              for col in lista_exo:
                df_coluna_exo = pd.DataFrame([[1,1]],
                                            index=[1],
                                            columns=['teste',col])
                df_renomear_colunas = pd.merge(df_renomear_colunas,df_coluna_exo,how='left',on=['teste'])
              lista_exo = list(df_renomear_colunas.columns.values)[1:]



              # Para cada lista de colunas exógenas iguais, vamos agrupar de acordo com a abrangência das datas:
              #-------------------------------------------------------------------------------------

              # Definimos o nome que a coluna agrupada final vai ter:
              nomes_desduplicados = list(dict.fromkeys(lista_exo_original))
              if len(nomes_desduplicados) == 1:
                nome_exo_novo = nomes_desduplicados[0]
                if len(lista_exo_original) >= 3:
                  lista_exo[lista_exo.index(nome_exo_novo)] = nome_exo_novo+'_z'
                  base_modelo_final = base_modelo_final.rename(columns={nome_exo_novo:nome_exo_novo+'_z'})
              else:
                nome_exo_novo = '_&_'.join(nomes_desduplicados)


              # Definimos uma lista contendo a data máxima e mínima dos valores de cada coluna que não são nulos
              datas_max_min = []
              for col in lista_exo:

                base_modelo_final_filtrada = base_modelo_final[base_modelo_final[col].notnull()]
                data_max = base_modelo_final_filtrada[col_data_base_modelo].values.max()
                data_min = base_modelo_final_filtrada[col_data_base_modelo].values.min()
                datas_max_min = datas_max_min + [[col,data_max,data_min]]

              df_datas_max_min = pd.DataFrame(datas_max_min,
                                            columns=['exo','data_max','data_min'])

              # Para cada coluna de exógena, vamos agrupar na base final pela ordem das datas máximas:
              for col in range(len(lista_exo)):
                if col == 0:
                  data_limite_inicial = df_datas_max_min['data_max'].values.min()
                  exo_inicial = df_datas_max_min.loc[df_datas_max_min['data_max'] == data_limite_inicial,['exo']].values
                  if len(exo_inicial) > 1:
                    data_limite_inicial = df_datas_max_min['data_min'].values.min()
                    data_hist = data_limite_inicial
                    exo_inicial = df_datas_max_min.loc[df_datas_max_min['data_min'] == data_limite_inicial,['exo']].values[0][0]
                    data_limite_inicial = df_datas_max_min.loc[df_datas_max_min['exo'] == exo_inicial,['data_max']].values.min()#-pd.Timedelta(1, unit='D')
                  else:
                    exo_inicial = exo_inicial[0][0]
                    data_hist = df_datas_max_min.loc[df_datas_max_min['exo'] == exo_inicial,['data_min']].values[0][0]

                  exo_hist = exo_inicial
                  base_modelo_final[nome_exo_novo] = base_modelo_final[exo_inicial].values
                else:

                  data_limite = df_datas_max_min.loc[df_datas_max_min['exo'] != exo_inicial,['data_max']].values.min()
                  exo = df_datas_max_min.loc[(df_datas_max_min['data_max'] == data_limite) & (df_datas_max_min['exo'] != exo_inicial),['exo']].values[0][0]
                  base_modelo_final.loc[base_modelo_final[col_data_base_modelo] > data_limite_inicial,[nome_exo_novo]] = base_modelo_final.loc[base_modelo_final[col_data_base_modelo] > data_limite_inicial,[exo]].values
                  data_limite_inicial = data_limite
                  exo_inicial = exo


                col += 1

              # Agora, caso a exógena de histórico tenha uma data mínima menor do que a data mínina de outra exógena,
              # vamos substituir os valores passados pela exógena de maior histórico:
              datas_menores = df_datas_max_min.loc[df_datas_max_min['data_min'] < data_hist,['data_min']].values
              if len(datas_menores) > 0:
                data_menor = datas_menores.min()
                exo_menor = df_datas_max_min.loc[df_datas_max_min['data_min'] == data_menor,['exo']].values[0][0]
                base_modelo_final.loc[base_modelo_final[col_data_base_modelo] < data_hist,[nome_exo_novo]] = base_modelo_final.loc[base_modelo_final[col_data_base_modelo] < data_hist,[exo_menor]].values
              # Removemos as colunas antigas que estavam separadas
              base_modelo_final = base_modelo_final.drop(columns = lista_exo)



        # Com todas as exógenas agrupadas com histórico e projeção unidos, vamos identificar os dados
        # faltantes e fazer o preenchimento das lacunas.
        #___________________________________________________________________________________________

        exogenas_finais = list(set(list(base_modelo_final.columns.values)) - set(aberturas_da_base_modelo+[col_data_base_modelo]))

        df_lista_bases_exo_aux = df_lista_bases_exo.copy()
        #df_lista_bases_exo_aux['Valores'] = df_lista_bases_exo_aux['Valores'].apply(lambda x: str(x).lower())

        base_modelo_final_concat = pd.DataFrame()
        lista_base_modelo_final_concat = [base_modelo_final_concat]

        # Para cada abertura da base, vamos realizar a projeção:
        for linha in range(len(aberturas_modelo)):
          chaves = list(aberturas_modelo.iloc[linha,:].values)
          base_modelo_final_filtrada = base_modelo_final.copy()

          # Vamos filtrar a base final nas chaves selecionadas:
          for abertura in range(len(aberturas_da_base_modelo)):
            base_modelo_final_filtrada = base_modelo_final_filtrada.loc[base_modelo_final_filtrada[aberturas_da_base_modelo[abertura]] == chaves[abertura]]

          # Para cada série exógena:
          for exo in exogenas_finais:

            # Vamos encontrar qual deveria ser o modelo de projeção dessa exógena:
            if "_&_" in exo:
              exo_aux = exo.split('_&_')[0]
            else:
              exo_aux = exo
            for v in range(len(list(df_lista_bases_exo_aux['Valores'].values))):
              valor = list(df_lista_bases_exo_aux['Valores'].values)[v].split(',')
              if len(valor) > 0:
                valor = [l.strip() for l in valor]
              else:
                valor = [valor]
              if exo_aux in valor:
                modelo_exo = lista_de_modelos[v]

            # Definimos o valor de substituição caso não seja nenhum modelo que envolve os valores da abertura
            if modelo_exo not in ['Média','Média Móvel','Média Móvel Ponderada Linear','Média Móvel Ponderada Exponencial','Linear','Repetir','Zerar','']:
              try:
                subs = float(modelo_exo)
              except:
                subs = 0
            else:
              subs = 0



            # Checar se estão faltando dados nessa abertura:
            base_modelo_final_filtrada_vazios = base_modelo_final_filtrada.loc[base_modelo_final_filtrada[exo].isnull()]

            if len(base_modelo_final_filtrada_vazios) > 0:

              # Checar se existem valores:
              base_modelo_final_filtrada_n_vazios = base_modelo_final_filtrada.loc[base_modelo_final_filtrada[exo].notnull()]

              if len(base_modelo_final_filtrada_n_vazios) == 0 or modelo_exo not in ['Média','Média Móvel','Média Móvel Ponderada Linear','Média Móvel Ponderada Exponencial','Linear','Repetir']:
                base_modelo_final_filtrada[exo] = base_modelo_final_filtrada[exo].fillna(subs)

              else:
                ultima_data = base_modelo_final_filtrada_n_vazios[col_data_base_modelo].values.max()
                primeira_data = base_modelo_final_filtrada_n_vazios[col_data_base_modelo].values.min()
                data_corte_modelo = ultima_data - pd.Timedelta(qtd_semanas*7, unit='D')

                if modelo_exo == 'Repetir':
                  ultimo_valor = base_modelo_final_filtrada_n_vazios.loc[base_modelo_final_filtrada_n_vazios[col_data_base_modelo] == ultima_data,[exo]].values[0][0]
                  base_modelo_final_filtrada.loc[base_modelo_final_filtrada[col_data_base_modelo] > ultima_data,[exo]] = base_modelo_final_filtrada.loc[base_modelo_final_filtrada[col_data_base_modelo] > ultima_data,[exo]].fillna(ultimo_valor)

                  primeiro_valor = base_modelo_final_filtrada_n_vazios.loc[base_modelo_final_filtrada_n_vazios[col_data_base_modelo] == primeira_data,[exo]].values[0][0]
                  base_modelo_final_filtrada.loc[base_modelo_final_filtrada[col_data_base_modelo] < primeira_data,[exo]] = base_modelo_final_filtrada.loc[base_modelo_final_filtrada[col_data_base_modelo] < primeira_data,[exo]].fillna(primeiro_valor)

                elif modelo_exo == 'Média':
                  valor_medio = np.average(base_modelo_final_filtrada_n_vazios.loc[(base_modelo_final_filtrada_n_vazios[col_data_base_modelo] <= ultima_data) & (base_modelo_final_filtrada_n_vazios[col_data_base_modelo] > data_corte_modelo),[exo]].values)

                  base_modelo_final_filtrada.loc[base_modelo_final_filtrada[col_data_base_modelo] > ultima_data,[exo]] = base_modelo_final_filtrada.loc[base_modelo_final_filtrada[col_data_base_modelo] > ultima_data,[exo]].fillna(valor_medio)

                else:
                  base_modelo_final_filtrada[exo] = base_modelo_final_filtrada[exo].fillna(0)
                  #base_modelo_final_filtrada[exo+'_projecao']  = base_modelo_final_filtrada[exo].rolling(qtd_semanas_media,center=False,min_periods=1).mean()
                  #base_modelo_final_filtrada['projecao'].update(base_modelo_final_filtrada[exo] )


          lista_base_modelo_final_concat = lista_base_modelo_final_concat+[base_modelo_final_filtrada.copy()]

        base_modelo_final_concat = pd.concat(lista_base_modelo_final_concat)
        base_modelo_final = base_modelo_final_concat
        base_modelo_final = base_modelo_final.fillna(0)
    else:
        # Vamos criar uma base de exógenas modelo, com todas as chaves e todas as semanas de histórico
        # e forecast:
        base_modelo_copy['teste'] = 1
        aberturas_modelo = base_modelo_copy.groupby(aberturas_da_base_modelo, as_index=False)['teste'].sum()
        aberturas_modelo = aberturas_modelo[aberturas_da_base_modelo]
        # Para todas as combinações de aberturas encontradas, vamos repet-las em todas as semanas de historiuco e de forecast:

        lista_semanas_completa = list(pd.date_range(start=data_inicio_historico, end=(data_fim_forecast+pd.Timedelta(7, unit='D')), freq='W'))

        for semana in lista_semanas_completa:
          semana = semana - pd.Timedelta(6, unit='D') # date_range gera lista de domingos
          aux = aberturas_modelo.copy()
          aux[col_data_base_modelo] = semana
          base_modelo_final = pd.concat([base_modelo_final,aux])
        base_modelo_final[col_data_base_modelo] = pd.to_datetime(base_modelo_final[col_data_base_modelo], infer_datetime_format=True)

  return base_modelo_final,mensagem











