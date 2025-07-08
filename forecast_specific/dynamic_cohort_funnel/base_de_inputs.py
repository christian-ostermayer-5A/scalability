#@title Def base_de_inputs
from itertools import product
import pandas as pd
import numpy as np

def base_de_inputs(df_inputs,  # DataFrame contendo os inputs
                   modelo,     # Só precisa ser um DataFrame contendo todos os ítens de todas as [aberturas + 'Week Origin']
                   aberturas,  # Lista com o nome das aberturas. Ex: ['city_group', 'mkt_channel', 'lead']
                   max_origin, # Inteiro representando a maior cohort da base. Ex: 5
                   col_data,   # Noma da coluna que contém as datas. Deve ser o mesmo nome usado nas outras bases. Ex: 'week_start'
                   dia_inicio, # DateTime indicando a data mínima da aplicação dos inputs. Se na base de inputs houver uma data menor do que essa, será cortada nessa
                   dia_fim):   # DateTime indicando a data máxima da aplicação dos inputs. Se na base de inputs houver uma data maior do que essa, será cortada nessa

  # Criamos uma cópia do DataFrame modelo só para não correr o risco de alterá-lo
  modelo_base = modelo.copy()

  # Renomeamos as colunas caso tenham passado pelos checks e ficado minúsculas:
  df_inputs = df_inputs.rename(columns={'data início':'Data Início','data fim':'Data Fim','etapa':'Etapa','cohort':'Cohort','métrica':'Métrica','input':'Input'})


  # Transformamos os valores dos inputs em float
  df_inputs['Input'] = df_inputs['Input'].astype(float)

  # Definimos o formato da base final de inputs. Será essa base que vai ser multiplicada pelo baseline de conversões
  base_inputs = pd.DataFrame(columns=[col_data,'Etapa','Week Origin']+aberturas+['Métrica','Input'])

  # São essas as 'chaves' nas quais os inputs estão distribuídos. Vamos percorrer a base de input
  # linha por linha e definir os ítens que serão incluídos nessas chaves a cada linha.
  chaves_modelo = [col_data,'Week Origin']+aberturas

  #Para cada linha de df_inputs, vamos criar uma base de inputs com todas as datas e aberturas
  #_________________________________________________________________________________________________
  for l in range(len(df_inputs)):

    # definimos as datas iniciais e finais da aplicação do input com base nessas duas colunas da base de inputs
    #-----------------------------------------------------------------------------------------------
    start_date = df_inputs['Data Início'].values[l]
    end_date = df_inputs['Data Fim'].values[l]

    if end_date == 'Final':
      end_date = dia_fim
    if start_date == 'Início':
      start_date = dia_inicio

    start_date = pd.to_datetime(start_date, infer_datetime_format=True)
    end_date = pd.to_datetime(end_date, infer_datetime_format=True)

    if start_date < dia_inicio:
      start_date = dia_inicio
    if end_date > dia_fim:
      end_date = dia_fim
    #-----------------------------------------------------------------------------------------------


    # Vamos criar uma base com a repetição de todas as aberturas nas menores e maiores datas
    # Esse vai ser nosso modelo para mergear com o histórico e variáveis exógenas, de forma que
    # quando uma abertura não existir em alguma dessas bases a partir de alguma data, conseguimos incluir
    # um valor zerado no lugar e não prejudicar as equações do forecast com buracos nas bases:
    #-----------------------------------------------------------------------------------------------
    # Criamos uma cópia da lista de chaves, que receberá uma lista de ítens para cada chave (será uma lista de listas)
    list_of_columns_list = chaves_modelo.copy()
    # Indicamos em qual posição da lista "chaves_modelo" estamos. Iniciamos na "0"
    posi = 0
    # Para cada chave na lista de chaves:
    for coluna in chaves_modelo:

      # Sabendo que a primeira chave é a data, a lista de ítens dentro da chave de datas vão ser todas
      # as semanas compreendidas entre a data inicial e final da aplicação do input da linha "l" da
      # base de inputs:
      #.............................................................................................
      if coluna == chaves_modelo[0]:
        lista = list(pd.date_range(start=start_date, end=end_date, freq='7D'))

      #.............................................................................................


      # Caso a chave seja a 'Week Origin', devemos selecionar as cohorts de aplicação dos inputs
      # com base na coluna 'cohort' da base de inputs:
      #.............................................................................................
      elif coluna == chaves_modelo[1]:
        # Caso a métrica seja volume, vamos selecionar todas as cohorts
        if df_inputs['Métrica'].values[l] == 'Volume':
          lista = list(range(0,max_origin+1))+['Coincident']

        # Caso a métrica seja seja cohort:
        else:

          # Caso a aplicação na coluna cohort seja 'Coincident',
          # criamos uma lista com todas as cohorts fechadas, excluíndo a última e adicionando a
          # cohort de ajuste (coincident)
          if df_inputs['Cohort'].values[l] == 'Coincident':
            lista = list(range(0,max_origin))+['Coincident']

          # Caso a aplicação seja 'Aberta', selecionamos todas as cohorts incluíndo a última e excluíndo a de ajuste
          elif df_inputs['Cohort'].values[l] == 'Aberta':
            lista = list(range(0,max_origin+1))

          # Caso a aplicação seja uma cohort específica, a lista conterá apenas a cohort escolhida
          else:
            lista = [df_inputs['Cohort'].values[l]]
        #.............................................................................................



      # Caso a chave seja alguma das outras aberturas:
      #.............................................................................................
      else:
        # Caso a aplicação seja 'Todos', criamos uma lista contendo todos os valores únicos daquela abertura
        # a partir da base modelo que fornecemos
        if df_inputs[coluna].values[l] == 'Todos':
          lista = list(pd.unique(modelo_base[coluna].values))

        # Caso seja uma abertura específica, criamos uma lista contendo apenas um ítem
        else:
          lista = [df_inputs[coluna].values[l]]
      #.............................................................................................

      # Adicionamos a lista de ítens na chave na posição da lista de lista referente à posição da chave
      # na lista de chaves:
      list_of_columns_list[posi] = lista

      # Atualizamos a posição na lista de chaves
      posi = posi+1
    #-----------------------------------------------------------------------------------------------


    # Com a lista de listas criadas, usamos a função 'product' para criar um DataFrame contendo
    # todas as combinações possíveis entre os ítens da lista de listas
    modelo_l = pd.DataFrame(list(product(*list_of_columns_list)), columns=chaves_modelo)


    # Em seguida, adicionamos a coluna de 'Etapa' onde o input daquela linha será aplicado
    modelo_l['Etapa'] = df_inputs['Etapa'].values[l]

    # Adicionamos a coluna 'Métrica' informando a métrica onde o input será aplicado
    modelo_l['Métrica'] = df_inputs['Métrica'].values[l]

    # Caso a métrica seja volume, a etapa de conversão deve ser transformada na primeira etapa de
    # volume daquela conversão

    if modelo_l['Métrica'].values[0] == 'Volume':
      modelo_l['Métrica'] = modelo_l['Etapa'].values[0].split('2')[0]

    # Caso contrário, a métrica de aplicação é a conversão indicada na etapa
    else:
      modelo_l['Métrica'] = modelo_l['Etapa'].values[0]

    # Por fim, repetimos o valor dos inputs na linha "l" na base "modelo_l"
    modelo_l['Input'] = df_inputs['Input'].values[l]


    # Caso a aplicação da cohort seja "coincident", temos que deslocar as datas de aplicação de acordo com a cohort:
    if df_inputs['Cohort'].values[l] == 'Coincident' and df_inputs['Métrica'].values[l] != 'Volume':
      modelo_l['Aux'] = modelo_l['Week Origin'].values
      modelo_l.loc[modelo_l['Aux'] == 'Coincident',['Aux']] = max_origin
      modelo_l['Aux'] = modelo_l['Aux'].astype(int)
      modelo_l['Aux'] = modelo_l['Aux'].apply(lambda x: pd.Timedelta(x*7, unit='D'))

      modelo_l[col_data] = modelo_l[col_data] - modelo_l['Aux']

      modelo_l = modelo_l.drop(columns=['Aux'])

    # Adicionamos a base referente à linha "l" na base final total de inputs
    base_inputs = pd.concat([base_inputs,modelo_l])
  #_________________________________________________________________________________________________




  # Agora precisamos agrupar os inputs repetidos:
  #_________________________________________________________________________________________________

  # Somamos os inputs que se repetem nas aberturas
  base_inputs = base_inputs.groupby(chaves_modelo+['Etapa','Métrica'], as_index=False)[['Input']].sum()

  # Vamos substituir os nomes das métricas por i_Métrica, para facilitar o merge com a base final:
  for metrica in np.unique(base_inputs['Métrica'].values):
    base_inputs['Métrica'].replace(metrica,"i_"+metrica,inplace=True)

  # Vamos pivotar as métricas para conseguir mergear com a base final corretamente
  #base_inputs = pd.pivot_table(base_inputs, values='Input', columns='Métrica')
  base_inputs = base_inputs.pivot(index=chaves_modelo+['Etapa'],columns='Métrica',values='Input').reset_index(inplace=False).fillna(0) # Transformamos novamente para que tenhamos uma coluna com valores para cada conversão.

  return base_inputs


