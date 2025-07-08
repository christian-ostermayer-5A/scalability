#@title Def gerador_base_diaria

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from redutor_de_base import *
from rounding_tool import *

# Gera uma proxi de base diária a partir da base semanal

# O primeiro passo de quebrar as metas semanais numa base diária é definir os dias dessa base.
# Fazemos isso criando uma base diária que é a base semanal repetida 7 vezes, mas cada vex adicionamos 1
# dia na coluna das semanas. Assim, obtemos todos os 7 dias de todas as semanas na base.

# Os volumes ainda serão os semanais, repetidos todos os dias.

def gerador_base_diaria(base_coincident, # --> DataFrame de volumes semanais coincident
                        coluna_de_semanas):

  coluna_de_semanas_c = coluna_de_semanas

  # Encontramos a coluna com as semanas e formatamos como datetime
  base_coincident[coluna_de_semanas_c] = pd.to_datetime(base_coincident[coluna_de_semanas_c], infer_datetime_format=True)
  
  # Criamos a base diária copiando a base semanal
  base_diaria = base_coincident.copy()

  base_diaria = base_diaria.rename(columns={coluna_de_semanas_c:'col_data_original'})

  coluna_de_semanas_c = 'col_data_original'

  # Definimos a coluna das semanas na base diária como sendo as datas originais da base semanal
  base_diaria['semana'] = base_diaria[coluna_de_semanas_c].values
  base_diaria['data'] = base_diaria[coluna_de_semanas_c].values
  copy = base_diaria.copy()
  


  # Como a primeira cópia da base semanal contém apenas as segundas-feiras, vamos repetir a cópia mais 6 vezes,
  # adicionando 1 dia na coluna das datas a cada repetição
  for i in range(6):
    copy['data'] = copy[coluna_de_semanas_c].apply(lambda x: x+pd.Timedelta(i+1, unit='D'))
    base_diaria = pd.concat([copy,base_diaria])

  # Definimos as outras colunas de datas que estão presentes na base diária
  base_diaria['dia da semana'] = base_diaria['data'].dt.day_name()
  base_diaria['mês'] = base_diaria['data'].dt.month
  base_diaria['ano'] = base_diaria['data'].dt.year

  base_diaria = base_diaria.drop(columns=[coluna_de_semanas_c])
  
  return base_diaria # <-- DataFrame contendo as datas corretas da base diária (volumes ainda são os semanais repetidos todos os dias)



#@title Def base_impacto_feriados

# Gera uma base de impactos de feriados somados para todas as regiões

# Com base na informação das datas, locais e tipos de feriado, força do impacto em cada dia da semana e
# share que a cidade onde cai o feriado representa na região, usamos esta função para construir uma base
# única de impactos por região, unindo todas essas informações

def base_impacto_feriados(feriados,        # Dataframe contendo as datas, cidades e tipos de feriado
                          impactos,        # DataFrame contendo o impacto estimado de feriados
                          share_cidades):  # DataFrame contendo o share que cada cidade representa dentro de uma região


  # definimos os cabeçalhos das bases
  cb_impactos = list(impactos.columns.values)

  # definimos as etapas do funil a partir do cabeçalho da base de impactos
  etapas = cb_impactos[1:]
  # Como faremos múltiplas uniões entre bases, as colunas com as etapas repetidas adicionam "_x",
  # "_y", "_z".. nos nomes das colunas. Aqui já definimos quais etapas esses nomes estão se referindo
  etapas_share = etapas.copy()
  etapas_impactos = etapas.copy()
  etapas_finais = etapas.copy()
  for i in range(len(etapas)):
    etapas_share[i] = etapas_share[i]+"_x"
    etapas_impactos[i] = etapas_impactos[i]+"_y"
    etapas_finais[i] = etapas_finais[i]+"_z"

  if 'Dia da Semana' not in cb_impactos:
    chaves_impacto = ['dia da semana']
  else:
    chaves_impacto = ['Dia da Semana']



  if len(share_cidades) > 0:


    # Abaixo definimos a base de feriados estaduais e uma base com o restante dos feriados
    #______________________________________________

    # A base de share de cidades também possui um de-para entre região, cidade e estado.
    # Usamos essa informação para unir a base de feriados com a base de share de cidades
    feriados_E = pd.merge(feriados,share_cidades,how='left',on=['estado'])
    # Unimos essa base com a base de impactos pelas chaves definidas
    feriados_E = pd.merge(feriados_E,impactos,how='left',on=chaves_impacto)

    # Definimos uma base geral contendo os outros tipos de feriado
    feriados = pd.merge(feriados,share_cidades,how='left',on=['cidade'])
    # Unimos essa base com a base de impactos pelas chaves definidas
    feriados = pd.merge(feriados,impactos,how='left',on=chaves_impacto)

    # Definimos o cabeçalho final provisório que a base final de feriados vai ter
    cb_final = ['data','região_y']+etapas_finais

    # Criamos uma lista com todas as regiões da base
    regioes = share_cidades['região'].unique()


    # Cálculo do impacto de feriados municipais
    #______________________________________________

    # Selecionamos os feriados municipais pela coluna de tipo de feriado
    feriados_M = feriados.loc[feriados['tipo'] == 'M']
    
    # O impacto dos feriados finais no caso dos municipais é = 1-((1-impacto)*share_cidade)
    feriados_M[etapas_impactos] = 1-feriados_M[etapas_impactos].astype('float').values
    feriados_M[etapas_finais] = 1-feriados_M[etapas_impactos].multiply(feriados_M[etapas_share].astype('float').values, axis="index")

    # selecionamos apenas os dados finais
    feriados_M = feriados_M[cb_final]

    # Cálculo do impacto de feriados nacionais
    #______________________________________________

    # Selecionamos os feriados nacionais pela coluna de tipo de feriado
    feriados_N = feriados.loc[feriados['tipo'] == 'N']
    
    # O impacto dos feriados nacionais é o impacto original do feriado
    feriados_N[etapas_finais] = feriados_N[etapas_impactos].astype('float').values

    feriados_N = feriados_N[cb_final]

    # O impacto do feriado nacional é repetido para todas as regiões da base
    # Criamos uma cópia da base geral e depois repetimos ela adicionando todas as regiões
    feriados_N['região_y'] = regioes[0]
    feriados_N_final = feriados_N.copy()
    for i in range(1,len(regioes)):
      feriados_N_copy = feriados_N.copy()
      feriados_N_copy['região_y'] = regioes[i]
      frames = [feriados_N_final,feriados_N_copy]
      feriados_N_final = pd.concat(frames)


    # Cálculo do impacto de feriados estaduais
    #______________________________________________

    # Selecionamos os feriados estaduais pela coluna de tipo de feriado
    feriados_E = feriados_E.loc[feriados_E['tipo'] == 'E']

    # Definimos, inicialmente, o impacto dos feriados estaduais como sendo o impacto dos feriados sem alteração
    feriados_E[etapas_finais] = feriados_E[etapas_impactos].astype('float').values
    
    # selecionamos apenas os dados finais
    feriados_E = feriados_E[cb_final]

    # Como a base de feriados estaduais possui regiões repetidas (pois feriados estaduais que cairam em
    # cidades pertencentes à uma mesma região foram contadas individualmente), definimos o impacto final
    # do feriado estadual por região como sendo a média dos impactos das cidades dentro daquela região
    feriados_E = feriados_E.groupby(['data','região_y'], as_index=False)[etapas_finais].mean()


    # Compor todos os tipos de feriado
    #______________________________________________
    
    # Aqui só concatenamos as bases de feriados nacionais, municipais e estaduais na base final

    feriados_finais = pd.concat([feriados_N_final,feriados_E,feriados_M], axis=0)

    feriados_finais = feriados_finais.groupby(['data','região_y'], as_index=False)[etapas_finais].prod()

    feriados_finais = feriados_finais.rename(columns={"região_y": "região"})

    feriados_finais['data'] = pd.to_datetime(feriados_finais['data'], infer_datetime_format=True)

  
  # Caso não tenhamos uma base de share de cidades, vamos gerar uma base somente com as datas e impactos
  else:
    
    # Selecionamos os feriados nacionais pela coluna de tipo de feriado
    feriados_N = feriados.loc[feriados['tipo'] == 'N']
    feriados_finais = pd.merge(feriados_N,impactos,how='left',on=chaves_impacto)
    feriados_finais[etapas_finais] = feriados_finais[etapas].values
    feriados_finais = feriados_finais.groupby(['data'], as_index=False)[etapas_finais].prod()

  return feriados_finais # <-- retornamos DataFrame final de feriados, com somente a data, região e impacto final em cada etapa do funil



#@title Definição da Quebra Diária DataFrame 1.3


'''
Essa função serve para quebrar as metas coincident semanais em metas diárias, além de aplicar efeito
de feriados em dias específicos
'''

def quebra_diaria(volumes_semanais,  # DataFrame com os volumes semanais coincident
                  coluna_de_semanas,
                  aberturas_das_bases,
                  share_diario,      # DataFrame com os dados de share diário
                  feriados,          # DataFrame com a lista de feriados
                  impactos_feriados, # DataFrame com os impactos de feriado
                  share_cidades,     # DataFrame com o share das cidades dentro das regiões
                  topo_de_funil,     # Lista com os nomes das colunas que representam as etapas de topo de funil
                  base_diaria_ToF,   # Dataframe importado no início do código
                 round_output = False):    
  

  # Definições iniciais
  #_________________________________________________________________________________________________

  # definindo quais são as colunas de BB presentes:
  building_blocks = [x for x in list(volumes_semanais.columns.values) if x in ['building block cohort','building block tof']]

  # definindo quais são as colunas de valores na base de volumes
  etapas_volume = list(set(list(volumes_semanais.columns.values))-set([coluna_de_semanas]+aberturas_das_bases+building_blocks))

  # Vamos ordenar as etapas da forma como aparecem na base
  aux = [x for x in list(volumes_semanais.columns.values) if x in etapas_volume]
  etapas_volume = aux

  # definindo as colunas de valores auxilieares para ajudar nos cálculos futuros
  etapas_volume_x = [x+'_x' for x in etapas_volume]
  etapas_volume_y = [x+'_y' for x in etapas_volume]

  # Com a lista de feriados, base de impactos e share de cidades, construímos um DataFrame
  # que contém as datas, regiões e impactos de cada feriado nas diferentes etapas de volume do funil.
  # Essa trasformação é feita com a função auxiliar "base_impacto_feriados"
  base_feriados = base_impacto_feriados(feriados,impactos_feriados,share_cidades) # @função_auxiliar
  #-------------------------------------------------------------------------------------------------


  # Vamos criar uma base semanal com o share relativo de cada building block, pois vamos realizar a quebra diária
  # dos building blocks agrupados, já que eles não influenciam na quebra. Depois, vamos redistribuar a base
  # diária nos building blocks de acordo com o share original de cada um:
  volumes_semanais_agrupados = volumes_semanais.groupby([coluna_de_semanas]+aberturas_das_bases, as_index=False)[etapas_volume].sum()
  
  volumes_proporcao_bb = pd.merge(volumes_semanais,volumes_semanais_agrupados,how='left',on=[coluna_de_semanas]+aberturas_das_bases)
  volumes_proporcao_bb[etapas_volume] = volumes_proporcao_bb[etapas_volume_x].astype(float).values / (volumes_proporcao_bb[etapas_volume_y].astype(float).values + 0.00000000001)
  volumes_proporcao_bb = volumes_proporcao_bb.drop(columns=etapas_volume_x+etapas_volume_y)
  volumes_proporcao_bb = volumes_proporcao_bb.rename(columns={coluna_de_semanas:'semana'})

  # Geramos uma primeira versão da base diária através de uma função auxiliar que simplesmente repete
  # os volumes semanais 7 vezes (1 pra cada dia da semana) e já formata com todas as colunas novas
  # (Ano, Mês, Semana, Dia etc)
  base_diaria = gerador_base_diaria(volumes_semanais_agrupados,coluna_de_semanas) # @função_auxiliar
  #-------------------------------------------------------------------------------------------------


  # Aqui identificamos todas as colunas que não contém dados numéricos. Ou seja: são as aberturas
  # completas das bases, que vamos chamar de 'chaves'
  cabecalho_share_diario = list(share_diario.columns.values)
  chaves_share_diario = cabecalho_share_diario[:cabecalho_share_diario.index(topo_de_funil[0])]
  chaves_volumes_semanais = list(volumes_semanais_agrupados.columns.values)
  chaves_volumes_semanais = [coluna_de_semanas]+aberturas_das_bases

  # Definimos os nomes das colunas que irão conter as etapas do funil
  etapas = etapas_volume

  #-------------------------------------------------------------------------------------------------


  # Abaixo definimos os nomes das colunas que irão conter as etapas do funil ao longo das transformações.
  # Como vamos unir bases com mesmas etapas mas agregações distintas, os nomes das colunas repetidos
  # ficam com + "_x" para a primeira ocorrência, + "_y" para a segunda e assim por diante (pandas faz
  # isso automaticamente). Assim, já identificamos as mudanças dos nomes das colunas para ao longo
  # das trasformações referenciar as etapas corretamente. Após cada transformação, anotamos abaixo como
  # ficaram os nomes das etapas

  etapas_semanal = etapas.copy()
  etapas_share = etapas.copy()
  etapas_impactos = etapas.copy()
  etapas_share_atual= etapas.copy()
  etapas_semanal_atual = etapas.copy()
  etapas_diff = etapas.copy()
  etapas_semanal_x = etapas.copy()
  etapas_diario_corrigido = etapas.copy()
  etapas_subs_metas_antigas = etapas.copy()
  etapas_subs_metas_antigas_x = etapas.copy()
  etapas_subs_metas_antigas_y = etapas.copy()
  etapas_auxiliar = etapas.copy()
  etapas_auxiliar_x = etapas.copy()
  etapas_auxiliar_y = etapas.copy()
  etapas_corrigidos_diarios_agg = etapas.copy()
  etapas_corrigidos_diarios = etapas.copy()
  for i in range(len(etapas)):
    etapas_semanal[i] = etapas_semanal[i]+"_x"
    etapas_semanal_x[i] = etapas_semanal_x[i]+"_x_x"
    etapas_share[i] = etapas_share[i]+"_y"
    etapas_impactos[i] = etapas_impactos[i]+"_z"
    etapas_share_atual[i] = etapas_share_atual[i]+"_s"
    etapas_semanal_atual[i] = etapas_semanal_atual[i]+"_y_y"
    etapas_diff[i] = etapas_diff[i]+"_d"
    etapas_diario_corrigido[i] = etapas_diario_corrigido[i]+"_c"
    etapas_subs_metas_antigas[i] = etapas_subs_metas_antigas[i]+"_a"
    etapas_subs_metas_antigas_x[i] = etapas_subs_metas_antigas_x[i]+"_a_x"
    etapas_subs_metas_antigas_y[i] = etapas_subs_metas_antigas_y[i]+"_a_y"
    etapas_auxiliar[i] = etapas_auxiliar[i]+"_i"
    etapas_auxiliar_x[i] = etapas_auxiliar_x[i]+"_i_x"
    etapas_auxiliar_y[i] = etapas_auxiliar_y[i]+"_i_y"
    etapas_corrigidos_diarios_agg[i] = etapas_corrigidos_diarios_agg[i]+"_c_y"
    etapas_corrigidos_diarios[i] = etapas_corrigidos_diarios[i]+'_c_x'
  #-------------------------------------------------------------------------------------------------

  # Definimos as chaves da base que será apenas o agregado semanal da base diária
  chaves_agg_semanais = ['semana']+aberturas_das_bases
  chaves_finais = chaves_agg_semanais.copy() # guardamos o cabeçalho diário final




  # Cálculos e transformações da base diária
  #_________________________________________________________________________________________________

  # Primeiro damos 'left join' na base diária (que por enquanto só repetiu os volumes semanais) com a base
  # de impacto de feriados. Nas chaves onde não há impacto de feriado preenchemos com 1
  base_diaria['data'] = pd.to_datetime(base_diaria['data'], errors='coerce')
  if 'região' in base_diaria.columns.values:
    base_diaria = pd.merge(base_diaria,base_feriados,how='left',on=['data','região']).fillna(1)
  elif 'city_group' in base_diaria.columns.values:
    base_diaria = pd.merge(base_diaria,base_feriados,how='left',left_on=['data','city_group'],right_on=['data','região']).fillna(1)
  else:
    base_diaria = pd.merge(base_diaria,base_feriados,how='left',on=['data']).fillna(1)
    values = list(set(base_diaria.columns.values) - set(aberturas_das_bases + ['semana','data','dia da semana','mês', 'ano','região']))
    base_diaria = base_diaria.groupby(aberturas_das_bases + ['semana','data','dia da semana','mês', 'ano'],as_index=False)[values].mean()

    
  #base_diaria[etapas_impactos] = base_diaria[etapas_impactos].fillna(1)
  #base_diaria.dropna(axis=0, how='any', inplace=True)

  # etapas do volume semanal = _ (etapas)
  # etapas dos impactos = _z (etapas_impactos)

  # Depois damos 'left join' na base diária+impactos com a base de share diário. Caso alguma chave
  # não tenha share, substituímos por 1/7
  base_diaria = pd.merge(base_diaria,share_diario,how='left',on=chaves_share_diario).fillna(1/7)
  # etapas do volume semanal = _x (etapas_semanal)
  # etapas dos impactos = _z (etapas_impactos)
  # etapas do share = _y (etapas_share)
  #-------------------------------------------------------------------------------------------------


  # Aqui realizamos o primeiro cálculo: multiplicamos o impacto de feriado pelo share diário e pelo
  # volume semanal que está repetido na base diária. O nome das etapas realmente diarizadas é definido
  # pela lista "etapas"

  base_diaria[etapas] = base_diaria[etapas_impactos].multiply(\
                        base_diaria[etapas_share].astype('float').values, axis="index").multiply(\
                        base_diaria[etapas_semanal].astype('float').values, axis="index")
  # etapas do volume semanal = _x (etapas_semanal)
  # etapas dos impactos = _z (etapas_impactos)
  # etapas do share = _y (etapas_share)
  # etapas do volume diário = (etapas)




  # Cálculo da correção do efeito do feriado no agregado semanal
  #-------------------------------------------------------------------------------------------------

  # Como o impacto de feriado removeu a propriedade da soma do share diário dar sempre 100% em qualquer
  # semana, temos que agregar o valor semanal resultante da primeira quebra diária e verificar as
  # diferenças com a base semanal original.

  # Abaixo definimos a base semanal nova (com os valores "errados") agregando a base diária
  # pelas 'chaves_agg_semanais' e somando nas semanas as 'etapas'
  base_semanal_nova = base_diaria[list(chaves_agg_semanais+etapas)]
  base_semanal_nova = base_semanal_nova.groupby(chaves_agg_semanais, as_index=False)[etapas].sum()
  # etapas do volume semanal novo = _ (etapas)
  #print(base_semanal_nova.loc[(base_semanal_nova['Data'] == '2021-03-29') & (base_semanal_nova['Região'] == 'Belo Horizonte') & (base_semanal_nova['Operação'] == 'B2B')][['Fluxo','Guarantee','VB']])

  # Trocamos o nome da coluna e das chaves para unir corretamente a base semanal nova com a base de volumes original
  base_semanal_nova = base_semanal_nova.rename(columns={"semana": coluna_de_semanas}) # trocamos o nome da coluna
  chaves_agg_semanais.remove('semana')
  chaves_agg_semanais = chaves_agg_semanais+[coluna_de_semanas]

  # Fazemos um 'left join' da base de volumes semanais originais com a base de volumes semanais nova,
  # com o objetivo de comparar os valores linha a linha
  merge_semanas = pd.merge(volumes_semanais_agrupados,base_semanal_nova,how='left',on=chaves_agg_semanais)
  # Data = Semana
  # etapas do volume semanal = _x (etapas_semanal)
  # etapas do volume semanal novo = _y (etapas_share)

  # Calculamos a diferença entre os volumes de todas as etapas, linha a linha, entre
  # a base semanal nova e a original
  merge_semanas[etapas_diff] = merge_semanas[etapas_semanal].astype('float').values - merge_semanas[etapas_share].astype('float').values
  # Data = Semana
  # etapas do volume semanal = _x (etapas_semanal)
  # etapas do volume semanal novo = _y (etapas_share)
  # etapas da diferença = _d (etapas_diff)

  # Trocamos novamente o nome da coluna e das chaves para unir corretamente a base diária nova com a base das diferenças semanais
  chaves_agg_semanais.remove(coluna_de_semanas)
  chaves_agg_semanais = chaves_agg_semanais+['semana']
  merge_semanas = merge_semanas.rename(columns={coluna_de_semanas: "semana"})

  # Fazemos um 'left join' da base diária com a base semanal que contém a diferença de volumes entre as semanais originais e as novas
  # Chamamos essa base unida de 'share_diario_atual' pois nela vamos calcular o share diário dos
  # volumes diários em relação ao volume semanal atual
  share_diario_atual = pd.merge(base_diaria,merge_semanas,how='left',on=chaves_agg_semanais)[chaves_finais+etapas+etapas_diff+etapas_semanal_atual+etapas_semanal_x]
  # etapas do volume diário = _ (etapas)
  # etapas do volume semanal atual = _y_y (etapas_semanal_atual)
  # etapas do volume semanal original = _x_x (etapas_semanal_x)
  # etapas da diferença = _d (etapas_diff)

  # Calculamos o share diário das metas em relação ao agregado semanal.
  # Somamos um número pequeno nos volumes semanais para evitar div/0.
  # Esse share diário atual é diferente do share diário original, pois contém
  # a informação dos impactos de feriado embutido nele
  share_diario_atual[etapas_share_atual] = share_diario_atual[etapas].multiply(1./(share_diario_atual[etapas_semanal_atual].astype('float').values + 0.00000001), axis="index")
  # etapas do volume diário = _ (etapas)
  # etapas do volume semanal novo = _y_y (etapas_semanal_atual)
  # etapas da diferença = _d (etapas_diff)
  # etapas do share diario atual = _s (etapas_share_atual)

  # Obtemos os volumes diários corrigidos multiplicando o share diário atual pelos volumes semanais
  # originais. Assim, os volumes diários contemplam os efeitos de feriado e, ao mesmo tempo, se somados,
  # retornam os volumes semanais originais.
  share_diario_atual[etapas_diario_corrigido] = share_diario_atual[etapas_semanal_x].astype('float').multiply(share_diario_atual[etapas_share_atual].astype('float').values, axis="index")
                    
  # Substituímos os volumes diários calculados pela primeira vez pelos volumes diários corrigidos na
  # base diária final
  base_diaria[etapas] = share_diario_atual[etapas_diario_corrigido]
  #print(base_diaria.loc[(base_diaria['Data'] == '2021-04-21') & (base_diaria['Região'] == 'RMSP') & (base_diaria['Operação'] == 'Core') & (base_diaria['Guarantee'] == 'FALSE') & (base_diaria['Fluxo'] == 'VISIT')][['VB','VC','OS','OA','DS','CA','CS']])

  #-------------------------------------------------------------------------------------------------


  # Limpamos as colunas auxiliares da base diária final
  categorias = aberturas_das_bases
  chaves_finais = ['ano','mês', 'dia da semana','data','semana']+aberturas_das_bases+etapas
  base_diaria = base_diaria[chaves_finais]

  # Formatamos as datas da base final
  #base_diaria['data'] = base_diaria['data'].apply(lambda x: x.strftime('%m/%d/%Y'))
  #base_diaria['semana'] = base_diaria['semana'].apply(lambda x: x.strftime('%m/%d/%Y'))




  # Redistribuir a base diária pelo share semanal dos Building Blocks:
  #-------------------------------------------------------------------------------------------------

  # Primeiro, precisamos criar as colunas com os BB's na base diária para poder efetuar o merge corretamente
  for bb in building_blocks:
    aux = base_diaria.copy()
    lista_bb = list(np.unique(volumes_semanais[bb].values))
    for i in range(len(lista_bb)):
      if i == 0:
        base_diaria[bb] = lista_bb[i]
      else:
        aux[bb] = lista_bb[i]
        base_diaria = pd.concat([base_diaria,aux])

  # Vamos unir a base diária com a base que contém os shares semanais dos BB's:
  base_diaria = pd.merge(base_diaria,volumes_proporcao_bb,how='left',on=['semana']+building_blocks+aberturas_das_bases)

  # Definindo os valores diários como sendo os valores diários multiplicados pelo share dos BB's
  base_diaria[etapas] = base_diaria[etapas_volume_x].astype(float).values * base_diaria[etapas_volume_y].astype(float).values

  # Limpando as colunas desnecessárias
  base_diaria = base_diaria[['ano','mês', 'dia da semana','data','semana']+building_blocks+aberturas_das_bases+etapas]




  # Limpando dados zerados na base final (para economizar o tamanho da base)
  # Primeiro verificamos se as etapas são somente as etapas de topo de funil.
  # Se esse for o caso, significa que estamos realizando a diarização somente do ToF.
  # Como essa diarização no fim vai redefinir a base semanal, não podemos remover valores zerados,
  # pois precisamos manter todas as aberturas.
  if etapas != topo_de_funil:
    cabecalho = list(base_diaria.columns.values)
    base_diaria['check'] = base_diaria[etapas].astype('float').sum(axis=1)
    base_diaria = base_diaria[(base_diaria['check'] != 0)][cabecalho]

  
  # Substituir dados de ToF diariazados pelos que mantém o valor mensal, se existirem
  #-------------------------------------------------------------------------------------------------

  # Caso tenha sido feita uma pré diarização do ToF para bater valores mensais, devemos
  # substituir o ToF diarizado da base diária atual pelo ToF diarizado inicialmente
  if (len(base_diaria_ToF) > 0):

    base_diaria_ToF['building block cohort'] = 'baseline'


    # Vamos redefinir as categorias como sendo as aberturas das bases mais as colunas de BB's:
    categorias = building_blocks+aberturas_das_bases
    categorias_completas = ['ano','mês', 'dia da semana','data','semana']+building_blocks+aberturas_das_bases


    # Formatando as datas das bases para facilitar os merges
    base_diaria_ToF['semana'] = pd.to_datetime(base_diaria_ToF['semana'], errors='coerce')
    base_diaria_ToF['semana'] = base_diaria_ToF['semana'].dt.date

    base_diaria['semana'] = pd.to_datetime(base_diaria['semana'], errors='coerce')
    base_diaria['semana'] = base_diaria['semana'].dt.date


    # Transformamos em float os valores numéricos da base diária ToF
    base_diaria_ToF[topo_de_funil] = base_diaria_ToF[topo_de_funil].astype('float')

    # Criamos listas com os novos nomes de colunas de valores
    topos_de_funil_x = topo_de_funil.copy()
    topos_de_funil_y = topo_de_funil.copy()
    topos_de_funil_diff = topo_de_funil.copy()
    for i in range(len(topo_de_funil)):
      topos_de_funil_x[i] = topos_de_funil_x[i]+'_x'
      topos_de_funil_y[i] = topos_de_funil_y[i]+'_y'
      topos_de_funil_diff[i] = topos_de_funil_diff[i]+'_d'
    
    # Definimos as bases gerais (para identificar quais aberturas possuem valor na base ToF e na base diária atual)
    # Fazemos isso agrupando todos os valores de ToF nas categorias. Assim podemos identificar quais aberturas
    # novas receberam valores que não existiam no ToF inicial e precisam ser mantidos na base diária final
    base_geral_ToF = base_diaria_ToF.groupby(categorias, as_index=False)[topo_de_funil].sum()
    base_geral = base_diaria.groupby(categorias, as_index=False)[topo_de_funil].sum()

    # Nas bases gerais, as aberturas que possuem valores recebem o valor '1', e o restante permanece zerado,
    # pois o valor absoluto da soma dessas aberturas não é relevante. 
    base_geral_ToF[topo_de_funil] = (base_geral_ToF[topo_de_funil] != 0).astype(int)
    base_geral[topo_de_funil] = (base_geral[topo_de_funil] != 0).astype(int)

    # Unimos as bases gerais para determinar quais aberturas atuais não tem valores na base diária ToF inicial
    base_geral_merged = pd.merge(base_geral,base_geral_ToF,how='left',on=categorias)
    base_geral_merged = base_geral_merged.fillna(0)

    # Definimos colunas auxiliares que contém a informação de quais aberturas possuem valores na base
    # diária atual mas não existem na base diária ToF inicial. Como todos os valores existentes são '1',
    # uma coluna contendo a diferença entre os valores poderá ter somente 3 valores distintos:
    # 0 se a abertura continha valores nas duas bases
    # 1 se a abertura só passou a conter valores na base atual
    # -1 se a abertura só existia na base ToF inicial (este caso não deveria existir)
    base_geral_merged[topos_de_funil_diff] = base_geral_merged[topos_de_funil_x].values - base_geral_merged[topos_de_funil_y].values
    base_geral_merged = base_geral_merged[categorias+topos_de_funil_diff]

    # Unimos a base geral merged com a base diária
    base_diaria['data'] = pd.to_datetime(base_diaria['data'], errors='coerce')
    base_diaria_geral = pd.merge(base_diaria,base_diaria_ToF,how='outer',on=categorias_completas)
    base_diaria_geral = pd.merge(base_diaria_geral,base_geral_merged,how='left',on=categorias)
    base_diaria_geral = base_diaria_geral.fillna(0)

    # Agora, para cada etapa do ToF, vamos selecionar os valores diários da base diária atual
    # ou da base diária de ToF inicial com base na coluna auxiliar 'topos_de_funil_diff',
    # que contém a informação se aquela abertura existe ou não na base diária ToF inicial.

    # Criamos a base que irá conter os valores finais
    base_diaria_final = pd.DataFrame(columns=categorias_completas+topo_de_funil)
    # Vamos definir a lista de etapas restantes, que são aquelas que não sejam ToF
    restante = etapas.copy()
    for i in range(len(topos_de_funil_diff)):

      # Removemos das etapas restantes a etapa de ToF
      restante.remove(topo_de_funil[i])

      # Definimos inicialmente as bases diárias ToF e atual como cópias da base geral
      base_diaria_ToF = base_diaria_geral.copy()
      base_diaria_atual = base_diaria_geral.copy()

      # Agora selecionamos os dados da base ToF onde eles existem ou não na base diária atual
      base_diaria_ToF = base_diaria_ToF.loc[base_diaria_geral[topos_de_funil_diff[i]] <= 0]
      # Agora selecionamos os dados da base atual onde eles não existem na base diária ToF
      base_diaria_atual = base_diaria_atual.loc[base_diaria_geral[topos_de_funil_diff[i]] > 0]

      # Somente renomeamos a etapa para que o nome dela fique igual nas duas bases
      base_diaria_ToF[topo_de_funil[i]] = base_diaria_ToF[topos_de_funil_y[i]].values
      base_diaria_atual[topo_de_funil[i]] = base_diaria_atual[topos_de_funil_x[i]].values

      # Removemos todas as colunas auxiliares das bases
      base_diaria_ToF = base_diaria_ToF[categorias_completas+[topo_de_funil[i]]]
      base_diaria_atual = base_diaria_atual[categorias_completas+[topo_de_funil[i]]]

      # Adicionamos, na forma de 'append', as colunas de ToF da base atual e da base de ToF na base
      # final. Essa forma de adicionar as bases repete os índices quando as colunas não possuem o mesmo
      # nome. Mas, dessa forma, garantimos que não estamos excluíndo nenhum dado.
      base_diaria_final = base_diaria_final.append(base_diaria_atual, ignore_index=True, sort=False)
      base_diaria_final = base_diaria_final.append(base_diaria_ToF, ignore_index=True, sort=False)

    # Como a base_diaria_final foi criada adicionando as colunas de ToF,
    # é provável que as chaves estjam repetidas. Vamos reagrupa-las
    base_diaria_final = base_diaria_final.fillna(0)

    base_diaria_final['semana'] = pd.to_datetime(base_diaria_final['semana'], errors='coerce')
    base_diaria_final['semana'] = base_diaria_final['semana'].dt.date
    base_diaria_final['data'] = pd.to_datetime(base_diaria_final['data'], errors='coerce')
    base_diaria_final['data'] = base_diaria_final['data'].dt.date
    
    base_diaria_final = base_diaria_final.groupby(categorias_completas, as_index=False)[topo_de_funil].sum()

    # Unimos a base ToF com o restante da base
    base_restante = base_diaria[categorias_completas+restante]
    
    base_restante['semana'] = pd.to_datetime(base_restante['semana'], errors='coerce')
    base_restante['semana'] = base_restante['semana'].dt.date
    base_restante['data'] = pd.to_datetime(base_restante['data'], errors='coerce')
    base_restante['data'] = base_restante['data'].dt.date
    
    base_diaria = pd.merge(base_diaria_final,base_restante,how='outer',on=categorias_completas)
    base_diaria = base_diaria.fillna(0)

    # Reordenamos a base
    base_diaria = base_diaria.sort_values(by=['ano','mês', 'dia da semana','data','semana']+building_blocks+aberturas_das_bases+etapas)
    base_diaria = base_diaria[['ano','mês', 'dia da semana','data','semana']+building_blocks+aberturas_das_bases+etapas]

    # Arrumar formato da data no final. De '12/27/2021 0:00:00' Para '12/27/2021'
    base_diaria['semana'] = pd.to_datetime(base_diaria['semana'], errors='coerce')
    base_diaria['semana'] = base_diaria['semana'].dt.date

    # Remover linhas completamente zeradas:
    base_diaria = redutor_de_base(df = base_diaria,
                                  col_valores = etapas)

    
  
  # Fim dos cálculos
  #_________________________________________________________________________________________________

  # Arredondar output final
  if round_output:
    base_diaria = rounding_tool(df = base_diaria,
                                aberturas = ['ano','mês', 'dia da semana','data','semana']+building_blocks+aberturas_das_bases,
                                col_valores = etapas,
                                ordem_hirarquica = ['data']+aberturas_das_bases)

  return base_diaria
