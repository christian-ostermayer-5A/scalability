#@title Def separa_conv

# Importando funções auxiliares para rodar o funil:
from dynamic_cohort_funnel_functions import * 
import pandas as pd
import numpy as np

# Separa a primeira etapa "A" de uma lista de conversoes "A2B"

# Dada uma lista de conversões cohort, retorna uma lista contendo a primeira etapa de cada conversão.
# Serve para extrair o nome das etapas de volume de uma lista de conversões cohort de um funil

def separa_conv(lista_conv): # --> lista (lista de strings ['VB2VC', 'VC2OS', 'OS2OA'])
  out = lista_conv.copy()
  for i in range(len(lista_conv)):
    out[i] = lista_conv[i].split("2")[0]
  return out # <-- lista (lista de strings ['VB', 'VC', 'OS'])
  
 
 
#---------------------------------------------------------------------------------------------------

# Adiciona datas na base repetindo volumes e conversoes para o passado sem dados

# Serve para adicionar dados repetidos nas bases de ToF e conversão.
# Fazemos isso para garantir que a primeira semana vai receber volumes de cohorts
# antigas que nao estavam na base. Para garantir que estamos considerando um passado
# grande o suficiente, repetimos os dados da primeira semana (baseline) um número de vezes
# igual à maior cohort da base. 
 
def add_datas_passado(base,      # DataFrame cohort ou ToF
                      col_data,
                      intervalo, # interio que define o intervalo da cohort em dias (no caso semanal = 7 dias)
                      data_min,  # datetime (primiera data da base = baseline)
                      max_conv): # max_conv (inteiro que define qual é a cohort máxima (no caso = 5)

  for i in range(max_conv):
    copy = base.loc[base[col_data]==data_min]
    copy[col_data] = copy[col_data].apply(lambda x: x-pd.Timedelta((i+1)*intervalo, unit='D'))
    copy[col_data] = copy[col_data].apply(lambda x: pd.Timestamp(x, freq=None))
    copy[col_data] = copy[col_data].apply(lambda x: x.normalize())
    base = pd.concat([copy,base])
  return base # <-- base (DataFrame igual ao input, mas extendido para o passado)
  
  
  
#-----------------------------------------------------------------------------------------------

#@title Def base_geral

# Criar DF Merged com conversões, volumes e split

# Essa base geral é o join, através de todas as aberturas, das bases de ToF, cohort e split.
# Como todas essas base possuem as mesmas aberturas (a base cohort adquire coluna de datas depois de
# aplicarmos inputs), facilitamos as contas criando uma base só, onde podemos fazer operações com colunas
# inteiras de uma vez, ao invés de ficar procurando a abertura correta entre as bases.


def base_geral(base_cohort,  # DataFrame das conversões cohort
               base_ToF,     # DataFrame com os volumes de ToF
               nome_coluna_week_origin,
               aplicacao_ajuste,
               coluna_de_semanas):  

  # identificamos os cabecalhos como listas
  cb_cohort = list(base_cohort.columns.values)

  cb_ToF = list(base_ToF.columns.values)

  # definimos onde começam os dados de todas as bases com base na posição da coluna "Week Origin"
  # na base cohort
  posi_coh = cb_cohort.index(nome_coluna_week_origin)+1
  posi_ToF = posi_coh-1
  posi_split = posi_ToF
  
  # definimos as etapas cohort e de volume com base na posição de onde começam os dados
  etapas_ToF = cb_ToF[posi_ToF:]
  etapas_coh = cb_cohort[posi_coh:]

  # definimos as chaves das base (data, região, etc...)
  chaves_ToF = cb_ToF[:posi_ToF]

  # redefinimos as chaves das base (data, região, etc...) também com base onde começam os dados
  chaves_coh = cb_cohort[:posi_coh]
  chaves_coh_sem_origin = cb_cohort[:posi_coh-1]
  chaves_ToF = cb_ToF[:posi_ToF]

  # unimos a base cohort e a base de ToF com base nessas chaves
  merged = pd.merge(base_cohort,base_ToF,how='left',on=chaves_ToF)
  #print(chaves_ToF)
  #print(base_ToF.loc[(base_ToF['Região']=='RMSP') & (base_ToF['Lead']=='FSS') & (base_ToF['Canal']=='Indica Aí - General')][['Data','OP']])
  #print(base_cohort.loc[(base_cohort['Região']=='RMSP') & (base_cohort['Lead']=='FSS') & (base_cohort['Canal']=='Indica Aí - General')][['Data','OP2Q']])
  #print(merged.loc[(merged['Região']=='RMSP') & (merged['Lead']=='FSS') & (merged['Canal']=='Indica Aí - General')][['Data','OP2Q','OP']])
  
  # formatamos a coluna de datas
  merged[coluna_de_semanas] = pd.to_datetime(merged[coluna_de_semanas], infer_datetime_format=True)

  # encontramos qual é a maior cohort
  max_origin = merged[merged[nome_coluna_week_origin] != "Coincident"][nome_coluna_week_origin].astype('int').max()

  # encontramos algumas definições sobre as datas que serão úteis em outras funções
  datas_unicas = merged[coluna_de_semanas].unique()
  datas_unicas_list = list(datas_unicas)
  data_min = datas_unicas.min()
  data_max = datas_unicas.max()
  qtd_datas = datas_unicas.size
  intervalo = ((data_max-data_min)/ np.timedelta64(1,"D")) / (qtd_datas-1)

  # Com a base total gerada, repetimos a primeira semana adicionando datas para o passado
  # para executar a cohort de forma correta.
  # Utilizamos uma função auxiliar que repete os dados da primeira semana um número de vezes igual à maior
  # cohort e gera as datas passadas corretas
  merged = add_datas_passado(merged,
                             coluna_de_semanas,
                             intervalo,
                             data_min,
                             max_origin) # @função_auxiliar

  # Utilizamos uma função auxiliar para gerar uma coluna de datas deslocadas. Essas datas deslocadas
  # servirão para somar a "diagonal" da cohort e obter os volumes coincident.
  try:
    aplicacao_ajuste = int(aplicacao_ajuste)
  except:
    aplicacao_ajuste = int(max_origin)
  if int(aplicacao_ajuste) > int(max_origin):
    aplicacao_ajuste = int(max_origin)

  merged['shifted'] = shift_datas(merged[coluna_de_semanas],merged[nome_coluna_week_origin],intervalo,aplicacao_ajuste) # @função_auxiliar

  # <-- retornamos a base final e todas as outras informações que geramos de início
  return merged,etapas_coh,etapas_ToF,chaves_coh,chaves_ToF,max_origin,datas_unicas_list,data_min

#---------------------------------------------------------------------------------------------------





#@title Def on_top_ratios

# Soma um volume adicional numa etapa anterior, que não vem de conversões cohort, 
# com base num share de um volume posterior.

def on_top_ratios(base,         # DataFrame com os volumes sendo gerados
                  on_top,       # DataFrame com a base de shares de volumes "on top"
                  chaves_ToF):  # Categorias da base de volumes

  # removemos a coluna de datas da lista de categorias, pois a base de ratios não possui data
  chaves_sem_data = chaves_ToF[1:]

  # definimos os cabeçalhos das bases e separamos as etapas que possuem ratio on top
  cb_ontop = list(on_top.columns.values)
  cb_base = list(base.columns.values)
  etapas_ontop = cb_ontop[len(chaves_sem_data):]

  # unimos a base de volumes com a base que possui os ratios
  base_com_ontop = pd.merge(base,on_top,how='left',on=chaves_sem_data)

  # para cada etapa da base de ratios, separamos qual é a etapa que vai ser multiplicada
  # por um ratio (etapa_1) e qual vai ser acrescida do volume resultante (etapa_2)
  for i in range(len(etapas_ontop)):
    etapa_1 = etapas_ontop[i].split("/")[0]
    etapa_2 = etapas_ontop[i].split("/")[1]

    # o volume adicional é definido multiplicando a coluna com a etapa_1 pela coluna que contem os ratios
    vol_add = base[etapa_1].values*base_com_ontop[etapas_ontop[i]].values.astype('float')

    # caso a etapa que vai ser acrescida de um volume ainda não existe na base, criamos a coluna com
    # essa etapa (exemplo do ES e DS)
    if etapa_2 not in cb_base:
      base[etapa_2] = vol_add
    # caso exista, somamos o volume do ratio na etapa
    else:
      base[etapa_2] = base[etapa_2].astype('float').add(vol_add, fill_value=0)

  return base # <-- retorna o mesmo DataFrame, mas acrescido ou não do volume vindo do ratio on top
#---------------------------------------------------------------------------------------------------




#@title Def funil_dinamico

# Essa função executa a função de progerssão do funil num loop sucessivo até acabarem as etapas

def funil_dinamico(base_merged, # DataFrame total gerado pela função auxiliar "def base_geral"
                   topo,        # string que define qual é a primeira etapa de volume do funil
                   max_origin,  # inteiro representando a cohort máxima, gerado pela função auxiliar "def base_geral"
                   etapas_coh,  # lista com as etapas cohort, gerado pela função auxiliar "def base_geral"
                   chaves_coh,  # lista com as aberturas cohort gerado pela função auxiliar "def base_geral"
                   chaves_ToF,  # lista com as aberturas ToF gerado pela função auxiliar "def base_geral"
                   data_min,    # datetime da primeira data da base gerado pela função auxiliar "def base_geral"
                   nome_coluna_week_origin,
                   coluna_de_semanas):   
  
  # definimos as etapas de volume com base nas etapas de conversão.
  # Utilizamos uma função auxiliar que retorna uma lista com as etapas de volume.
  etapas_vol = separa_conv(etapas_coh) # @função_auxiliar
  # definimos as etapas que serão calculadas como sendo todas as que ficam à direita da etapa do topo
  posi_topo = etapas_vol.index(topo) 
  etapas_coh = etapas_coh[posi_topo:]
  etapas_vol = etapas_vol[posi_topo:]
  
  # para cada etapa de conversão que identificamos, vamos calcular os volumes das cohorts e o volume
  # coincident da etapa seguinte utilizando a função auxiliar de progerssão do funil
  for e in range(len(etapas_coh)):

    # calculamos aqui através da função auxiliar 
    base_merged = progressao_funil(base_merged,
                                   etapas_coh[e],
                                   chaves_coh,
                                   chaves_ToF,
                                   max_origin,
                                   nome_coluna_week_origin,
                                   coluna_de_semanas)
    # Aqui vamos repetir o volume da etapa nova calculado na semana inicial e repeti-lo nas semanas
    # do passado. Se nao fizermos isso, vamos propagar volumes menores na próxima conversão, pois
    # somente repetir os volumes e conversoes do topo um número de vezes igual à conversao máxima no
    # passado não é o suficiente para garantir a repetição igual dos volumes das etapas seguintes. 
    nova_etapa = etapas_coh[e].split("2")[1]
    repeat = base_merged.loc[base_merged[coluna_de_semanas] == data_min][nova_etapa]
    for d in range(max_origin):
      data = data_min - pd.Timedelta((d+1)*7, unit='D')
      base_merged.loc[base_merged[coluna_de_semanas] == data,nova_etapa] = repeat.values

  return base_merged
#---------------------------------------------------------------------------------------------------




#@title Def formatacao

# Retorna a base final no formato de base coincident e base de volumes cohort
def formatacao(base,
               etapas_coh,
               etapas_vol,
               datas_unicas,
               nome_coluna_week_origin,
               coluna_de_semanas):

  # selecionar apensas as datas que importam.
  # não vamos mostrar os dados auxiliares criados em semanas anteriores à primeira semana de baseline
  base = base[base[coluna_de_semanas].isin(datas_unicas)]

  base= base.fillna(0.0)

  cb_base = list(base.columns.values)
  posi_origin = cb_base.index(nome_coluna_week_origin)

  base[coluna_de_semanas] = base[coluna_de_semanas].apply(lambda x: x.strftime('%Y/%m/%d'))

  etapas_vol_coh = etapas_coh.copy()
  for i in range(len(etapas_coh)):
    etapas_vol_coh[i] = "coh_vol_"+etapas_coh[i]


  cb_output_cohort = cb_base[:posi_origin+1]+etapas_vol_coh
  cb_output_cohort_final = cb_base[:posi_origin+1]+etapas_coh
  cb_output_coincident = cb_base[:posi_origin]+etapas_vol

  output_cohort = base[cb_output_cohort]
  output_coincident = base.loc[base[nome_coluna_week_origin] == '0'][cb_output_coincident]
                 
  # Removemos possíveis valores negativos gerados pelas cohorts de ajuste
  output_coincident = output_coincident.clip(lower=0)




  return output_cohort,output_coincident
#---------------------------------------------------------------------------------------------------



#@title Def formata_base_unica

def formata_base_unica(base_coincident,
                       base_cohort,
               nome_coluna_week_origin,
               aberturas,
               etapas_vol,
               etapas_coh,
               coluna_de_semanas):
  
  # Removendo etapas de volume que não estão na base cohort:
  etapas_vol_cohort = []
  for c in etapas_coh:
    etapas_vol_cohort = etapas_vol_cohort + c.split('2')

  etapas_vol_nova = []
  for e in etapas_vol:
    if e in etapas_vol_cohort:
      etapas_vol_nova = etapas_vol_nova + [e]


  # Definindo output único:

  cb_output_cohort = list(base_cohort.columns.values)
  cb_output_coincident = list(base_coincident.columns.values)

  etapas_vol_coh = ["coh_vol_"+i for i in etapas_coh]
  etapas_vol_nao_conv = ["nao_conv_"+i for i in etapas_coh]


  chaves = ['building block cohort', 'building block tof']+aberturas
  col_data = cb_output_cohort[0]

  output_vol_parcial = base_cohort.loc[base_cohort[nome_coluna_week_origin] != 'Coincident']
  output_vol_parcial = output_vol_parcial.groupby([col_data]+chaves, as_index=False)[etapas_vol_coh].sum()

  nao_convertido = pd.merge(base_coincident,output_vol_parcial,how='outer',on=[col_data]+chaves)
  nao_convertido = nao_convertido.fillna(0)

  nao_convertido[etapas_vol_nao_conv] = nao_convertido[etapas_vol_nova[:-1]].values - nao_convertido[etapas_vol_coh].values
  nao_convertido[nome_coluna_week_origin] = 'Não Convertido'
  nao_convertido = nao_convertido[[col_data]+chaves+[nome_coluna_week_origin]+etapas_vol_nao_conv+[etapas_vol[-1]]]
  nao_convertido = nao_convertido.rename(columns=dict(zip(etapas_vol_nao_conv, etapas_vol_coh)))

  output = pd.concat([base_cohort,nao_convertido])

  output = output[[col_data]+chaves+[nome_coluna_week_origin]+etapas_vol_coh+[etapas_vol_nova[-1]]]

  output = output.rename(columns=dict(zip(etapas_vol_coh, etapas_coh)))
  cb_final = list(output.columns.values)

  # Remove linhas zeradas:
  output = output.fillna(0)
  output['Soma'] = output[etapas_coh+[etapas_vol_nova[-1]]].sum(axis=1)
  output = output.loc[output['Soma'] != 0,cb_final]


  #output = output.groupby([col_data]+chaves+[nome_coluna_week_origin], as_index=False)[etapas_coh+[etapas_vol[-1]]].sum()

  return output
#---------------------------------------------------------------------------------------------------------



#@title Funil Dinâmico DataFrame 1.0

'''
Essa função aplica a base de conversões cohort nos volumes de ToF fornecido e calcula o restante
do funil
'''

def Funil_Dinamico_DataFrame(df_ToF,                # DataFrame contendo os volumes de ToF
                             df_cohort,             # DataFrame contendo as conversões cohort já com inputs aplicados e semanalizada
                             df_ratio,              # Matriz contedo a base com as razões de volume de são adicionadas por fora do funil
                             df_impacto_feriados,    # Matriz com os impactos de feriados. (os impactos não são aplicados aqui. Usamos essa base somente para ter a referência de quais são todas as etapas do funil)
                             nome_coluna_week_origin,
                             aplicacao_ajuste,
                             coluna_de_semanas): 
  '''
  print("Executando Funil Dinâmico DataFrame 1.0")
  print("_______________________________________")
  print("Definições Iniciais")
  '''
  # Os dados inputados na forma de matriz são transformados em DataFrames

  if len(df_ratio.index)>0:
    cb_ratio = list(df_ratio.columns)
    df_ontop = df_ratio.copy()
  else:
    df_ontop = pd.DataFrame(np.array(['sem_ratio','sem_ratio']), columns = ['sem_ratio'])


  # Definimos as etapas de volume do funil pelo cabeçalho da base de impacto de feriados.
  # A base de conversões cohort não possui, necessariamente, todas as etapas de volume que queremos calcular.
  # Alguns volumes são calculados exclusivamente via ratio on top e não existem nem na base de ToF nem
  # fazem parte de alguma conversão cohort. Usamos a base de impactos para definir em qual ordem se
  # encaixam as etapas de volume que não existem nas outras bases.
  etapas_volume = list(df_impacto_feriados.columns)[1:]

  #-------------------------------------------------------------------------------------------------


  #print("Unindo base de ToF e base de conversões")
  # Primeiro unimos as bases Cohort ,ToF e Split utilizando uma função auxiliar "base_geral".
  # Essa função, além de retornar uma base geral única, onde todas as conversões, volumes e ratios correspondem
  # nas datas e aberturas da base, também retorna muitas outras informações sobre o funil, que ficam armazenadas
  # na variável "merged".

  merged = base_geral(df_cohort,
                      df_ToF,
                      nome_coluna_week_origin,
                      aplicacao_ajuste,
                      coluna_de_semanas) # @função_auxiliar, retorna: --> merged, etapas_coh, etapas_ToF, etapas_split, chaves_coh, chaves_ToF, max_origin, datas_unicas_list, data_min


  # Uma das informações que a função "base_geral" retorna são as etapas de ToF.
  # Selecionamos a primeira para iniciar o cálculo do funil.
  topo_de_funil = merged[2][0]
  topos_de_funil = merged[2]
  #-------------------------------------------------------------------------------------------------

  #print("Calculando o funil a partir de ",topo_de_funil)
  # Com base nas informações geradas pela função "base_geral", podemos executar a função auxiliar
  # "funil_dinamico", que vai efetivamente calcular todas as etapas do funil, inclusive realizando
  # o split de fluxos (se existir).
  # A função retorna a base geral acrescida de colunas de volume coincident e volume de cohort de todas
  # as etapas
  # @função_auxiliar
  funil = funil_dinamico(base_merged = merged[0],  # DataFrame completo (ToF + Cohort + Split)
                      topo = topo_de_funil,        # String que define a etapa ToF pela qual começamos
                      max_origin = merged[5],      # Int que define a cohort de origem máxima da base
                      etapas_coh = merged[1],      # Lista contendo o nome das etapas de conversão
                      chaves_coh = merged[3],      # Lista contendo o nome das colunas com as aberturas da base cohort
                      chaves_ToF = merged[4],      # Lista contendo o nome das colunas com as aberturas da base ToF
                      data_min = merged[7],        # datetime com a primeira data da base
                      nome_coluna_week_origin = nome_coluna_week_origin,
                      coluna_de_semanas = coluna_de_semanas)        

  #-------------------------------------------------------------------------------------------------


  #print('Adicionando volumes via ratio on top')
  # Após calcular o funil completo, aplicamos outra função auxiliar para gerar os volumes que são
  # calculados via ratio on top e somá-los na base.
  if df_ontop.iloc[0,0] != 'sem_ratio':
    # @função_auxiliar
    funil = on_top_ratios(funil,     # DataFrame do funil completo
                          df_ontop,  # DataFrame com o ratio on top
                          merged[4]) # Lista contendo o nome das colunas com as aberturas da base ToF

  
  else:
    print(" --> Sem volumes adicionais")
  #-------------------------------------------------------------------------------------------------


  #print('Formatando bases finais')
  # Após calcular a base completa, formatamos as bases finais separando a base de volumes cohort dos
  # volumes coincident, utilizando uma função auxiliar
  # @função_auxiliar
  output_cohort,output_coincident = formatacao(funil,          # DataFrame do funil completo
                                               merged[1],      # Lista contendo o nome das etapas de conversão
                                               etapas_volume,  # Lista contendo o nome das etapas de volume coincident
                                               merged[6],      # Lista contendo as datas únicas da base
                                               nome_coluna_week_origin,
                                               coluna_de_semanas)     

  #-------------------------------------------------------------------------------------------------


  # Aqui removevos todas as linhas da base final que não contém nenhum dado
  cabecalho = list(output_cohort.columns.values)
  etapas_cohort = ['coh_vol_'+x for x in merged[1]]
  output_cohort['check'] = output_cohort[etapas_cohort].astype('float').sum(axis=1)
  output_cohort = output_cohort[(output_cohort['check'] != 0)][cabecalho]

  cabecalho = list(output_coincident.columns.values)
  output_coincident['check'] = output_coincident[etapas_volume].astype('float').sum(axis=1)
  output_coincident = output_coincident[(output_coincident['check'] != 0)][cabecalho]
  #-------------------------------------------------------------------------------------------------
  #print("_______________________________________")
  #print('Funil calculado')

  return output_cohort,output_coincident,topo_de_funil,topos_de_funil # <-- retornamos as bases coincident, cohort e a informação de qual foi a etapa de ToF utilizada
#---------------------------------------------------------------------------------------------------





