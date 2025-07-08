#@title Funções Auxiliares Funil Dinâmico
import numpy as np
import pandas as pd
'''
Nesta célula definimos inúmeras funções auxiliares que são usadas frequentemente em outras partes do código.

Em algumas situações a função vai ser executada em apenas uma posição do código, mas foi escolhido escrevê-la
e definí-la separadamente para facilitar a leitura do código que utiliza a função ou para facilitar
futuras edições que só irão alterar a função auxiliar.

Ao longo do código, toda vez que uma função auxiliar for executada, será comentado qual a sua função,
quais tipos de inputs ela recebe e qual output ela fornece, para facilitar a navegação.
'''

'''
Funções usadas em várias células de código:
____________________________________________________________________________________________________
'''
# Função auxiliar para gerar colunas auxiliares.

# Essa função é utilizada para otimizar a localização de posições nas bases e também auxiliar
# nas transformações das mesmas.
# Basicamente unimos várias colunas linha a linha, fomrando uma única coluna que contém todas as informações
# das colunas unidas na forma de strings.

def join_aux(dados,      # np.array de 2 dimensões contendo as colunas que serão unidas
             separador): # string definida que irá unir uma coluna na outra
  aux=np.transpose(np.array([np.copy(dados[:,0])])).astype(str)
  aux[:] = "*"
  dados_aux = np.append(dados.astype(str),aux,1)
  dados_aux = np.concatenate(dados_aux)
  dados_aux = separador.join(dados_aux)
  dados_aux = dados_aux.split(separador+"*"+separador)
  dados_aux[-1]=dados_aux[-1].replace(separador+"*","")
  return np.array(dados_aux) # --> dados_aux (vetor contendo a união das colunas)

#---------------------------------------------------------------------------------------------------

# Função auxiliar para separar colunas auxiliares

# Dado um vetor de strings e a definição de um separador, define uma matriz de 2 dimensões onde as colunas
# são os elementos das strings separadas pelo separador.
# (é o oposto da "join_aux")


def split_aux(dados,      # vetor de strings
              separador): # matriz com os elementos das strings do vetor original separados em colunas

  columns = dados[0].count(separador)+1
  rows = len(dados)
  dados_aux = separador.join(dados)
  dados_aux = dados_aux.split(separador)
  dados_aux = np.reshape(dados_aux,(rows,columns))
  return np.array(dados_aux) # --> dados_aux (matriz com os elementos das strings do vetor original separados em colunas)
#---------------------------------------------------------------------------------------------------



# Função tranforma strings em datetime

def str_to_datetime(data): # <-- vetor de strings

  data = list(data.astype('str'))
  data_ja_formatada = False
  # Caso a base já tenha sido aberta e as datas transformadas, a tentativa de conversão
  # geraria um erro. Por isso, se a tentativa de conversão gerar erro, consideramos que
  # a base já foi aberta uma vez e as datas já foram transformadas em datetime
  try:
    datetime.strptime(data[0],'%m/%d/%Y')
  except (ValueError):
    data_ja_formatada = True
  if not data_ja_formatada:
    data = [datetime.strptime(x, '%m/%d/%Y') for x in data]

  return data # --> data (vetor de datetime)
#---------------------------------------------------------------------------------------------------


'''
Funções para a base de inputs
____________________________________________________________________________________________________
'''
# Soma inputs com as mesmas chaves

# Essa função serve para agrupar todos os inputs da base que são aplicados na mesma abertura.

def soma_inputs_repetidos_v2(b_inputs): # --> b_inputs (matriz com apenas duas colunas, onde a primeira
                                        # é uma coluna auxiliar contendo todas as informações da abertura
                                        # (data, região, fluxo, etc) na forma de string, e outra
                                        # coluna com o valor do input que será aplicado naquela data
                                        # e abertura)

  b_inputs[:,1] = b_inputs[:,1].astype('float')
  cabecalho = ['chave','valores']
  data_pd = pd.DataFrame(b_inputs, columns = cabecalho)
  data_pd['chave']=data_pd['chave']
  # Somamos 1 em todos os valores, pois a forma correta de agrupar inputs é multiplicando-os.
  # Então se temos um input de +10% e outro de +20% na mesma semana e abertura, multiplicamos
  # 1.1*1.2 = 1.32
  data_pd['valores'] += 1
  # agrupamos todos inputs na mesma abertura multiplicando-os
  data_pd = data_pd.groupby('chave')[['valores']].prod()
  # subtraímos -1 dos inputs finais para ficarem no formato anterior:
  # 1.32 - 1 = +32% (input agregado final)
  data_pd['valores'] += -1

  # <-- retorna a mesma base, mas onde as aberturas eram repetidas os inputs foram agregados
  return np.append(np.transpose(np.array([data_pd.index.to_numpy()])),data_pd.to_numpy(),1)

# Obs: essa é a etapa mais demorada de todo processo de leitura e aplicação de inputs. Ela já está
# bem simples com apenas 1 groupby. Seria interessante pensar em formas de otimizá-la no futuro.
#---------------------------------------------------------------------------------------------------


'''
Funções para o Funil Dinâmico
____________________________________________________________________________________________________
'''
# Deslocar colunas de datas pela conversão

# Dado uma tabela de dados contendo os volumes das cohorts e suas respectivas datas,
# podemos facilmente somar o valor coincident correspondente à estas cohorts nas datas corretas criando
# uma coluna de datas auxiliares.

# Como a forma correta de somar o volume coincident oriundo das cohorts é somar o volume da W0, com o volume
# da W1 da semana anterior, o volume da W2 da anterior da anterior e assim por diante, criamos uma coluna
# de datas que é igual às datas das cohorts, mas somada à cada data o número de semanas daquela cohort.

# Assim, quando precisamos obter o volume coincident por semana, basta somar as cohorts agrupando pelas
# semanas "deslocadas" que criamos.

def shift_datas(datas,     # vetor contendo as datas das cohorts em formato datetime
                conv,      # vetor contendo as cohorts correspondentes
                intervalo, # interio que define o intervalo da cohort em dias (no caso semanal = 7 dias)
                max_conv,  # inteiro que define qual é a cohort máxima (no caso = 5)
                aplicar_ajuste_w0):

  if aplicar_ajuste_w0:
    conv = conv.replace(['Coincident'],0)
  else:
    conv = conv.replace(['Coincident'],str(max_conv))
  conv = conv.astype(float)
  conv = conv.astype(int)
  temp = conv.apply(lambda x: pd.Timedelta(x*intervalo, unit='D'))
  datas = datas + temp

  return datas # <-- datas (vetor de datas datetime deslocado através das conversões em relação às datas originais)
#---------------------------------------------------------------------------------------------------

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
                      intervalo, # interio que define o intervalo da cohort em dias (no caso semanal = 7 dias)
                      data_min,  # datetime (primiera data da base = baseline)
                      max_conv): # max_conv (inteiro que define qual é a cohort máxima (no caso = 5)
  lista_base = [base]
  for i in range(max_conv):
    copy = base.loc[base['Data']==data_min]
    copy['Data'] = copy['Data'].apply(lambda x: x-pd.Timedelta((i+1)*intervalo, unit='D'))
    lista_base = [copy.copy()] + lista_base
  base = pd.concat(lista_base)
  return base # <-- base (DataFrame igual ao input, mas extendido para o passado)
#---------------------------------------------------------------------------------------------------


# Executa o Split do volume de uma etapa na abertura nova

# No meio da execução do funil, caso uma etapa possua um split de categorias (como o caso de CA, onde
# o volume calculado deve ser dividido entre garantia e sem garantia nessa etapa), essa função faz a
# separação dos volumes.

def split_etapa(base,           # DataFrame com o volume do funil calculado até o momento. Essa base já contém as colunas com as proporções do "Split"
                etapa,          # etapa do funil em que foi indentificada a existência do "Split"
                etapas_split,   # lista de strings contendo a informação da etapa e qual abertura será o split
                chaves):        # lista com o nome das colunas que defimem as aberturas da base

  # criamos uma lista contendo as etapas que sofrerão o split e outra com as categorias
  etapas = []
  categorias = []
  aux_chaves = chaves.copy()
  for i in range(len(etapas_split)):
    e,c = etapas_split[i].split("/")
    etapas.append(e)
    categorias.append(c)

  # Caso a etapa do funil que está sendo calculada não for uma das etapas que sofrem split, retornamos
  # um vetor contendo a informação de que não existe split nessa etapa
  if etapa not in etapas:
    return np.array(["sem split","sem split"])

  else:
    # caso contrário, identificamos na base a coluna da etapa que vai sofrer split
    # e a coluna contendo as razões do split
    posi_etapa = etapas.index(etapa)
    posi_split = etapas_split[posi_etapa]

    # removemos da lista de aberturas a abertura que sofrerá split, para que seja agregada no groupby
    categoria = categorias[posi_etapa]
    aux_chaves.remove(categoria)

    # transformamos em float os valores que serão multiplicados
    base[etapa] = base[etapa].astype('float')
    base[posi_split] = base[posi_split].astype('float')

    # agregamos os valores da etapa ignorando a abertura que sofrerá split
    agg = base.groupby(aux_chaves, as_index=False)[[etapa]].sum()

    # Unimos a base original com a base de volumes agregados
    base_agg = pd.merge(base,agg,how='left',on=aux_chaves)

    # Definimos a coluna com o volume da etapa que sofre split multiplicando o volume agregado pela
    # coluna que conté as razões de split
    vol_split = base_agg[etapa+"_y"].astype('float') * base_agg[posi_split]

    return vol_split

# <-- vol_split (coluna na base com o volume que sofreu split)
#     ou
#     vetor com a informação que não existe split nessa etapa
#---------------------------------------------------------------------------------------------------



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



# Criar DF Merged com conversões, volumes e split

# Essa base geral é o join, através de todas as aberturas, das bases de ToF, cohort e split.
# Como todas essas base possuem as mesmas aberturas (a base cohort adquire coluna de datas depois de
# aplicarmos inputs), facilitamos as contas criando uma base só, onde podemos fazer operações com colunas
# inteiras de uma vez, ao invés de ficar procurando a abertura correta entre as bases.


def base_geral(base_cohort,  # DataFrame das conversões cohort
               base_ToF,     # DataFrame com os volumes de ToF
               base_split):  # DataFrame com os shares da base de split de fluxos

  # identificamos os cabecalhos como listas
  cb_cohort = list(base_cohort.columns.values)
  cb_split = list(base_split.columns.values)
  cb_ToF = list(base_ToF.columns.values)

  # definimos onde começam os dados de todas as bases com base na posição da coluna "Week Origin"
  # na base cohort
  posi_coh = cb_cohort.index("Week Origin")+1
  posi_ToF = posi_coh-1
  posi_split = posi_ToF

  # definimos as etapas cohort e de volume com base na posição de onde começam os dados
  etapas_ToF = cb_ToF[posi_ToF:]
  etapas_coh = cb_cohort[posi_coh:]

  # definimos as chaves das base (data, região, etc...)
  chaves_ToF = cb_ToF[:posi_ToF]

  # no caso da base de split, pode ser que a base não exista, então checamos isso primeiro
  if base_split.size > 2:
    # caso a base exista, definimos as etapas de split
    etapas_split = cb_split[posi_split:]
    # já unimos a base ToF com a de split. (@aqui como as bases costumavam ter os mesmo formato,
    # optamos de início uní-las via concatenação. Mas o mais correto seria dar um "merge" para evitar
    # o erro caso as bases não estejam na mesma ordem)
    #base_ToF = pd.concat([base_ToF, base_split[etapas_split]], axis=1)
    base_ToF = pd.merge(base_ToF,base_split,how='left',on=chaves_ToF)
    # redefininos a base ToF e o cabeçalho da mesma
    cb_ToF = list(base_ToF.columns.values)
  else:
    # se não existir, registramos sua não existência através de um vetor
    etapas_split = np.array([["sem split","sem split"]])


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
  merged['Data'] = pd.to_datetime(merged['Data'], infer_datetime_format=True)

  # encontramos qual é a maior cohort
  max_origin = merged[merged['Week Origin'] != "Coincident"]['Week Origin'].astype('int').max()

  # encontramos algumas definições sobre as datas que serão úteis em outras funções
  datas_unicas = merged.Data.unique()
  datas_unicas_list = list(datas_unicas)
  data_min = datas_unicas.min()
  data_max = datas_unicas.max()
  qtd_datas = datas_unicas.size
  intervalo = ((data_max-data_min)/ np.timedelta64(1,"D")) / (qtd_datas-1)

  # Com a base total gerada, repetimos a primeira semana adicionando datas para o passado
  # para executar a cohort de forma correta.
  # Utilizamos uma função auxiliar que repete os dados da primeira semana um número de vezes igual à maior
  # cohort e gera as datas passadas corretas
  merged = add_datas_passado(merged,intervalo,data_min,max_origin) # @função_auxiliar

  # Utilizamos uma função auxiliar para gerar uma coluna de datas deslocadas. Essas datas deslocadas
  # servirão para somar a "diagonal" da cohort e obter os volumes coincident.
  merged['Shifted'] = shift_datas(merged['Data'],merged['Week Origin'],intervalo,max_origin,False) # @função_auxiliar

  # <-- retornamos a base final e todas as outras informações que geramos de início
  return merged,etapas_coh,etapas_ToF,etapas_split,chaves_coh,chaves_ToF,max_origin,datas_unicas_list,data_min
#---------------------------------------------------------------------------------------------------




# Retorna a base final no formato de base coincident e base de volumes cohort
def formatacao(base,etapas_coh,etapas_vol,datas_unicas):

  # selecionar apensas as datas que importam.
  # não vamos mostrar os dados auxiliares criados em semanas anteriores à primeira semana de baseline
  base = base[base['Data'].isin(datas_unicas)]

  base= base.fillna(0.0)

  cb_base = list(base.columns.values)
  posi_origin = cb_base.index('Week Origin')

  base['Data'] = base['Data'].apply(lambda x: x.strftime('%Y/%m/%d'))

  etapas_vol_coh = etapas_coh.copy()
  for i in range(len(etapas_coh)):
    etapas_vol_coh[i] = "coh_vol_"+etapas_coh[i]

  cb_output_cohort = cb_base[:posi_origin+1]+etapas_vol_coh
  cb_output_cohort_final = cb_base[:posi_origin+1]+etapas_coh
  cb_output_coincident = cb_base[:posi_origin]+etapas_vol

  output_cohort = base[cb_output_cohort]
  output_coincident = base.loc[base['Week Origin'] == '0'][cb_output_coincident]
  '''
  output_cohort = output_cohort.to_numpy()
  output_cohort = np.append(np.array([cb_output_cohort_final]),output_cohort,0)

  output_coincident = output_coincident.to_numpy()
  output_coincident = np.append(np.array([cb_output_coincident]),output_coincident,0)
  '''
  return output_cohort,output_coincident
#---------------------------------------------------------------------------------------------------




'''
Funções para a quebra diária
____________________________________________________________________________________________________
'''

# Gera uma base de impactos de feriados somados para todas as regiões

# Com base na informação das datas, locais e tipos de feriado, força do impacto em cada dia da semana e
# share que a cidade onde cai o feriado representa na região, usamos esta função para construir uma base
# única de impactos por região, unindo todas essas informações

def base_impacto_feriados(feriados,        # Dataframe contendo as datas, cidades e tipos de feriado
                          impactos,        # DataFrame contendo o impacto estimado de feriados
                          share_cidades):  # DataFrame contendo o share que cada cidade representa dentro de uma região

  # definimos os cabeçalhos das bases
  cb_impactos = list(impactos.columns.values)
  cb_share_cidades = list(share_cidades.columns.values)

  # definimos as etapas do funil a partir do cabeçalho da base de share de cidades
  etapas = cb_share_cidades[cb_share_cidades.index('Cidade')+1:]
  # Como faremos múltiplas uniões entre bases, as colunas com as etapas repetidas adicionam "_x",
  # "_y", "_z".. nos nomes das colunas. Aqui já definimos quais etapas esses nomes estão se referindo
  etapas_share = etapas.copy()
  etapas_impactos = etapas.copy()
  etapas_finais = etapas.copy()
  for i in range(len(etapas)):
    etapas_share[i] = etapas_share[i]+"_x"
    etapas_impactos[i] = etapas_impactos[i]+"_y"
    etapas_finais[i] = etapas_finais[i]+"_z"

  # Criamos uma lista com as categoriais que serão usadas para dar join nas bases.
  # Partimos do cabeçalho completo da base de impactos e removemos todas as colunas
  # que sejam etapas do funil
  chaves_impacto = cb_impactos.copy()
  for element in etapas:
    if element in chaves_impacto:
      chaves_impacto.remove(element)

  # Abaixo definimos a base de feriados estaduais e uma base com o restante dos feriados
  #______________________________________________

  # A base de share de cidades também possui um de-para entre região, cidade e estado.
  # Usamos essa informação para unir a base de feriados com a base de share de cidades
  feriados_E = pd.merge(feriados,share_cidades,how='left',on=['Estado'])
  # Unimos essa base com a base de impactos pelas chaves definidas
  feriados_E = pd.merge(feriados_E,impactos,how='left',on=chaves_impacto)

  # Definimos uma base geral contendo os outros tipos de feriado
  feriados = pd.merge(feriados,share_cidades,how='left',on=['Cidade'])
  # Unimos essa base com a base de impactos pelas chaves definidas
  feriados = pd.merge(feriados,impactos,how='left',on=chaves_impacto)

  # Definimos o cabeçalho final provisório que a base final de feriados vai ter
  cb_final = ['Data','Região_y']+etapas_finais

  # Criamos uma lista com todas as regiões da base
  regioes = share_cidades['Região'].unique()


  # Cálculo do impacto de feriados municipais
  #______________________________________________

  # Selecionamos os feriados municipais pela coluna de tipo de feriado
  feriados_M = feriados.loc[feriados['Tipo'] == 'M']

  # O impacto dos feriados finais no caso dos municipais é = 1-((1-impacto)*share_cidade)
  feriados_M[etapas_impactos] = 1-feriados_M[etapas_impactos].astype('float').values
  feriados_M[etapas_finais] = 1-feriados_M[etapas_impactos].multiply(feriados_M[etapas_share].astype('float').values, axis="index")

  # selecionamos apenas os dados finais
  feriados_M = feriados_M[cb_final]

  # Cálculo do impacto de feriados nacionais
  #______________________________________________

  # Selecionamos os feriados nacionais pela coluna de tipo de feriado
  feriados_N = feriados.loc[feriados['Tipo'] == 'N']

  # O impacto dos feriados nacionais é o impacto original do feriado
  feriados_N[etapas_finais] = feriados_N[etapas_impactos].astype('float').values

  feriados_N = feriados_N[cb_final]

  # O impacto do feriado nacional é repetido para todas as regiões da base
  # Criamos uma cópia da base geral e depois repetimos ela adicionando todas as regiões
  feriados_N['Região_y'] = regioes[0]
  feriados_N_final = feriados_N.copy()
  for i in range(1,len(regioes)):
    feriados_N_copy = feriados_N.copy()
    feriados_N_copy['Região_y'] = regioes[i]
    frames = [feriados_N_final,feriados_N_copy]
    feriados_N_final = pd.concat(frames)


  # Cálculo do impacto de feriados estaduais
  #______________________________________________

  # Selecionamos os feriados estaduais pela coluna de tipo de feriado
  feriados_E = feriados_E.loc[feriados_E['Tipo'] == 'E']

  # Definimos, inicialmente, o impacto dos feriados estaduais como sendo o impacto dos feriados sem alteração
  feriados_E[etapas_finais] = feriados_E[etapas_impactos].astype('float').values

  # selecionamos apenas os dados finais
  feriados_E = feriados_E[cb_final]

  # Como a base de feriados estaduais possui regiões repetidas (pois feriados estaduais que cairam em
  # cidades pertencentes à uma mesma região foram contadas individualmente), definimos o impacto final
  # do feriado estadual por região como sendo a média dos impactos das cidades dentro daquela região
  feriados_E = feriados_E.groupby(['Data','Região_y'], as_index=False)[etapas_finais].mean()


  # Compor todos os tipos de feriado
  #______________________________________________

  # Aqui só concatenamos as bases de feriados nacionais, municipais e estaduais na base final

  feriados_finais = pd.concat([feriados_N_final,feriados_E,feriados_M], axis=0)

  feriados_finais = feriados_finais.groupby(['Data','Região_y'], as_index=False)[etapas_finais].prod()

  feriados_finais = feriados_finais.rename(columns={"Região_y": "Região"})

  feriados_finais['Data'] = pd.to_datetime(feriados_finais['Data'], infer_datetime_format=True)

  return feriados_finais # <-- retornamos DataFrame final de feriados, com somente a data, região e impacto final em cada etapa do funil

#-------------------------------------------------------------------------------------------------



# Gera uma proxi de base diária a partir da base semanal

# O primeiro passo de quebrar as metas semanais numa base diária é definir os dias dessa base.
# Fazemos isso criando uma base diária que é a base semanal repetida 7 vezes, mas cada vex adicionamos 1
# dia na coluna das semanas. Assim, obtemos todos os 7 dias de todas as semanas na base.

# Os volumes ainda serão os semanais, repetidos todos os dias.

def gerador_base_diaria(base_coincident): # --> DataFrame de volumes semanais coincident

  # Encontramos a coluna com as semanas e formatamos como datetime
  base_coincident['Data'] = pd.to_datetime(base_coincident['Data'], infer_datetime_format=True)
  # Criamos a base diária copiando a base semanal
  base_diaria = base_coincident.copy()
  # Definimos a coluna das semanas na base diária como sendo as datas originais da base semanal
  base_diaria['Semana'] = base_coincident['Data'].values

  # Como a primeira cópia da base semanal contém apenas as segundas-feiras, vamos repetir a cópia mais 6 vezes,
  # adicionando 1 dia na coluna das datas a cada repetição
  lista_base_diaria = [base_diaria]
  for i in range(6):
    copy = base_coincident.copy()
    copy['Data'] = copy['Data'].apply(lambda x: x+pd.Timedelta(i+1, unit='D'))
    copy['Semana'] = base_coincident['Data'].values
    lista_base_diaria = [copy.copy()] + lista_base_diaria
  base_diaria = pd.concat(lista_base_diaria)

  # Definimos as outras colunas de datas que estão presentes na base diária
  base_diaria['Dia da Semana'] = base_diaria['Data'].dt.day_name()
  base_diaria['Mês'] = base_diaria['Data'].dt.month
  base_diaria['Ano'] = base_diaria['Data'].dt.year

  return base_diaria # <-- DataFrame contendo as datas corretas da base diária (volumes ainda são os semanais repetidos todos os dias)
#-------------------------------------------------------------------------------------------------



