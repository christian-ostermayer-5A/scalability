#@title Def check_etapas_do_funil

# Importando Bibliotecas necessárias:
import numpy as np
from colored import colored
import pandas as pd
from tabulate import tabulate
from difflib import SequenceMatcher
import itertools
from itertools import compress
import datetime
from datetime import datetime 


def check_etapas_do_funil(lista_etapas_conversao, # lista com todas as etapas de conversão definidas pelo usuário no painel de controle
                          lista_topos_de_funil,   # lista com os ToF's definida pelo usuário no painel de controle
                          df_on_top_ratio,
                          racional_on_top_ratio,
                          flag_gerar_on_top_ratio,
                          nome_do_arquivo):       # string com o nome do arquivo

  # Definições iniciais
  #-------------------------------------------------------------------------------------------------
  mensagem = ''

  erro = 0

  etapas = lista_etapas_conversao.copy()
  topo = lista_topos_de_funil.copy()

  etapas_conv = []
  etapas_vol = []

  # Removemos os valores duplicados das etapas e ToF's
  #-------------------------------------------------------------------------------------------------
  etapas = list(dict.fromkeys(etapas)) # remove duplicadas
  topo = list(dict.fromkeys(topo)) # remove duplicadas


  # Caso existam etapas extra no on top rations, vamos identifica-las:
  #-------------------------------------------------------------------------------------------------
  etapas_ratio = []
  if not flag_gerar_on_top_ratio and len(df_on_top_ratio) > 0:
    colunas_on_top = list(df_on_top_ratio.columns.values)
    etapas_ratio_aux = [x for x in colunas_on_top if "/" in x]
    for x in etapas_ratio_aux:
      etapas_ratio = etapas_ratio + [x.split("/")[0].lower()] + [x.split("/")[1].lower()]
    etapas_ratio = list(dict.fromkeys(etapas_ratio))
  
  if flag_gerar_on_top_ratio:
    div_racionais = racional_on_top_ratio
    div_racionais = div_racionais.split(',')
    racionais = [item.strip().split('|') for item in div_racionais] #Primeira separação dos racionais que devem ser zerados, em certas circusntâncias.
    aux = list() #Criação da lista que vai possuir as listas de racionais e regras.
    for i in range (len(racionais)): #Nesse for o aux se torna uma lista, dentro de uma lista, dentro de outra. trazendo no micro: 'vb/os', 'abertura', 'todos'
      a = [item.strip().split(':') for item in racionais[i]]
      aux.append(a) 
    ratios = list()
    for i in range (len(aux)): #separação de uma lista trazendo só os racionais.
      kpi = aux[i][0][0]
      ratios.append(kpi)

    ratios = [x.lower() for x in ratios] #transforma a lista de strings para letras minúsculas
    ratios_split_1 = [item.split('/', 1)[0].strip() for item in ratios] #Separação para pegar só os dividendos
    ratios_split_2 = [item.split('/', 1)[1].strip() for item in ratios] #Separação para pegar só os divisores 
    
    etapas_ratio = list(dict.fromkeys(ratios_split_1+ratios_split_2))  


  # Check de etapas e ToF's duplicados
  #-------------------------------------------------------------------------------------------------
  if len(lista_etapas_conversao) > len(etapas):
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', no '+colored('Painel de Controle','yellow')+', existem etapas do funil duplicadas.'
    erro = erro+1
  if len(lista_topos_de_funil) > len(topo):
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', no '+colored('Painel de Controle','yellow')+', existem etapas de ToF duplicadas.'
    erro = erro+1


  # Daixamos todas as etapas com caracteres minúsculos
  #-------------------------------------------------------------------------------------------------
  etapas = [e.lower() for e in etapas]
  topo = [t.lower() for t in topo]


  # Checar formatação da divisão em vírgula e "2"
  i=1
  for e in etapas:
    teste = e.split('2')    
    if len(teste) == 1:
      if (i==len(etapas) or i==1) and teste == ['']:
        mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na aba '+colored('Painel de Controle','yellow')+', parece que a sua lista de conversões está começando ou terminando com uma vírgula.\nAs vírgulas devem estar entre as conversões. Não devem iniciar ou terminar a lista.'
        erro = erro+1
      else:
        mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na aba '+colored('Painel de Controle','yellow')+', não foi possível separar as etapas da seguinte conversão "'+colored(e,'red')+'", que é a '+str(i)+'º conversão indicada na lista de conversões. \nCertifique-se de que todas as conversões tenham o caractere "2" para indicar a primeira e segunda etapa da conversão e que todas as conversões da lista sejam separadas por vírgulas.'
        erro = erro+1
    else:   
      mensagem = mensagem
    i += 1

  # Ordenamos as etapas do funil
  #-------------------------------------------------------------------------------------------------
  if erro == 0: # caso não tenha erros anteriores:

    primeiras_etapas = []
    segundas_etapas = []

    for e in etapas:
      primeiras_etapas = primeiras_etapas + [e.split('2')[0]]
      segundas_etapas = segundas_etapas+[e.split('2')[1]]
    
    for e in primeiras_etapas:
      if e not in segundas_etapas:
        primeira_etapa = e


    primeira_etapa_aux = primeira_etapa
    while len(etapas_conv) < len(etapas):
      
      encontrou_etapa = False

      for e in etapas:
        if e.split('2')[0] == primeira_etapa_aux:
          etapas_conv = etapas_conv+[e]
          etapas_vol = etapas_vol+[e.split('2')[0]]
          primeira_etapa_aux = e.split('2')[1]
          encontrou_etapa = True
      
      # Checagem se uma das etapas de conversão não está ligada à nenhuma outra etapa do funil
      if not encontrou_etapa:
        mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', no '+colored('Painel de Controle','yellow')+', a etapa '+colored(primeira_etapa_aux,'red')+' não possui conversão. \nSe essa etapa for a última do funil e você receber essa mensagem, é sinal que existem etapas conflitantes.'
        erro = erro+1
        break
        
    etapas_vol = etapas_vol+[primeira_etapa_aux]


    if len(etapas_conv) < len(etapas):
      mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', no '+colored('Painel de Controle','yellow')+', não foi possível ordenar todas as etapas de conversão. \nAs conversões fornecidas pelo usuário foram: '+colored(str(etapas),'red')+'\nA tentativa de ordenar as etapas resultou num número inferior de conversões: '+colored(str(etapas_conv),'red')
      erro = erro+1

    # Caso não tenha ocorrido erro com as etapas do funil, vamos checar os ToF's
    #-----------------------------------------------------------------------------------------------
    if erro == 0:

      # Checando se os ToF's estão presentes nas etapas de volume e se não são a última etapa:
      if etapas_vol[0] not in topo:
        mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', no '+colored('Painel de Controle','yellow')+', a primeira etapa de funil deve ser um ToF'
        erro = erro+1          
      for t in topo:
        if t not in etapas_vol:
          mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', no '+colored('Painel de Controle','yellow')+', o ToF '+colored(t,'red')+' não consta nas seguintes etapas do funil definidas pelo usuário: '+colored(etapas_vol,'red')
          erro = erro+1      
        if t == etapas_vol[-1]:
          mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', no '+colored('Painel de Controle','yellow')+', o ToF '+colored(t,'red')+' não pode ser a última etapa do funil definidas pelo usuário: '+colored(etapas_vol,'red')
          erro = erro+1        
  
  # Definimos as etapas extra de ratio, caso existam:
  #-------------------------------------------------------------------------------------------------
  if erro == 0:
    etapas_extra = list(set(etapas_ratio)-set(etapas_vol))

  # Retorna:
  # lista com as etapas de conversão do funil ordenadas
  # lista com as etapas de volume do funil ordenadas
  # lista com as etapas extra de volume
  # string formatada (pulando lunhas e com cores) da mensagem de erro ou aviso
  # inteiro representando o número de erros que esse check encontrou
  return etapas_conv,etapas_vol,etapas_extra,mensagem,erro 





#@title encontra_melhor_correspondencia_string


def encontra_melhor_correspondencia_string(teste,lista,valor_corte):

  #-------------------------------------------------------------------------------------------------
  def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):
      """Return best matching substring of corpus.

      Parameters
      ----------
      query : str
      corpus : str
      step : int
          Step size of first match-value scan through corpus. Can be thought of
          as a sort of "scan resolution". Should not exceed length of query.
      flex : int
          Max. left/right substring position adjustment value. Should not
          exceed length of query / 2.

      Outputs
      -------
      output0 : str
          Best matching substring.
      output1 : float
          Match ratio of best matching substring. 1 is perfect match.
      """

      def _match(a, b):
          """Compact alias for SequenceMatcher."""
          return SequenceMatcher(None, a, b).ratio()

      def scan_corpus(step):
          """Return list of match values from corpus-wide scan."""
          match_values = []

          m = 0
          while m + qlen - step <= len(corpus):
              match_values.append(_match(query, corpus[m : m-1+qlen]))
              if verbose:
                  print(query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
              m += step

          return match_values

      def index_max(v):
          """Return index of max value."""
          return max(range(len(v)), key=v.__getitem__)

      def adjust_left_right_positions():
          """Return left/right positions for best string match."""
          # bp_* is synonym for 'Best Position Left/Right' and are adjusted 
          # to optimize bmv_*
          p_l, bp_l = [pos] * 2
          p_r, bp_r = [pos + qlen] * 2

          # bmv_* are declared here in case they are untouched in optimization
          bmv_l = match_values[p_l // step]
          bmv_r = match_values[p_l // step]

          for f in range(flex):
              ll = _match(query, corpus[p_l - f: p_r])
              if ll > bmv_l:
                  bmv_l = ll
                  bp_l = p_l - f

              lr = _match(query, corpus[p_l + f: p_r])
              if lr > bmv_l:
                  bmv_l = lr
                  bp_l = p_l + f

              rl = _match(query, corpus[p_l: p_r - f])
              if rl > bmv_r:
                  bmv_r = rl
                  bp_r = p_r - f

              rr = _match(query, corpus[p_l: p_r + f])
              if rr > bmv_r:
                  bmv_r = rr
                  bp_r = p_r + f

              if verbose:
                  print("\n" + str(f))
                  print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                  print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                  print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                  print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))

          return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])

      if not case_sensitive:
          query = query.lower()
          corpus = corpus.lower()

      qlen = len(query)

      if flex >= qlen/2:
          #print("Warning: flex exceeds length of query / 2. Setting to default.")
          flex = 1
      if step <= 0:
          #print("Warning: flex exceeds length of query / 2. Setting to default.")
          step = 1

      match_values = scan_corpus(step)
      pos = index_max(match_values) * step

      pos_left, pos_right, match_value = adjust_left_right_positions()

      return corpus[pos_left: pos_right].strip(), match_value
  #-------------------------------------------------------------------------------------------------

  col_match = ''
  deu_certo = False

  score_list = []
  substring_list = []

  for string in lista:
    substring_list = substring_list + [get_best_match(query = teste, corpus = string, step=len(teste)-1, flex=round(len(teste)/3), case_sensitive=False, verbose=False)[0]]
    score_list = score_list + [get_best_match(query = teste, corpus = string, step=len(teste)-1, flex=round(len(teste)/3), case_sensitive=False, verbose=False)[1]]

  best_score = max(score_list)

  index = np.where(np.array(score_list)==best_score)[0]

  if len(index) != 1 or best_score < valor_corte:
    return index[0],col_match,deu_certo


  else:

    best_substring = substring_list[index[0]]
    col_match = lista[index[0]]
    deu_certo = True

    '''
    split_possibilities = ['2','_',' ']

    best_substring_splited = []
    teste_splited = []

    for spt in split_possibilities:

      if spt in best_substring:
        best_substring_splited = best_substring.split(spt)

      if spt in teste:
        teste_splited = teste.split(spt)

    print(teste_splited,best_substring_splited)

    if len(teste_splited) == 0 and len(best_substring_splited) == 0:
      if best_substring in teste or teste in best_substring:
        deu_certo = True
 
    elif len(teste_splited) != 0 and len(best_substring_splited) == 0:  
      count_match = 0
      for sub_teste_splited in teste_splited:
        if sub_teste_splited in best_substring:
          count_match += 1
      if count_match > 0:
        deu_certo = True

    elif len(teste_splited) == 0 and len(best_substring_splited) != 0:  
      count_match = 0
      for sub_best_splited in best_substring_splited:
        if sub_best_splited in teste:
          count_match += 1
      if count_match > 0:
        deu_certo = True

    else:
      diff_1 = list(set(teste_splited) - set(best_substring_splited))
      diff_2 = list(set(best_substring_splited) - set(teste_splited))
      if len(diff_1) == 0 and len(diff_2) == 0:
        deu_certo = True

    if deu_certo:
      col_match = lista[index[0]]
    '''

    return index[0],col_match,deu_certo
  
  
  #@title Def check_colunas

def check_colunas(df,                    # DataFrame
                  lista_df,              # Lista de DataFrame que será usado como modelo de conteúdo e nome das colunas.
                  aberturas,             # Lista com as aberturas das bases.
                  colunas_obrigatorias,  # lista com as colunas obrigatórias e a ordem
                  retorna_col_valores,   # booleano que determina se a função vai retornar as colunas de valores
                  mantem_formatacao_original,
                  dict_renames,          # dicionário com o de/para de colunas numéricas
                  coluna_de_conversoes,
                  colunas_datas,
                  nome_do_arquivo):  
  
  # Definições iniciais
  #-------------------------------------------------------------------------------------------------
  mensagem = ''

  erro = 0
  
  colunas_extra = []

  nome_da_base = df.name
  
  colunas = df.columns.values
  colunas_o = colunas_obrigatorias.copy()

  aberturas_copy = [c.lower() for c in aberturas] + ['tier']


  # Removemos os valores duplicados das colunas da base e das colunas obrigatórias
  #-------------------------------------------------------------------------------------------------
  colunas = list(dict.fromkeys(colunas)) # remove duplicadas
  colunas_o = list(dict.fromkeys(colunas_o)) # remove duplicadas

  # Checamos se existem valores duplicados nas colunas:
  #-------------------------------------------------------------------------------------------------
  if len(df.columns.values) > len(colunas):
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome_da_base,'yellow')+', foram encontradas colunas duplicadas.'
    erro = erro+1
  if len(colunas_obrigatorias) > len(colunas_o):
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored('Painel de Controle','yellow')+', foram encontradas colunas duplicadas.'
    erro = erro+1


  # Checamos a existência das colunas obrigatórias na base:
  #-------------------------------------------------------------------------------------------------
  if erro == 0: # caso não tenha colunas repetidas

    # Vamos deixar todos os nomes das colunas em letras minúsculas:
    if not mantem_formatacao_original:
      colunas = [c.lower() for c in colunas]
      colunas_o = [c.lower() for c in colunas_o]

    # Renomeamos as colunas da base para ficar com tudo minúsculo
    df = df.rename(columns=dict(zip(df.columns.values, colunas)))
    
    # Vamos determinar quais colunas estão faltando e quais estão com nomes errados na base:
    colunas_faltantes = list(set(colunas_o) - set(colunas))


    # Check das colunas de aberturas ou week origin ou tier que estão faltando na base:
    #-----------------------------------------------------------------------------------------------
    if len(colunas_faltantes) > 0:

      # Vamos tentar encontrar as colunas faltantes na base modelo, mas somente aquelas que sejam aberturas:
      colunas_substituidas = []
      for col_faltante in colunas_faltantes:
        # Procuramos a coluna faltante em alguma outra base fornecida:
        for base_modelo in lista_df:
          df_modelo_copy = base_modelo 
          nome_da_base_modelo = df_modelo_copy.name
          colunas_modelo = df_modelo_copy.columns.values
          colunas_modelo = [c.lower() for c in colunas_modelo]
          df_modelo_copy = df_modelo_copy.rename(columns=dict(zip(df_modelo_copy.columns.values, colunas_modelo)))
          if col_faltante in colunas_modelo and col_faltante in aberturas_copy:
            break

        # Se a coluna faltante existir na base modelo, definimos o conteúdo modelo dessa coluna:
        if col_faltante in colunas_modelo and col_faltante in aberturas_copy:
          conteudo_modelo = list(df_modelo_copy[col_faltante].unique())

          # Vamos procurar qual coluna da base sendo testada tem a maior compatibilidade com o conteúdo modelo:
          comparacao = []
          comparacao_faltantes = []
          for col in colunas:
            conteudo_col = list(df[col].unique())
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

              df = df.rename(columns={coluna_compativel:col_faltante})
              colunas = list(df.columns.values)
              colunas_substituidas = colunas_substituidas + [col_faltante]
              mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome_da_base,'yellow')+', não foi encontrada a coluna: '+colored(str(col_faltante),'red')+'\nPorém, o conteúdo da coluna '+colored(str(coluna_compativel),'red')+' parece ser o mais compatível com o conteúdo da coluna '+colored(str(col_faltante),'red')+' encontrada na base modelo '+colored(nome_da_base_modelo,'yellow')+'\nVai ser assumido que estas colunas são equivalentes.'
              

          except:
            colunas_substituidas = colunas_substituidas

      
      # Após renomear as colunas encontradas, redefinimos as colunas faltantes, se ainda existirem. 
      colunas_faltantes = list(set(colunas_faltantes) - set(colunas_substituidas))


      # Agora vamos procurar as colunas de valores
      #--------------------------------------------------------------------------------------------_

      if len(colunas_faltantes) > 0:
        if len(set(dict_renames).intersection(set(df.columns))) > 0:
          mensagem = mensagem + f"\n\nAs seguintes colunas {set(dict_renames).intersection(set(df.columns))} no arquivo "+colored(nome_da_base,'yellow')+" foram substituídas com base no dicionário de colunas"
        for coluna in dict_renames:
          df = df.rename(columns={coluna:dict_renames[coluna]})

      colunas = df.columns.values
      colunas_faltantes = list(set(colunas_o) - set(colunas))


      # Caso a aplicação do dicionário ainda assim resultou em colunas faltantes, vamos tentar encontrar
      # as colunas de valores pelo nome mais parecido no dicionário:
      if len(colunas_faltantes) > 0:

        colunas_substituidas = []

        lista_de_chaves_dict = list(dict_renames.keys())
        lista_de_valores_dict = list(dict_renames.values())

        for coluna in colunas:
          
          # Função auxiliar que encontra a melhor correspondência única entre strings
          index_dict,string_match,match_check = encontra_melhor_correspondencia_string(teste = coluna,
                                                                            lista = lista_de_chaves_dict,
                                                                            valor_corte = 0.6)
          
          # Caso houve sucesso em encontrar um nome parecido no dicionário e este nome
          # seja uma das colunas faltantes:
          if match_check and lista_de_valores_dict[index_dict] in colunas_faltantes:

            # como estamos tentando encontrar coluas de valores, se não for possivel formatar como
            # float a coluna encontrada não vamos vazer a substituicao:
            df_copy = df.copy()
            check_valores = False
            try:
              df_copy[coluna] = df_copy[coluna].astype(float)
              check_valores = True
            except:
              check_valores = False
            
            # Se deu certo formatar em valores, vamos trocar o nome da coluna
            if check_valores:
              df = df.rename(columns={coluna:lista_de_valores_dict[index_dict]})
              colunas = list(df.columns.values)
              colunas_substituidas = colunas_substituidas + [coluna]            
              mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome_da_base,'yellow')+', não foram encontradas as colunas: '+colored(str(colunas_faltantes),'red')+'\nPorém, foi encontrada na base a coluna '+colored(str(coluna),'red')+' que parece ser a mais compatível com a coluna '+colored(str(lista_de_valores_dict[index_dict]),'red')+' encontrada no dicionário definido intermanete.'+'\nVai ser assumido que estas colunas são equivalentes.'
            else:
              mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome_da_base,'yellow')+', não foram encontradas as colunas: '+colored(str(colunas_faltantes),'red')+'\nPorém, foi encontrada na base a coluna '+colored(str(coluna),'red')+' que parece ser a mais compatível com a coluna '+colored(str(lista_de_valores_dict[index_dict]),'red')+' encontrada no dicionário definido intermanete.'+'\nPorém, não foi possível formatar como valor a coluna '+colored(str(coluna),'red')

          
      colunas = df.columns.values
      colunas_faltantes = list(set(colunas_o) - set(colunas))


      # Caso a coluna faltante seja a coluna de conversões, vamos tentar encontrá-la pelo conteúdo:
      if len(colunas_faltantes) > 0 and (coluna_de_conversoes in colunas_faltantes or coluna_de_conversoes.lower() in colunas_faltantes):
        
        col_faltante = coluna_de_conversoes.lower()
        conteudo_modelo_1 = ['0','1','2','3','4','5','Coincident']
        conteudo_modelo_2 = ['w0','w1','w2','w3','w4','w5+','']

        # Vamos procurar qual coluna da base sendo testada tem a maior compatibilidade com o conteúdo modelo:
        comparacao = []
        for col in colunas:
          conteudo_col = list(df[col].unique())
          conteudo_col = [str(c).lower() for c in conteudo_col]
          diferenca_1 = list(set(conteudo_modelo_1) - set(conteudo_col)) + list(set(conteudo_col) - set(conteudo_modelo_1))
          diferenca_2 = list(set(conteudo_modelo_2) - set(conteudo_col)) + list(set(conteudo_col) - set(conteudo_modelo_2))
          comparacao = comparacao+[min(len(diferenca_1),len(diferenca_2))]

        try:
          maior_compatibilidade = min(comparacao)
          coluna_compativel = colunas[comparacao.index(maior_compatibilidade)]

          # Caso tenha sido encontrada somente uma coluna na base original com o máximo de compatibilidade, vamos renomer a coluna:
          df = df.rename(columns={coluna_compativel:col_faltante})
          colunas = list(df.columns.values)
          colunas_substituidas = colunas_substituidas + [col_faltante]
          mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome_da_base,'yellow')+', não foi encontrada a coluna: '+colored(str(col_faltante),'red')+'\nPorém, o conteúdo da coluna '+colored(str(coluna_compativel),'red')+' parece ser o mais compatível com o conteúdo modelo de '+colored(str(col_faltante),'red')+', definido como '+colored(conteudo_modelo_1,'yellow')+' ou '+colored(conteudo_modelo_2,'yellow')+'\nVai ser assumido que estas colunas são equivalentes.'

        except:
          colunas_substituidas = colunas_substituidas

      colunas = df.columns.values
      colunas_faltantes = list(set(colunas_o) - set(colunas))


      # Caso a coluna faltante seja somente uma coluna de datas, vamos tentar renomear se encontrarmos apenas
      # uma coluna na base passível de ser formatada como data:
      if len(colunas_faltantes) > 0 and len(colunas_datas) == 1 and (colunas_datas[0] in colunas_faltantes or colunas_datas[0].lower() in colunas_faltantes):
        
        col_faltante =  colunas_datas[0].lower()
        comparacao = []

        df_copy = df.copy()

        # tentamos formatar como data e como valor cada uma das colunas das bases:
        for coluna in colunas:   

          formatou_como_data = False
          check_valores = False

          try:
            df_copy[coluna] = df_copy[coluna].astype(float)
            check_valores = True
          except:
            check_valores = False

          try:
            df_copy[coluna] = pd.to_datetime(df_copy[coluna],infer_datetime_format=True)
            formatou_como_data = True
          except Exception as e:
            formatou_como_data = False


          # Só aceitaos a coluna como sendo uma coluna de datas se não tiver sido possível
          # formatar como valor e tiver sido possível formatar como data.            
          if not check_valores and formatou_como_data:
            comparacao = comparacao + [coluna]
          else:
            comparacao = comparacao

        # Se somente 1 coluna for formatável, vamos considerá-la como sendo a coluna de datas:
        if len(comparacao) == 1:
          df = df.rename(columns={comparacao[0]:col_faltante})
          colunas = list(df.columns.values)
          mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome_da_base,'yellow')+', não foi encontrada a coluna: '+colored(str(col_faltante),'red')+'\nPorém, o conteúdo da coluna '+colored(str(comparacao[0]),'red')+' parece ser a única coluna que pode ser formatada no formato de datas.'+'\nVai ser assumido que estas colunas são equivalentes.'



      colunas = df.columns.values
      colunas_faltantes = list(set(colunas_o) - set(colunas))


      if len(colunas_faltantes) > 0:
        mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome_da_base,'yellow')+', estão faltando as seguintes colunas: '+colored(str(colunas_faltantes),'red')+'\nBaseado na lista de colunas obrigatórias definida pelo usuário: '+colored(str(colunas_o),'red')
        erro = erro+1


    # Depois de verificar as colunas faltantes, vamos verificar se após uma possível renomeação de colunas ainda existem colunas sobrando:
    #-----------------------------------------------------------------------------------------------
    colunas = list(df.columns.values)
    colunas_extra = list(set(colunas) - set(colunas_o))

    # Caso desejamos retornar as colunas de valores, é importante retornar na ordem que aparecem na base original
    if retorna_col_valores and len(colunas_extra) > 0:
      colunas_extra_original = colunas_extra.copy()
      colunas_extra = []
      for c in colunas:
        if c in colunas_extra_original:
          colunas_extra = colunas_extra + [c]

    # Check das colunas que estão sobrando na base. Se não estiver definido quais são as colunas de valores,
    # verificamos apenas a existência das colunas obrigatórias:
    if len(colunas_extra) > 0 and not retorna_col_valores:
      df = df.drop(columns=colunas_extra)
      mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(nome_da_base,'yellow')+', as seguintes colunas não devem existir: '+colored(str(colunas_extra),'red')+'\nEssas colunas serão removidas e os valores serão somados e agrupados.'+'\nBaseado na lista de colunas obrigatórias definida pelo usuário: '+colored(str(colunas_o),'red')
      #erro = erro+1

    # Se não estiver definido quais são as colunas de valores, retornamos as colunas que foram encontradas
    # em excesso como sendo as colunas de valores dessa base.
    if erro == 0 and retorna_col_valores:
      df = df[colunas_o+colunas_extra]
    elif erro == 0:
      df = df[colunas_o]


  df.name = nome_da_base
  #print(df.head(3))

  # Retorna
  # DataFrame com as colunas ordenadas
  # Lista com as colunas contendo valores (se houver necessidade)
  # string formatada (pulando lunhas e com cores) da mensagem de erro ou aviso
  # inteiro representando o número de erros que esse check encontrou
  return df,colunas_extra,mensagem,erro



#@title Def check_valores

def check_valores(df,                    # DataFrame já deve ter checado a existencia das colunas de valores
                  colunas_de_valores,    # lista com as colunas que contém valores
                  check_valores_vazios,  # Boleano que, caso seja verdadeiro, verifica se existem blocos de valores vazios nas colunas
                  mantem_formatacao_original,
                  nome_do_arquivo):
  

  # Definições iniciais
  #-------------------------------------------------------------------------------------------------
  mensagem = ''

  erro = 0

  nome_da_base = df.name

  # Vamos transformar as colunas de valores em caracteres minusculos:
  if not mantem_formatacao_original:
    colunas_de_valores = [e.lower() for e in colunas_de_valores]

  # Primeiro, vamos trasnformar as colunas de valores em texto:
  df[colunas_de_valores] = df[colunas_de_valores].astype(str)


  # Para cada coluna de valor, vamos verificar o preenchimento das colunas:
  #-------------------------------------------------------------------------------------------------
  for coluna in colunas_de_valores:
    # Checar se não houve erro de colunas incompletas.
    if check_valores_vazios: # Só checamos bases específicas.

      vazios = df.loc[df[coluna] == '',[coluna]] # selecionamos os valores vazios de cada coluna

      # Caso existam valores vazios, verificamos se o final da coluna possui vários valores vazios
      # em sequência contínua:
      if len(vazios) > 0:
        indice_vazios = vazios.index.astype(int).values
        delta_indices = indice_vazios[1:]-indice_vazios[:-1]
        vazios_seguidos = np.where(delta_indices==1)[0]
        vazios_seguidos_ajustados = vazios_seguidos
        for v in range(len(vazios_seguidos)):
          if v != len(vazios_seguidos)-1:
            if (vazios_seguidos[v+1] - vazios_seguidos[v]) > 1:
              vazios_seguidos_ajustados = np.append(vazios_seguidos_ajustados,np.array([vazios_seguidos[v]+1]))
          else:
            vazios_seguidos_ajustados = np.append(vazios_seguidos_ajustados,np.array([vazios_seguidos[-1]+1]))
        vazios_seguidos_ajustados = np.sort(vazios_seguidos_ajustados)

        indice_vazios = np.append(indice_vazios,np.array([indice_vazios[-1]+1]))

        if len(vazios_seguidos_ajustados) > 0:
          linhas_blocos_vazios = indice_vazios[vazios_seguidos_ajustados]+1
          linhas_blocos_vazios_mensagem = linhas_blocos_vazios
          if len(linhas_blocos_vazios_mensagem) > 20:
            linhas_blocos_vazios_mensagem = str(linhas_blocos_vazios_mensagem[:20])+'... mais outras '+str(len(linhas_blocos_vazios_mensagem)-20)+' linhas.'
          else:
            linhas_blocos_vazios_mensagem = str(linhas_blocos_vazios_mensagem)

          if np.max(delta_indices) == 1 and linhas_blocos_vazios[-1] == len(df):
            mensagem = mensagem+'\n\nNo arquivo '+f'\033[1;34m"{nome_do_arquivo}"\034[0;0;0m'+', parece que a coluna '+f'\033[1;31m"{coluna}"\033[0;0;0m'+' da base '+colored(nome_da_base,'yellow') +' não está com valores preenchidos até o final. \nA última linha com valores é a linha '+colored(str(linhas_blocos_vazios[0]-1),'red')+', sendo que a base possui um total de '+str(len(df)+1)+' linhas.\nComo não existem outras células vazias nesta coluna, o algoritmo vai interpretar que houve um erro no preenchimento destas células.'
            erro = erro+1
          elif np.max(delta_indices) == 1:
            mensagem = mensagem+'\n\nNo arquivo '+f'\033[1;34m"{nome_do_arquivo}"\034[0;0;0m'+', parece que a coluna '+f'\033[1;31m"{coluna}"\033[0;0;0m'+' da base '+colored(nome_da_base,'yellow') +' contém um bloco de células vazias que se inicia na linha '+colored(str(linhas_blocos_vazios[0]),'red')+'. \nComo não existem outras células vazias nesta coluna, o algoritmo vai interpretar que houve um erro no preenchimento destas células.'
            erro = erro+1
          else:
            mensagem = mensagem+'\n\nNo arquivo '+f'\033[1;34m"{nome_do_arquivo}"\034[0;0;0m'+', parece que a coluna '+f'\033[1;31m"{coluna}"\033[0;0;0m'+' da base '+colored(nome_da_base,'yellow') +' contém blocos de células vazias nas linhas '+colored(linhas_blocos_vazios_mensagem,'red')


  # Para cada coluna de valor, vamos tentar formatar para float:
  #-------------------------------------------------------------------------------------------------
  for coluna in colunas_de_valores:

    # Vamos verificar se a formatação do sheets não está errada (vírgulas no lugar de pontos na divisão decimal),
    # causada por uma configuração errada de país.
    erro_de_formatacao = False # iniciamos o marcador de erro de formatação como falso

    # Para cada coluna, separamos os valores que contém vírgula. Depois, separamos dentre estes
    # os que contém ponto também.
    virgulas = df[df[coluna].str.contains(',')][coluna]
    virgulas = virgulas.str.replace('.','#') # substituímos os pontos por cerquilha, pois o código não funcionou procurando os pontos diretamente
    pontos_e_virgulas = virgulas[virgulas.str.contains('#')]

    # Checar se a formatação está correta:
    if len(virgulas) > 0:
      if len(pontos_e_virgulas) == 0:
        if len(virgulas) > len(df)/2:
          # Caso uma coluna contenha mais da metade dos valores com vírgula e nenhum ponto, vamos assumir
          # que provavelmente a formatação do arquivo está errada.
          erro_de_formatacao = True

      # Verificar se o ponto está antes da virgula. Se os valores de texto apresentarem pontos
      # em posições antes das vírgulas, com certeza a formatação do arquivo está errada.
      elif len(str(pontos_e_virgulas.values[0]).split('#')[0]) < len(str(pontos_e_virgulas.values[0]).split(',')[0]):
        erro_de_formatacao = True
    
    # Caso haja erro de formatação, avisamos o usuário:
    if erro_de_formatacao:
      mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', a base '+colored(nome_da_base,'yellow')+' parece estar com erro de formatação. \nCertifique-se que o arquivo utilizado para gerar essa base está na configuração '+colored("United States",'red')
      erro = erro+1
      break


    # Caso não haja erro de formatação do arquivo inteiro, vamos proceder com a formatação dos valores
    #-----------------------------------------------------------------------------------------------
    if erro == 0:

      # Vamos substituir vírgulas por pontos
      df[coluna] = df[coluna].str.replace(',','')

      # Vamos substituir os vazios por zero
      df.loc[df[coluna] == '',[coluna]] = '0.0'


      # Vamos remover as porcentagens
      porcentagens = df[df[coluna].str.contains('%')][coluna]
      try:
        porcentagens = porcentagens.str.rstrip('%').astype(float) / 100.0
      except Exception as e:
        erro = erro+1
        mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+' não foi possível formatar os valores da coluna '+colored(coluna,'red')+'da base '+colored(nome_da_base,'yellow') +'\nO erro na formatação foi o seguinte:'+'\n'+colored(str(e),'red')+'\nCaso a coluna mencionada não devesse conter valores numéricos, este erro pode indicar que houve um erro anterior na definição de colunas obrigatórias das bases'
      if erro == 0:
        df.loc[df[coluna].str.contains('%'),[coluna]] = np.transpose(np.array([porcentagens.values]))


        # Depois de substituir vírgulas, valores vazios e porcentagens, tentamos formatar todos os valores
        # da coluna para float:
        try:
          df[coluna] = df[coluna].astype(float)
        except Exception as e:
          erro = erro+1
          mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+' não foi possível formatar os valores da coluna '+colored(coluna,'red')+' da base '+colored(nome_da_base,'yellow') +'\nO erro na formatação foi o seguinte:'+'\n'+colored(str(e),'red')+'\nCaso a coluna mencionada não devesse conter valores numéricos, este erro pode indicar que houve um erro anterior na definição de colunas obrigatórias das bases'



  df.name = nome_da_base
  # Retorna:
  # DataFrame com as colunas de valores formatadas em float
  # string formatada (pulando lunhas e com cores) da mensagem de erro ou aviso
  # inteiro representando o número de erros que esse check encontrou
  return df,mensagem,erro



#@title Def check_chaves

def check_chaves(lista_df,                   # lista de DataFrames já devem ter as colunas de valores formatadas
                 aberturas_compartilhadas,   # lista com as aberturas que devem estar presentes em todas as bases da lista de dataframes
                 aberturas_especificas,      # lista com as aberturas que não precisam estar presentes em todas as bases
                 lista_comparacao_parcial,   # lista de booleanos indicando quais bases serão checadas se contém todas as aberturas de todas as bases ou se contém aberturas que outras bases não tem
                 lista_comparacao_a_mais,
                 tipo_de_tof,
                 chaves_ignoradas,           # lista de chaves a serem ignoradas se encontradas, como chaves globais "Todos" por exemplo
                 nome_do_arquivo,
                 agrupar_duplicados):     
  

  # Definições iniciais
  #-------------------------------------------------------------------------------------------------
  mensagem = ''

  erro = 0

  lista_aberturas = []

  aberturas_compartilhadas = [e.lower() for e in aberturas_compartilhadas]
  aberturas_especificas = [e.lower() for e in aberturas_especificas]

  lista_df_atualizada = []
  lista_comparacao_parcial_atualizada = []
  lista_comparacao_a_mais_atualizada = []


  # Primeiro, vamos verificar se não existem aberturas duplicadas:
  #-------------------------------------------------------------------------------------------------
  c = 0
  for df in lista_df:

    if len(df) > 0:

      cabecalho = list(df.columns.values)
      nome = df.name

      chaves = []
      
      for d in aberturas_especificas:
        if d in cabecalho:
          chaves = chaves + [d]

      chaves = chaves+aberturas_compartilhadas
      chaves = list(dict.fromkeys(chaves)) # remover duplicados
      col_valores = list(set(cabecalho) - set(chaves))

      df_original = df.copy()
      df.name = nome

      # Agrupamos os valores das bases nas aberturas definidas
      if agrupar_duplicados:
        df = df.groupby(chaves, as_index=False)[col_valores].sum()

        df.name = nome

        df['idx'] = df.index
        df_original['idx'] = df_original.index

        merge = pd.merge(df_original[chaves+['idx']],df[chaves+['idx']],how='left',on=chaves)
        merge['aux'] = 1
        merge = merge[chaves+['idx_y','aux']]
        merge = merge.groupby(chaves+['idx_y'], as_index=False)[['aux']].sum()

        chaves_repetidas = merge.loc[merge['aux'] > 1][chaves]

        # Caso existam aberturas repetidas, atualizamos uma mensagem de aviso, mas não aumentamos a contagem de erros
        if len(chaves_repetidas) > 0:
          if len(chaves_repetidas) == len(df):
            mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo[c],'blue')+', na base '+colored(nome,'yellow')+' parece que todas as aberturas estão repetidas:' + '\n' +str(chaves)+'\nAs aberturas repetidas foram agrupadas e os valores foram somados.'
          elif len(chaves_repetidas) > 100:
            total_linhas = len(chaves_repetidas)
            chaves_repetidas = chaves_repetidas.head(100)
            mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo[c],'blue')+', na base '+colored(nome,'yellow')+' parece que existem as seguintes aberturas repetidas:' + '\n' + tabulate(chaves_repetidas, headers='keys', tablefmt='psql')+'\nMais outras '+str(total_linhas-100)+' linhas.\nAs aberturas repetidas foram agrupadas e os valores foram somados.'
          else:
            mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo[c],'blue')+', na base '+colored(nome,'yellow')+' parece que existem as seguintes aberturas repetidas:' + '\n' + tabulate(chaves_repetidas, headers='keys', tablefmt='psql')+'\nAs aberturas repetidas foram agrupadas e os valores foram somados.'
        df = df.drop(columns=['idx'])



      # Criamos bases contendo apenas as aberturas únicas encontradas:
      if len(aberturas_compartilhadas) == len(chaves):
        aberturas = df[aberturas_compartilhadas]
      else:
        aberturas = df[aberturas_compartilhadas+col_valores]
        aberturas = aberturas.groupby(aberturas_compartilhadas, as_index=False)[col_valores].sum()
        aberturas = aberturas[aberturas_compartilhadas]

      # Atualizamos a lista aberturas únicas por base
      lista_aberturas = lista_aberturas+[aberturas]
      df.name = nome
      lista_df_atualizada = lista_df_atualizada+[df]
      lista_comparacao_parcial_atualizada = lista_comparacao_parcial_atualizada + [lista_comparacao_parcial[c]]
      if len(lista_comparacao_a_mais) > 0:
        lista_comparacao_a_mais_atualizada = lista_comparacao_a_mais_atualizada + [lista_comparacao_a_mais[c]]

    c += 1
  

  # Vamos checar a compatibilidade de aberturas entre cada par possível de bases na lista de bases:
  #-------------------------------------------------------------------------------------------------

  # Criamos um lista contendo as combinações de índices possíveis de pares de bases.
  # Com a combinação de índices podemos selecionar as aberturas na lista de aberturas e as
  # bases na lista de bases, sabendo que ambas estão na mesma ordem. 
  indices = list(range(len(lista_df_atualizada)))
  combinacoes = list(itertools.combinations(indices, 2))

  # Vamos considerar apenas as comparações de bases com a base modelo. A base modelo será a
  # primeira base da lista de bases, que no caso do planning é a base de ToF mensal. Não vamos nos importar
  # com a compatibilidade das aberturas entre as outras bases. Vamos comparar as bases apenas com a base modelo:
  indice_base_modelo = 0
  if tipo_de_tof == 'Input Externo' and len(lista_df) > 3:
    if len(lista_df[2])>0:
      indice_base_modelo = 2
  combinacoes = [x for x in combinacoes if 0 in list(x)]

  # Para cada combinação de índices:
  for i in range(len(combinacoes)):

    indice_1 = list(combinacoes[i])[0]
    indice_2 = list(combinacoes[i])[1]

    nome_df_1 = lista_df_atualizada[indice_1].name
    nome_df_2 = lista_df_atualizada[indice_2].name

    aberturas_1 = lista_aberturas[indice_1]
    aberturas_2 = lista_aberturas[indice_2]

    nome_do_arquivo_1 = nome_do_arquivo[indice_1]
    nome_do_arquivo_2 = nome_do_arquivo[indice_2]

    # Verificamos a existência de chaves entre as bases abertura por abertura:
    for col in aberturas_compartilhadas:
      chaves_unicas_1 = list(np.unique(aberturas_1[col].values))
      chaves_unicas_2 = list(np.unique(aberturas_2[col].values))

      # vamos remover as chaves que devem ser ignoradas na comparação
      for c in chaves_ignoradas:
        if c in chaves_unicas_1:
          chaves_unicas_1.remove(c)
        if c in chaves_unicas_2:
          chaves_unicas_2.remove(c)

      chaves_1_2 = list(set(chaves_unicas_1) - set(chaves_unicas_2))
      chaves_2_1 = list(set(chaves_unicas_2) - set(chaves_unicas_1))

      if len(lista_comparacao_a_mais_atualizada) > 0:
        if len(chaves_1_2) > 0 and not lista_comparacao_parcial_atualizada[indice_2] and not lista_comparacao_a_mais_atualizada[indice_1]:
          mensagem = mensagem + '\n\nAs chaves '+colored(str(chaves_1_2),'red')+' da abertura '+colored(col,'red')+' da base '+colored(nome_df_1,'yellow')+' do arquivo ' + colored(nome_do_arquivo_1,'blue') + ' não estão presentes na base '+colored(nome_df_2,'yellow')+' do arquivo ' + colored(nome_do_arquivo_2,'blue')
          erro = erro+1
  
        if len(chaves_2_1) > 0 and not lista_comparacao_parcial_atualizada[indice_1] and not lista_comparacao_a_mais_atualizada[indice_2]:
          mensagem = mensagem + '\n\nAs chaves '+colored(str(chaves_2_1),'red')+' da abertura '+colored(col,'red')+' da base '+colored(nome_df_2,'yellow')+' do arquivo ' + colored(nome_do_arquivo_2,'blue') +  ' não estão presentes na base '+colored(nome_df_1,'yellow')+' do arquivo ' + colored(nome_do_arquivo_1,'blue') 
          erro = erro+1
      else:
        if len(chaves_1_2) > 0 and not lista_comparacao_parcial_atualizada[indice_2]:
          mensagem = mensagem + '\n\nAs chaves '+colored(str(chaves_1_2),'red')+' da abertura '+colored(col,'red')+' da base '+colored(nome_df_1,'yellow')+' do arquivo ' + colored(nome_do_arquivo_1,'blue') + ' não estão presentes na base '+colored(nome_df_2,'yellow')+' do arquivo ' + colored(nome_do_arquivo_2,'blue')
          erro = erro+1
  
        if len(chaves_2_1) > 0 and not lista_comparacao_parcial_atualizada[indice_1]:
          mensagem = mensagem + '\n\nAs chaves '+colored(str(chaves_2_1),'red')+' da abertura '+colored(col,'red')+' da base '+colored(nome_df_2,'yellow')+' do arquivo ' + colored(nome_do_arquivo_2,'blue') +  ' não estão presentes na base '+colored(nome_df_1,'yellow')+' do arquivo ' + colored(nome_do_arquivo_1,'blue') 
          erro = erro+1
          
    # Vamos exluir as aberturas que contenham uma chave a ser ignorada:
    for c in chaves_ignoradas:
      for coluna in aberturas_compartilhadas:
        aberturas_1 = aberturas_1.loc[aberturas_1[coluna] != c]
        aberturas_2 = aberturas_2.loc[aberturas_2[coluna] != c]

    aberturas_1['aux'] = 1
    aberturas_2['aux'] = 1

    merge = pd.merge(aberturas_1,aberturas_2,how='outer',on=aberturas_compartilhadas)

    aberturas_nao_existentes_1 = merge.loc[merge['aux_x'].isnull()][aberturas_compartilhadas]
    aberturas_nao_existentes_2 = merge.loc[merge['aux_y'].isnull()][aberturas_compartilhadas]


    

    # Verificamos as combinações de chaves entre as bases:
    if len(lista_comparacao_a_mais_atualizada) > 0:
      if len(aberturas_nao_existentes_1) > 0 and not lista_comparacao_parcial_atualizada[indice_1] and not lista_comparacao_a_mais_atualizada[indice_2]:
        mensagem = mensagem + '\n\nAs seguintes combinações de aberturas da base '+colored(nome_df_2,'yellow')+' do arquivo ' + colored(nome_do_arquivo_2,'blue') + ' não estão presentes na base '+colored(nome_df_1,'yellow')+' do arquivo ' + colored(nome_do_arquivo_1,'blue') +  ': \n' + tabulate(aberturas_nao_existentes_1, headers='keys', tablefmt='psql')
        erro = erro+1
      if len(aberturas_nao_existentes_2) > 0 and not lista_comparacao_parcial_atualizada[indice_2] and not lista_comparacao_a_mais_atualizada[indice_1]:
        mensagem = mensagem + '\n\nAs seguintes combinações de aberturas da base '+colored(nome_df_1,'yellow')+' do arquivo ' + colored(nome_do_arquivo_1,'blue') +  ' não estão presentes na base '+colored(nome_df_2,'yellow')+' do arquivo ' + colored(nome_do_arquivo_2,'blue') +  ': \n' + tabulate(aberturas_nao_existentes_2, headers='keys', tablefmt='psql')
        erro = erro+1
    else:
      if len(aberturas_nao_existentes_1) > 0 and not lista_comparacao_parcial_atualizada[indice_1]:
        mensagem = mensagem + '\n\nAs seguintes combinações de aberturas da base '+colored(nome_df_2,'yellow')+' do arquivo ' + colored(nome_do_arquivo_2,'blue') + ' não estão presentes na base '+colored(nome_df_1,'yellow')+' do arquivo ' + colored(nome_do_arquivo_1,'blue') +  ': \n' + tabulate(aberturas_nao_existentes_1, headers='keys', tablefmt='psql')
        erro = erro+1
      if len(aberturas_nao_existentes_2) > 0 and not lista_comparacao_parcial_atualizada[indice_2]:
        mensagem = mensagem + '\n\nAs seguintes combinações de aberturas da base '+colored(nome_df_1,'yellow')+' do arquivo ' + colored(nome_do_arquivo_1,'blue') +  ' não estão presentes na base '+colored(nome_df_2,'yellow')+' do arquivo ' + colored(nome_do_arquivo_2,'blue') +  ': \n' + tabulate(aberturas_nao_existentes_2, headers='keys', tablefmt='psql')
        erro = erro+1      
      
  # Retorna:
  # lista de DataFrames 
  # string formatada (pulando lunhas e com cores) da mensagem de erro ou aviso
  # inteiro representando o número de erros que esse check encontrou
  return lista_df,mensagem,erro



#@title Def check_colunas_bases_especificas


def check_colunas_bases_especificas(nome_do_arquivo,
                                      df_on_top_ratios,
                                      df_share_diario,
                                      df_impacto_feraidos,
                                      df_city_share,
                                      aberturas,
                                      flag_gerar_city_share,
                                      col_valores_on_top,
                                      col_valores_share,
                                      col_valores_feriados,
                                      col_valores_city_share,
                                      etapas_volume):
  

  # Definições iniciais
  #-------------------------------------------------------------------------------------------------
  mensagem = ''

  erro = 0

  # Verificamos se existe "região" nas aberturas e se existe uma base de share de cidades
  aberturas_lower = [e.lower() for e in aberturas]
  if 'região' in aberturas_lower and len(df_city_share) == 0 and not flag_gerar_city_share:
    erro+=1
    mensagem=mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo[3],'blue')+' parece que não existe uma base de '+colored('City Share','yellow')+'. Esta base é obrigatória se houver uma abertura '+colored('"região"','red')

  if len(df_on_top_ratios) != 0:
    col_valores_on_top_originais = col_valores_on_top.copy()
    col_valores_on_top = [e.lower() for e in col_valores_on_top]
  col_valores_share = [e.lower() for e in col_valores_share]
  col_valores_feriados = [e.lower() for e in col_valores_feriados]
  col_valores_city_share = [e.lower() for e in col_valores_city_share]

  lista_colunas = [col_valores_on_top,col_valores_share,col_valores_feriados,col_valores_city_share]
  lista_df = [df_on_top_ratios,df_share_diario,df_impacto_feraidos,df_city_share]

  # Organizando a existência das bases:
  lista_existe = [len(col_valores_on_top),len(col_valores_share),len(col_valores_feriados),len(col_valores_city_share)]
  for x in range(len(lista_existe)):
    if lista_existe[x] > 0:
      lista_existe[x] = True
    else:
      lista_existe[x] = False

  lista_colunas = list(compress(lista_colunas[1:],lista_existe[1:]))
  lista_df  = list(compress(lista_df[1:],lista_existe[1:]))
  nomes = [x.name for x in lista_df]
  nomes_dos_arquivos = list(compress(nome_do_arquivo[1:],lista_existe[1:]))

  ordem_final = etapas_volume.copy()

  if len(col_valores_on_top) == 0:
    n=0
    for coluna in lista_colunas:
      diferenca_1 = list(set(coluna)-set(etapas_volume))
      diferenca_2 = list(set(etapas_volume)-set(coluna))
      if len(diferenca_1) != 0 or len(diferenca_2) != 0:
        erro = erro+1
        mensagem=mensagem+'\n\nNo arquivo '+colored(nomes_dos_arquivos[n],'blue')+' a base '+colored(str(nomes[n]),'yellow')+' não contém colunas de volume diferentes das obrigatórias. \nAs colunas de volume obrigatórias são: '+colored(str(etapas_volume),'red')+'.\nAs colunas de volume encontradas na base são: '+colored(str(coluna),'red')+'.\nComo não foi declarada uma base de "On Top Ratios", é necessário que todas as colunas de volumes sejam idêndicas às etapas presentes na lista de conversões declarada no painel de controle.'       
      n += 1

  # Checar se todas as colunas de valores são as mesmas entre as bases de share diário, impacto de feriados
  # e share de cidades:
  #-------------------------------------------------------------------------------------------------
  if erro == 0 and len(lista_colunas) > 1:
      
    # Criamos uma função auxiliar que compara as colunas entre duas bases e gera mensagens de erro:
    #-------------------------------------------------------------------------------------------------
    def aux_check(df1,
                  df2,
                  col1,
                  col2,
                  nome_do_arquivo):
      
      erro_aux = 0

      mensagem_aux = ''

      colunas_1_2 = list(set(col1) - set(col2))
      colunas_2_1 = list(set(col2) - set(col1))

      if len(colunas_1_2) > 0:
        erro_aux = erro_aux+1
        mensagem_aux='\n\nA base '+colored(str(df1.name),'yellow')+' no arquivo '+colored(nome_do_arquivo[0],'blue')+' contém as seguintes colunas que não existem na base '+colored(str(df2.name),'yellow')+' no arquivo '+colored(nome_do_arquivo[1],'blue')+':\n'+colored(str(colunas_1_2),'red')+'\nAs colunas de valores devem ser as mesmas entre essas bases.'
      if len(colunas_2_1) > 0:
        erro_aux = erro_aux+1
        mensagem_aux='\n\nA base '+colored(str(df2.name),'yellow')+' no arquivo '+colored(nome_do_arquivo[1],'blue')+' contém as seguintes colunas que não existem na base '+colored(str(df1.name),'yellow')+' no arquivo '+colored(nome_do_arquivo[0],'blue')+':\n'+colored(str(colunas_2_1),'red')+'\nAs colunas de valores devem ser as mesmas entre essas bases.'

      return erro_aux,mensagem_aux
    #-------------------------------------------------------------------------------------------------

    
    # Vamos gerar uma lista com as combinações de pares de índices das listas:
    indices = list(range(len(lista_df)))
    combinacoes = list(itertools.combinations(indices, 2))

    # Para cada combinação entre duas bases, vamos realizar a verificação das colunas:
    for i in range(len(combinacoes)):

      indice_1 = list(combinacoes[i])[0]
      indice_2 = list(combinacoes[i])[1]
      lista_nomes_dos_arquivos = [nomes_dos_arquivos[indice_1],nomes_dos_arquivos[indice_2]]

      erro_out,mensagem_out = aux_check(df1 = lista_df[indice_1],
                                        df2 = lista_df[indice_2],
                                        col1 = lista_colunas[indice_1],
                                        col2 = lista_colunas[indice_2],
                                        nome_do_arquivo = lista_nomes_dos_arquivos)
      
      erro = erro+erro_out
      mensagem = mensagem+mensagem_out



    # Verificar se as colunas de valores obrigatorias existem nas colunas de valores de todas as bases
    #-------------------------------------------------------------------------------------------------
    if erro == 0: # caso as bases contenham as mesmas colunas de valores:

      i=0
      for colunas in lista_colunas:

        colunas_faltantes = list(set(etapas_volume) - set(colunas))

        if len(colunas_faltantes) > 0:
          erro = erro+1
          mensagem=mensagem+'\n\nNo arquivo '+colored(nomes_dos_arquivos[i],'blue')+' a base '+colored(str(lista_df[i].name),'yellow')+' não contém todas as colunas de volume obrigatórias. \nAs colunas que estão faltando na base são: '+colored(str(colunas_faltantes),'red')
        i=i+1
  

    # Ordenar as colunas de valores com base na posição das colunas extra do share diário.
    #-------------------------------------------------------------------------------------------------
    if erro == 0:

      if len(col_valores_share) == 0:
        col_valores_modelo = col_valores_feriados
        nome_modelo = df_impacto_feraidos.name
        nome_do_arquivo_modelo = nome_do_arquivo[2]
      else:
        col_valores_modelo = col_valores_share
        nome_modelo = df_share_diario.name
        nome_do_arquivo_modelo = nome_do_arquivo[1]

      # Determinar a ordem final das colunas
      col_anterior = []

      colunas_extra = list(set(col_valores_modelo) - set(etapas_volume))

      if len(colunas_extra) > 0:
        # Precisamos ordenar as colunas extra da forma como aparecem na base:
        colunas_extra_aux = colunas_extra
        colunas_extra = []
        for col in col_valores_modelo:
          if col in colunas_extra_aux:
            colunas_extra = colunas_extra+[col]

        for col_extra in colunas_extra:

          # Caso a coluna extra na base de share diário seja a primeira que aparece, retornamos um erro:
          index_anterior = col_valores_modelo.index(col_extra)-1
          if index_anterior < 0:
            erro = erro+1
            mensagem=mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo_modelo,'blue')+' a base '+colored(str(nome_modelo),'yellow')+' possui uma coluna de volume que não existe nas colunas obrigatorias e é a primeira que aparece na base, indicando, pela sua ordem, que é anterior à primeira etapa do funil. \nA coluna em questão é: '+colored(str(col_valores_share[0]),'red')+'\nA base '+colored(str(df_share_diario.name),'yellow')+' é usada como referência para a ordem das etapas de volume que não constam no funil cohort.'
          else:
            col_anterior = col_anterior+[col_valores_modelo[index_anterior]]
        
        i=0
        if erro == 0:
          for col_a in col_anterior:

            index_col_anterior = ordem_final.index(col_a)

            ordem_final_1 = ordem_final[:index_col_anterior+1]
            ordem_final_2 = ordem_final[index_col_anterior+1:]
            ordem_final = ordem_final_1+[colunas_extra[i]]+ordem_final_2
            i=i+1


      # Ordenar as colunas das bases:
      if erro == 0:
        for df in lista_df:
          cb = list(df.columns.values)
          chaves = list(set(cb) - set(ordem_final))
          df = df[chaves+ordem_final]



    
  # Checar as colunas da base on_top_ratios
  #-------------------------------------------------------------------------------------------------


  if len(df_on_top_ratios) == 0 and erro == 0:
    nome_da_base = 'Racional On Top Ratio'
    tipo_de_base = ' os inputs no Painel de Controle '
    nome_do_arquivo_modelo = nome_do_arquivo[0]

    col_valores_on_top_formatada = []
    try:
      lista_racional_on_top_1 = col_valores_on_top.split(',')
      for l in lista_racional_on_top_1:
        l = l.strip()
        l = l.split('|')
        if len(l)>0:
          l = l[0].strip()
        col_valores_on_top_formatada = col_valores_on_top_formatada + [l]
      col_valores_on_top = col_valores_on_top_formatada
    except:
      col_valores_on_top = col_valores_on_top

  else:
    nome_da_base = df_on_top_ratios.name
    tipo_de_base = ' as colunas de valores da base '



  etapas_on_top = []
  for c in col_valores_on_top:

    if '/' not in c:
      mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo_modelo,'blue')+tipo_de_base+colored(str(nome_da_base),'yellow')+' devem todas conter o caractere "/" para separar as etapas. \nColunas de valores encontradas: n\ '+colored(str(col_valores_on_top),'red')
      erro = erro+1
    
    if erro == 0:
      etapa_existente = c.split('/')[0]
      etapa_nova = c.split('/')[1]

      if etapa_existente not in etapas_volume:
        mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo_modelo,'blue')+tipo_de_base+colored(str(nome_da_base),'yellow')+' devem iniciar com uma etapa cujos valores podem e já foram gerados pelas conversões cohort. \nA primeira etapa na coluna '+colored(str(c),'red')+' é '+colored(str(etapa_existente),'red')+'. Esta etapa não consta na lista de etapas possíveis de serem calculadas pelo funil via cohort: '+colored(str(etapas_volume),'red')
        erro = erro+1


  # renomear as colunas do on top rations para tudo minúsculo
  if erro == 0:
    if len(df_on_top_ratios) != 0:
      df_on_top_ratios = df_on_top_ratios.rename(columns=dict(zip(col_valores_on_top_originais, col_valores_on_top)))

  return ordem_final,df_on_top_ratios,df_share_diario,df_impacto_feraidos,df_city_share,mensagem,erro



#@title Def check_datas

def ajusta_formato_data(nome_do_arquivo, dataframe,lista_colunas_datas):

  contagem_de_erros = 0
  mensagem = ''

  lista_colunas_datas = [e.lower() for e in lista_colunas_datas]

  for coluna in lista_colunas_datas:
    try:
      dataframe[coluna] = pd.to_datetime(dataframe[coluna],infer_datetime_format=True)
    except(ValueError):
      mensagem = f'\n \nNão foi possível converter a coluna \033[1;33m"{coluna}"\033[0;0;0m da base \033[1;33m"{dataframe.name}"\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m para o formato de data. Por favor verifique se todos os os dados presentes nessa coluna estão com o formato correto'
      contagem_de_erros += 1
  
  return dataframe, contagem_de_erros, mensagem

#-------------------------------------------------------------------------------

def verifica_frequencia_datas(nome_do_arquivo, dataframe, lista_colunas_datas, lista_frequencia):
  contagem_de_erros = 0
  mensagem = ''
  
  lista_colunas_datas = [e.lower() for e in lista_colunas_datas]

  dataframe_copy = dataframe.copy()

  i=0
  for coluna in lista_colunas_datas:
    if lista_frequencia[i] != 'Período Mensal':
      try:
        freq = pd.infer_freq(sorted(dataframe[coluna].unique()))
        if freq != lista_frequencia[i] and lista_frequencia[i] != 'None':
          mensagem = f'\n \nNão foi possível inferir frequência semanal de segundas feiras nas datas da coluna \033[1;33m"{coluna}"\033[0;0;0m da base \033[1;33m"{dataframe.name}"\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m. Por favor verifique se os dados desta coluna estão corretos'
          contagem_de_erros += 1
      
      except(ValueError):
        mensagem = f'\n \nExistem menos de 3 datas distintas na coluna \033[1;33m"{coluna}"\033[0;0;0m da base \033[1;33m"{dataframe.name}"\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m. Não foi possível inferir frequência de datas'
        contagem_de_erros += 1
    else:
      dataframe_copy['dia'] = dataframe_copy[coluna].dt.day
      dataframe_copy['dia'] = dataframe_copy['dia'].astype(int)
      if dataframe_copy['dia'].max() != 1:
        mensagem = f'\n \nNão foi possível inferir frequência mensal (datas com somente o primeiro dia de cada mês) nas datas da coluna \033[1;33m"{coluna}"\033[0;0;0m da base \033[1;33m"{dataframe.name}"\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m. Por favor verifique se os dados desta coluna estão corretos'
        contagem_de_erros += 1        
    i += 1
    
  return dataframe, contagem_de_erros, mensagem

#-------------------------------------------------------------------------------

def verifica_datas_cabecalho(nome_do_arquivo, dataframe, lista_colunas_datas):
  aberturas_aux = list(set(dataframe.columns)-set(lista_colunas_datas))
  # Reordenamos as aberturas na ordem que aparecem na base:
  aberturas = [a for a in dataframe.columns.values if a in aberturas_aux]

  datas_formato_antigo = lista_colunas_datas.copy()
  contagem_de_erros = 0
  mensagem = ''

  try:
    lista_colunas_datas = pd.to_datetime(lista_colunas_datas,infer_datetime_format=True)
  except(ValueError):
    mensagem = f'\n \nNão foi possível converter as datas {lista_colunas_datas} da base \033[1;33m"{dataframe.name}"\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m para o formato de data. Por favor verifique se os dados estão corretos'
    contagem_de_erros += 1
    return dataframe, contagem_de_erros, mensagem

  if len(lista_colunas_datas) != len(list(set(lista_colunas_datas))):
    mensagem = f'\n \nExistem datas duplicadas nas colunas da base \033[1;33m"{dataframe.name}\033[0;0;0m" do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m'
    contagem_de_erros += 1
    return dataframe, contagem_de_erros, mensagem

  
  try:
    freq = pd.infer_freq(sorted(lista_colunas_datas))
    if freq != 'W-MON':
      mensagem = f'\n \nNão foi possível inferir frequência semanal de segundas feiras nas colunas de datas na base \033[1;33m"{dataframe.name}"\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m. Por favor verifique se os dados estão corretos'
      contagem_de_erros += 1
      return dataframe, contagem_de_erros, mensagem

  except(ValueError):
    mensagem = f'\n \nExistem menos de 3 datas distintas nas colunas de datas da base \033[1;33m"{dataframe.name}"\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m. Não foi possível inferir frequência de datas'
    contagem_de_erros += 1
    return dataframe, contagem_de_erros, mensagem

  if contagem_de_erros == 0:
    df_aux_1 = dataframe[aberturas]
    df_aux_2 = dataframe[datas_formato_antigo]
    df_aux_2.columns = pd.to_datetime(df_aux_2.columns)

    df_output = pd.concat([df_aux_1,df_aux_2], axis=1)
    df_output.name = dataframe.name

    return df_output, contagem_de_erros, mensagem

#-------------------------------------------------------------------------------
def confere_datas_inputs_tof(nome_do_arquivo,lista_datas_inputs, df_tof_semanal, nome_coluna_datas):
  contagem_de_erros = 0
  mensagem = ''
  #print(sorted(pd.to_datetime(lista_datas_inputs,infer_datetime_format=True).unique()),sorted(df_tof_semanal[nome_coluna_datas.lower()].unique()))
  if sorted(pd.to_datetime(lista_datas_inputs,infer_datetime_format=True).unique()) != sorted(df_tof_semanal[nome_coluna_datas.lower()].unique()):
    contagem_de_erros += 1
    mensagem = f'\n \nAs datas da base de \033[1;33minputs\033[0;0;0m não são as mesmas da base de \033[1;33mTOF Semanal\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m'
  
  return mensagem, contagem_de_erros


#@title Def confere_share_diario

def confere_share_diario(nome_do_arquivo, dataframe, lista_chaves, lista_etapas, nome_coluna_weekday):


  dataframe.columns = [e.lower() for e in list(dataframe.columns)]
  lista_chaves = [e.lower() for e in lista_chaves]
  lista_etapas = [e.lower() for e in lista_etapas]
  nome_coluna_weekday = nome_coluna_weekday.lower()

  #for etapa in lista_etapas:
  #  dataframe[etapa] = dataframe[etapa].astype('float')

  lista_check_dias = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

  contagem_de_erros = 0
  mensagem = ''

  if sorted(dataframe[nome_coluna_weekday].unique()) != sorted(lista_check_dias):
    contagem_de_erros += 1
    mensagem = f'\n\n Os dias na coluna \033[1;33m"{nome_coluna_weekday}"\033[0;0;0m da base de \033[1;33mShare Diário\033[0;0;0m no arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m não seguem o padrão {lista_check_dias}'
    return dataframe, contagem_de_erros, mensagem

  df_group = dataframe.groupby(lista_chaves, as_index=False)[lista_etapas].sum()

  for etapa in lista_etapas:
    df_group[etapa] = df_group[etapa].round(3)
    aux = df_group[df_group[etapa] != 1.0]
    if len(aux.index) > 0:
      contagem_de_erros += 1
      mensagem = mensagem + f'\n\n Os valores da coluna \033[1;33m"{etapa}"\033[0;0;0m na base de \033[1;33mShare diário\033[0;0;0m no arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m não somam 100%. Por favor verifique os dados inputados \n{tabulate(aux, headers=list(aux.columns), tablefmt="psql")}'
      break

  return dataframe, contagem_de_erros, mensagem
#-------------------------------------------------------------------------------



# @title Def check_valores_negativos
def check_valores_negativos(dataframe, lista_colunas_numericas):
  '''
  Esta função parte do princípio que os valores nas colunas de números já foram 
  passados para float
  '''
  dataframe_teste = dataframe.copy()

  contagem_de_erros = 0
  mensagem = ''

  lista_colunas_numericas = [e.lower() for e in lista_colunas_numericas]

  # Verifica se a base é uma base de baseline de cohorts, que pode conter valores negativos na conversão de ajuste "Coincident":
  coluna_coincident = []
  for col in list(dataframe_teste.columns.values):
    try:
      if dataframe_teste[col].str.contains('Coincident').any():
        coluna_coincident = coluna_coincident + [col]
    except:
      pass
  
  # Verifica se existe somente 1 coluna com a conversão coincident:
  if len(coluna_coincident) > 1:
    mensagem = f'\n\n Existe mais de uma coluna com conversões de ajuste na base \033[1;33m"{dataframe.name}"\033[0;0;0m. Por favor verifique os dados inputados'
    contagem_de_erros += 1    
    coluna_coincident[0]
  # Caso exista a coluna de conversão coincident, não checar se os valores são negativos
  elif len(coluna_coincident) == 1:
    coluna_coincident = coluna_coincident[0]
    dataframe_teste = dataframe_teste.loc[dataframe_teste[coluna_coincident] != 'Coincident']



  for coluna in lista_colunas_numericas:
    aux = dataframe_teste[dataframe_teste[coluna] < 0]
    if len(aux.index) > 0:
      mensagem = f'\n\n Existem valores negativos na coluna \033[1;33m"{coluna}"\033[0;0;0m na base \033[1;33m"{dataframe.name}"\033[0;0;0m. Por favor verifique os dados inputados'
      contagem_de_erros += 1
  
  return contagem_de_erros, mensagem



# @title Def confere_aberturas_zerados

def confere_aberturas_zerados(nome_do_arquivo, df_tof, df_baseline, lista_chaves, lista_topos, lista_etapas):
  '''
  Esta função parte do princípio que os valores nas colunas de números já foram 
  passados para float
  '''
  df_tof_c = df_tof.copy()
  df_baseline_c = df_baseline.copy()

  df_tof_c.columns = [e.lower() for e in list(df_tof_c.columns)]
  df_baseline_c.columns = [e.lower() for e in list(df_baseline_c.columns)]
  lista_chaves = [e.lower() for e in lista_chaves]
  lista_etapas = [e.lower() for e in lista_etapas]
  lista_topos = [e.lower() for e in lista_topos]

  contagem_de_erros = 0
  mensagem = ''

  #for topo in lista_topos:
  #  df_tof[topo] = df_tof[topo].astype('float')

  #for etapa in lista_etapas:
  #  df_baseline[etapa] = df_baseline[etapa].astype('float')

  df_tof_c['sum'] = df_tof_c[lista_topos].sum(axis=1)
  df_tof_group = df_tof_c.groupby(lista_chaves,as_index=False)['sum'].sum()
  df_tof_group.loc[(df_tof_group['sum'] != 0), ['sum']] = 1

  df_baseline_group = df_baseline_c.groupby(lista_chaves,as_index=False)[lista_etapas].sum()
  
  df_merge = pd.merge(df_baseline_group, df_tof_group, how='left', on=lista_chaves)
  
  for etapa in lista_etapas:
    df_merge.loc[(df_merge[etapa] != 0), [etapa]] = 1
    df_merge['verifica'] = df_merge[etapa] - df_merge['sum']

    df_aux_1 = df_merge[df_merge['verifica'] == -1]
    df_aux_2 = df_merge[df_merge['verifica'] == 1]
    
    if len(df_aux_1.index) > 0:
      df_aux_1 = df_aux_1.drop(columns=['verifica','sum']+lista_etapas)
      mensagem = mensagem + f'\n\n Atenção! No arquivo \033[1;34m{nome_do_arquivo[0]}\033[0;0;0m existem valores de \033[1;33mToF\033[0;0;0m porém, no arquivo \033[1;34m{nome_do_arquivo[1]}\033[0;0;0m o \033[1;33mbaseline\033[0;0;0m está zerado.\n {tabulate(df_aux_1, headers=list(df_aux_1.columns), tablefmt="psql")}' #adicionar na mensagem as chaves do dataframe aux #tabulate(aberturas_nao_existentes_1, headers='keys', tablefmt='psql')
      contagem_de_erros += 1
      break
    if len(df_aux_2.index) > 0:
      df_aux_2 = df_aux_2.drop(columns=['verifica','sum']+lista_etapas)
      mensagem = mensagem + f'\n\n Atenção! No arquivo \033[1;34m{nome_do_arquivo[1]}\033[0;0;0m existem valores de \033[1;33mbaseline\033[0;0;0m porém, no arquivo \033[1;34m{nome_do_arquivo[0]}\033[0;0;0m o \033[1;33mToF\033[0;0;0m está zerado no arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m.\n {tabulate(df_aux_2, headers=list(df_aux_2.columns), tablefmt="psql")}'
      contagem_de_erros += 1
      break

  return mensagem, contagem_de_erros



# @title Def check_city_share


def check_city_share(df_city_share,
                     df_baseline_conversoes,
                     col_valores,
                     nome_do_arquivo):
  

  mensagem = ''
  erro = 0


  # Agrupar a base pelas regiões e somar as colunas de valores:
  city_share_agrupada = df_city_share.groupby('região', as_index=False)[col_valores].sum()

  # Para cada coluna de valor, selecionar as regiões que passaram de 100%:
  for e in col_valores:
    city_share_agrupada[e] = city_share_agrupada[e].round(3)
    share_errado = city_share_agrupada.loc[city_share_agrupada[e] != 1,['região']].values
    if len(share_errado) > 0:
      mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', na base '+colored(df_city_share.name,'yellow')+', as proporções das cidades somadas é diferente de 100% nas seguintes regiões na etapa '+colored(e,'red')+':\n'+colored(str(np.unique(share_errado)),'red')
      erro=erro+1

  
  # Verificar se alguma das regiões da base de city share existe em alguma coluna de abertura da base de conversões:
  regioes = list(np.unique(city_share_agrupada['região'].values))
  colunas_com_regiao = []
  col_regiao = ''
  for e in list(df_baseline_conversoes.columns.values):
    if any(x in list(df_baseline_conversoes[e].values) for x in regioes):
      colunas_com_regiao = colunas_com_regiao + [e]
  
  # Se o número de colunas que contém regiões for maior do que 1 ou não tiver nenhuma, retornar um erro:
  if len(colunas_com_regiao) == 0:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', nenhuma região da coluna de regiões da base '+colored(df_city_share.name,'yellow')+' foi encontrada em alguma coluna da base '+colored(df_baseline_conversoes.name,'yellow')+'.\nPara aplicar efeitos de feriado é necessário que exista alguma abertura de cidades/regiões nas outras bases.'
    erro=erro+1  
  elif len(colunas_com_regiao) > 1:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', foi encontrada mais de uma coluna da base '+colored(df_baseline_conversoes.name,'yellow')+' que contém nomes de regiões da base '+colored(df_city_share.name,'yellow')+'.\nPara aplicar efeitos de feriado é necessário que exista somente uma coluna com abertura de cidades/regiões nas outras bases.\nAs colunas com nomes de regiões que foram encontradas são:\n'+colored(str(colunas_com_regiao),'red')
    erro=erro+1 
  else:
    col_regiao = colunas_com_regiao[0]

  return col_regiao,mensagem,erro



# @title Def check_feriados




def check_feriados(df_feriados,
                   df_impacto,
                   df_tof_mensal,
                   nome_do_arquivo):
  
  mensagem = ''
  erro = 0

  # Verificar o conteúdo das colunas da base de feriados
  #-------------------------------------------------------------------------------------------------

  # Verificar os tipos de feriados
  tipos_de_feriados = ['N','E','M']
  tipos_de_feriados_base = list(np.unique(df_feriados['tipo'].values))

  diferenca1 = list(set(tipos_de_feriados_base) - set(tipos_de_feriados))
  diferenca2 = list(set(tipos_de_feriados) - set(tipos_de_feriados_base))

  if len(diferenca1) != 0:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', os tipos de feriado na base '+colored(df_feriados.name,'yellow')+' estão errados. Os tipos de feriados devem ser exatamente estes: '+str(tipos_de_feriados)+'.\nOs tipos faltantes ou errados foram: '+colored(str(diferenca1),'red')
    erro += 1
  if len(diferenca2) != 0:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', os tipos de feriado na base '+colored(df_feriados.name,'yellow')+' estão errados. Os tipos de feriados devem ser exatamente estes: '+str(tipos_de_feriados)+'.\nOs tipos faltantes ou errados foram: '+colored(str(diferenca2),'red')
    erro += 1

  # Verificar se os tipos de feriados correspondem com as colunas de estado e cidade
  if erro == 0:

    for t in tipos_de_feriados:
      base_filtrada = df_feriados.loc[df_feriados['tipo'] == t]
      estados = list(np.unique(base_filtrada['estado'].values))
      regioes = list(np.unique(base_filtrada['região'].values))
      cidades = list(np.unique(base_filtrada['cidade'].values))

      if t == 'N':
        if len(estados) > 1 or estados[0] != '' or len(regioes) > 1 or regioes[0] != '' or len(cidades) > 1 or cidades[0] != '':
          mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', os tipos de feriado '+colored(t,'red')+' da coluna '+colored('tipo','red')+' na base '+colored(df_feriados.name,'yellow')+' não estão condizentes com as colunas '+colored(str(['estado','região','cidade']),'red')
          erro += 1          
      if t == 'E':
        if any(x == '' for x in estados) or len(regioes) > 1 or regioes[0] != '' or len(cidades) > 1 or cidades[0] != '':
          mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', os tipos de feriado '+colored(t,'red')+' da coluna '+colored('tipo','red')+' na base '+colored(df_feriados.name,'yellow')+' não estão condizentes com as colunas '+colored(str(['estado','região','cidade']),'red')
          erro += 1  
      if t == 'M':
        if len(estados) > 1 or estados[0] != '' or any(x == '' for x in regioes) or any(x == '' for x in cidades):
          mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', os tipos de feriado '+colored(t,'red')+' da coluna '+colored('tipo','red')+' na base '+colored(df_feriados.name,'yellow')+' não estão condizentes com as colunas '+colored(str(['estado','região','cidade']),'red')
          erro += 1  

  # Encontrar os dias da semana na base de feriados
  dias_da_semana_feriados = list(np.unique(df_impacto['dia da semana'].values))
  dias_da_semana = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

  diferenca1 = list(set(dias_da_semana_feriados) - set(dias_da_semana))

  if len(diferenca1) != 0:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', os dias da semana na base '+colored(df_feriados.name,'yellow')+' estão errados. Os dias da semana devem ser exatamente estes: '+str(dias_da_semana)+'.\nOs dias faltantes ou errados foram: '+colored(str(diferenca1),'red')
    erro += 1
  
  
  # Comparar datas da base ToF com as datas de feriados
  #-------------------------------------------------------------------------------------------------

  # Vamos encontrar o número de meses na base tof e as últimas e primeiras datas:
  ano = df_tof_mensal['ano'].astype(int).max()
  mes = df_tof_mensal.loc[df_tof_mensal['ano'] == str(ano),['mês']].astype(int).max().values[0]
  if mes < 10:
    zero = '0'
  else:
    zero = ''
  date_time_str = str(mes)+'/01/'+str(ano)
  ultima_data_tof = datetime.strptime(date_time_str, '%m/%d/%Y')

  ano = df_tof_mensal['ano'].astype(int).min()
  mes = df_tof_mensal.loc[df_tof_mensal['ano'] == str(ano),['mês']].astype(int).min().values[0]
  if mes < 10:
    zero = '0'
  else:
    zero = ''
  date_time_str = str(mes)+'/01/'+str(ano)
  primeira_data_tof = datetime.strptime(date_time_str, '%m/%d/%Y')


  if ultima_data_tof.year == primeira_data_tof.year:
    numero_de_meses_tof = ultima_data_tof.month - primeira_data_tof.month + 1
  else:
    years = ultima_data_tof.year - ultima_data_tof.year + 1
    numero_de_meses_tof = ultima_data_tof.month + (12 - primeira_data_tof.month) + 12*years



  # Vamos encontrar o número de meses na base de feriados e as últimas e primeiras datas: 
  ultima_data_feriados = df_feriados['data'].max()
  primeira_data_feriados = df_feriados['data'].min()

  if ultima_data_feriados.year == primeira_data_feriados.year:
    numero_de_meses_feriados = ultima_data_feriados.month - primeira_data_feriados.month + 1
  else:
    years = ultima_data_feriados.year - ultima_data_feriados.year + 1
    numero_de_meses_feriados = ultima_data_feriados.month + (12 - primeira_data_feriados.month) + 12*years



  # Verificar se os meses de tof estão contemplados nos feriados:
  if primeira_data_tof < primeira_data_feriados:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', a primeira data da base '+colored(df_tof_mensal.name,'yellow')+' ('+str(primeira_data_tof)+'), é anterior à primeira data da base '+colored(df_feriados.name,'yellow')+' ('+str(primeira_data_feriados)+').'
  if ultima_data_tof > ultima_data_feriados:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', a ultima data da base '+colored(df_tof_mensal.name,'yellow')+' ('+str(ultima_data_tof)+'), é posterior à ultima data da base '+colored(df_feriados.name,'yellow')+' ('+str(ultima_data_feriados)+').'
  if numero_de_meses_feriados < numero_de_meses_tof:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', parece que existem mais meses na base '+colored(df_tof_mensal.name,'yellow')+' do que meses na base '+colored(df_feriados.name,'yellow')+'.\nIsso pode significar que a base de feriados está incompleta em relação às datas da base de ToF'



  # Encontrar os dias da semana na base de impactos de feriados
  #-------------------------------------------------------------------------------------------------
  dias_da_semana_impacto = list(df_impacto['dia da semana'].values)
  dias_da_semana = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

  diferenca1 = list(set(dias_da_semana_impacto) - set(dias_da_semana))
  diferenca2 = list(set(dias_da_semana) - set(dias_da_semana_impacto))

  if len(diferenca1) != 0:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', os dias da semana na base '+colored(df_impacto.name,'yellow')+' estão errados. Os dias da semana devem ser exatamente estes: '+str(dias_da_semana)+'.\nOs dias faltantes ou errados foram: '+colored(str(diferenca1),'red')
    erro += 1
  if len(diferenca2) != 0:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', os dias da semana na base '+colored(df_impacto.name,'yellow')+' estão errados. Os dias da semana devem ser exatamente estes: '+str(dias_da_semana)+'.\nOs dias faltantes ou errados foram: '+colored(str(diferenca2),'red')
    erro += 1

  return mensagem,erro




# @title Def verifica_baseline_100% 

def verifica_baseline(nome_do_arquivo,df_baseline,lista_chaves,colunas_conversoes,coluna_idx_cohort='week origin'):

  mensagem = ''
  contagem_de_erros = 0
  
  df_baseline.columns = [e.lower() for e in list(df_baseline.columns)]
  lista_chaves = [e.lower() for e in lista_chaves]
  colunas_conversoes = [e.lower() for e in colunas_conversoes]

  #for etapa in colunas_conversoes:
  #  df_baseline[etapa] = df_baseline[etapa].astype('float')

  df_coincident = df_baseline[df_baseline[coluna_idx_cohort] == 'Coincident']
  df_aux_1 = df_baseline[df_baseline[coluna_idx_cohort] == '0']
  df_merge_coincident = pd.merge(df_aux_1,df_coincident,how='left',on=lista_chaves)
  df_merge_coincident = df_merge_coincident[df_merge_coincident['week origin_y'].isnull()]

  if len(df_merge_coincident.index) > 0:
    df_merge_coincident = df_merge_coincident.loc[:,~df_merge_coincident.columns.str.contains('_', case=False)]

    mensagem = mensagem + f'\n \n Atenção! Não foi encontrada a conversão \033[1;33m"Coincident"\033[0;0;0m na coluna de \033[1;33m{coluna_idx_cohort}\033[0;0;0m no \033[1;33mbaseline\033[0;0;0m das seguintes aberturas no arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m: \n {tabulate(df_merge_coincident, headers=list(df_merge_coincident.columns), tablefmt="psql")}'
    contagem_de_erros += 1

  #------------------
  df_baseline_sem_ajuste = df_baseline[df_baseline[coluna_idx_cohort] != 'Coincident']
  df_group = df_baseline_sem_ajuste.groupby(lista_chaves, as_index=False)[colunas_conversoes].sum()
  df1 = df_group[(df_group[colunas_conversoes] > 1.0001).any(axis=1)]

  # passando de float pra porcentagem pro print ficar bonitinho
  for coluna in colunas_conversoes:
    df1[coluna] = df1[coluna].map(lambda n: '{:.2%}'.format(n))

  if len(df1.index) >0:
    mensagem = mensagem + f'\n \n Atenção! Algumas das cohorts seguintes apresentam soma maior que 100% na base de \033[1;33mbaselines\033[0;0;0m no arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m \n {tabulate(df1, headers=list(df1.columns), tablefmt="psql")}'
    contagem_de_erros += 1
  
  #--------------------
  #verifica se todos os elementos da coluna de week origin são inteiros
  try:
    indices_cohort = list(map(int,set(df_baseline_sem_ajuste[coluna_idx_cohort])))
  except:
    mensagem = mensagem + f'\n \n Atenção! Além do "Coincident" todos os elementos na coluna {coluna_idx_cohort} de \033[1;33mbaseline\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m devem ser números inteiros. Alguns deles não puderam ser lidos desta forma: {set(df_baseline_sem_ajuste[coluna_idx_cohort])}'
    contagem_de_erros += 1

  #--------------------
  #verifica se existem aberturas com valor na w5 porém coincident nula
  df_w5 = df_baseline[df_baseline[coluna_idx_cohort] == '5']
  df_coincident = df_baseline[df_baseline[coluna_idx_cohort] == 'Coincident']
  for coluna in colunas_conversoes:
    df_w5_aux = df_w5[df_w5[coluna] > 0.00001]
    df_merge = pd.merge(df_w5_aux,df_coincident,how='left',on=lista_chaves).fillna(0)
    df_merge = df_merge[df_merge[f'{coluna}_y'] == 0]
    if len(df_merge.index) > 0:
      df_merge = df_merge[lista_chaves]
      mensagem = mensagem + f'\n \n Atenção! As seguintes aberturas de \033[1;33mbaseline\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m contém valores de \033[1;33m{coluna}\033[0;0;0m para W5 mas não contém nenhum dado para "Coincident". \n {tabulate(df_merge, headers=list(df_merge.columns), tablefmt="psql")}'
      contagem_de_erros += 1
  
  return mensagem, contagem_de_erros




# @title Def verifica_datas_tofs
def verifica_datas_tofs(nome_do_arquivo,
                        df_tof_semanal,
                        df_tof_mensal,
                        aberturas,
                        coluna_datas_tof_semanal='data',
                        coluna_mes_tof_mensal='mês',
                        coluna_ano_tof_mensal='ano'):
  
  df_tof_semanal_c = df_tof_semanal.copy()
  df_tof_semanal_c[coluna_datas_tof_semanal] = pd.to_datetime(df_tof_semanal_c[coluna_datas_tof_semanal],infer_datetime_format=True)
  df_tof_mensal_c = df_tof_mensal.copy()

  contador_erros = 0
  mensagem = ''

  # Checanco se a base mensal possui datas que estão contidas na base semanal de uma forma geral
  #_________________________________________________________________________________________________
  
  #criando coluna de datas conde concatenamos dia 01 com mes e ano
  df_tof_mensal_c['data'] = df_tof_mensal_c[coluna_mes_tof_mensal]+'/01/'+df_tof_mensal_c[coluna_ano_tof_mensal]
  #lista de datas são os elementos unicos da coluna de data criada acima
  lista_datas_tof_mensal = pd.to_datetime(list(set(df_tof_mensal_c['data'])),format="%m/%d/%Y")

  #datas semanais são os elementos unicos da coluna de datas do tof semanal
  lista_datas_tof_semanal = list(set(df_tof_semanal_c[coluna_datas_tof_semanal]))
  lista_aux = []

  # Vamos verificar se o nome das aberturas estão em maiúsculo e as aberturas da base em minúsculo:
  if aberturas[0] not in df_tof_mensal_c.columns.values:
    aberturas = [x.lower() for x in aberturas]
  
  for elemento in lista_datas_tof_semanal:
    lista_aux.append(elemento)
    for i in range(1,7):
      lista_aux.append(elemento+pd.to_timedelta(i,unit="D"))
      
  if not set(lista_datas_tof_mensal).issubset(lista_aux):
    mensagem = mensagem + f'\n\n Atenção! As datas presentes no \033[1;33mToF Mensal\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo[1]}\033[0;0;0m não batem com as do \033[1;33mToF Semanal\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo[0]}\033[0;0;0m'
    contador_erros += 1
  #print(set(lista_datas_tof_mensal).issubset(lista_aux))






  # Checando se, em cada abertura, a primeira e última data mensal estão compreendidas nas primeiras
  # e últimas datas semanais/diárias
  #_________________________________________________________________________________________________
  

  # Vamos criar bases com as primeiras e últimas datas semanais e mensais por abertura:
  #-------------------------------------------------------------------------------------------------

  df_tof_mensal_c['data'] = pd.to_datetime(df_tof_mensal_c['data'],infer_datetime_format=True)
  df_tof_semanal_c[coluna_datas_tof_semanal] = pd.to_datetime(df_tof_semanal_c[coluna_datas_tof_semanal],infer_datetime_format=True)

  df_data_mensal_min = df_tof_mensal_c.groupby(aberturas, as_index=False)['data'].min()
  df_data_mensal_max = df_tof_mensal_c.groupby(aberturas, as_index=False)['data'].max()

  df_data_semanal_min = df_tof_semanal_c.groupby(aberturas, as_index=False)[coluna_datas_tof_semanal].min()
  df_data_semanal_max = df_tof_semanal_c.groupby(aberturas, as_index=False)[coluna_datas_tof_semanal].max()

  # Adicionamos 6 dias na última semana para definir o último dia da base semanal
  df_data_semanal_max[coluna_datas_tof_semanal] = df_data_semanal_max[coluna_datas_tof_semanal] + pd.to_timedelta(6,unit="D")

  # Subtraímos 6 dias da data mínima mensal e adicionamos 36 dias na data máxima para garantir que
  # a primeira e última semana estejam contidas nos meses:
  df_data_mensal_min['data'] = df_data_mensal_min['data'] - pd.to_timedelta(6,unit="D")
  df_data_mensal_max['data'] = df_data_mensal_max['data'] + pd.to_timedelta(36,unit="D")


  # Unir as bases e realizar comparações
  #-------------------------------------------------------------------------------------------------

  # Vamos renomear as colunas de datas para não confundir quando unirmos todas as bases:
  df_data_mensal_min = df_data_mensal_min.rename(columns={'data':'data_mensal_min'})
  df_data_mensal_max = df_data_mensal_max.rename(columns={'data':'data_mensal_max'})
  df_data_semanal_min = df_data_semanal_min.rename(columns={coluna_datas_tof_semanal:'data_semanal_min'})
  df_data_semanal_max = df_data_semanal_max.rename(columns={coluna_datas_tof_semanal:'data_semanal_max'})

  # Unir todas as bases para conseguir fazer comarações abertura por abertura:
  df_merged = pd.merge(df_data_mensal_min,df_data_mensal_max,how='outer',on=aberturas)
  df_merged = pd.merge(df_merged,df_data_semanal_min,how='outer',on=aberturas)
  df_merged = pd.merge(df_merged,df_data_semanal_max,how='outer',on=aberturas)

  # Definimos as colunas com a diferença de dias entre as datas minimas e máximas mensais e semanais:
  df_merged['data_dif_min'] = df_merged['data_semanal_min'] - df_merged['data_mensal_min']
  df_merged['data_dif_max'] = df_merged['data_mensal_max'] - df_merged['data_semanal_max']
  df_merged[['data_dif_min','data_dif_max']] = df_merged[['data_dif_min','data_dif_max']].fillna(pd.to_timedelta(-1,unit="D"))

  # Selecionamos as aberturas com eventuais problemas onde a diferença entre as datas é negativa:
  df_erro_data_minima = df_merged.loc[df_merged['data_dif_min'] < pd.to_timedelta(0,unit="D")]
  df_erro_data_maxima = df_merged.loc[df_merged['data_dif_max'] < pd.to_timedelta(0,unit="D")]

  if len(df_erro_data_minima) > 0:
    mensagem = mensagem + f'\n\n Atenção! As datas presentes no \033[1;33mToF Mensal\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo[0]}\033[0;0;0m se iniciam em meses posteriores às primeiras semanas na base \033[1;33mToF Semanal\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo[1]}\033[0;0;0m nas seguintes aberturas' + '\n' + tabulate(df_erro_data_minima, headers='keys', tablefmt='psql')
    contador_erros += 1    

  if len(df_erro_data_maxima) > 0:
    mensagem = mensagem + f'\n\n Atenção! As datas presentes no \033[1;33mToF Mensal\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo[0]}\033[0;0;0m terminam em meses anteriores às últimas semanas na base \033[1;33mToF Semanal\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo[1]}\033[0;0;0m nas seguintes aberturas' + '\n' + tabulate(df_erro_data_maxima, headers='keys', tablefmt='psql')
    contador_erros += 1  





  return mensagem,contador_erros




# @title Def Check_grupos_inputs
def ajusta_formato_grupos_inputs(nome_do_arquivo,dict_grupos_inputs_bruto,nome_grupos_inputs):
  mensagem = ''
  cont_erros = 0
  
  try:
    for element in dict_grupos_inputs_bruto:
      dict_grupos_inputs_bruto[element] = eval(dict_grupos_inputs_bruto[element])
  except:
    mensagem = mensagem + f'\n\n Atenção! Não foi possível ler como lista o seguinte elemento da aba de \033[1;33m{nome_grupos_inputs}\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m: {dict_grupos_inputs_bruto[element]}'
    cont_erros += 1

  return dict_grupos_inputs_bruto, mensagem, cont_erros

#-------------------------
def check_inputs_baseline(nome_do_arquivo,dict_grupos_inputs_tratado,df_baseline,aberturas_das_bases):
  mensagem = ''
  cont_erros = 0

  aberturas_das_bases = [e.lower() for e in aberturas_das_bases]

  for element in dict_grupos_inputs_tratado:
    if len(set(dict_grupos_inputs_tratado[element])) != len(dict_grupos_inputs_tratado[element]):
      cont_erros += 1
      mensagem = mensagem + f'\n\n Atenção! Existem elementos duplicados na lista {dict_grupos_inputs_tratado[element]} da base de \033[1;33mGrupos de Inputs\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m'

    flag = False
    for abertura in aberturas_das_bases:
      if set(dict_grupos_inputs_tratado[element]).issubset(set(df_baseline[abertura])):
        flag = True
        break
    if not flag:
      cont_erros += 1
      mensagem = mensagem + f'\n\n Atenção! Existem elemenos da lista {dict_grupos_inputs_tratado[element]} que não foram encontrados nas aberturas do \033[1;33mbaseline\033[0;0;0m do arquivo \033[1;34m{nome_do_arquivo}\033[0;0;0m'

  return(mensagem,cont_erros)



# @title Def check_base_inputs

def check_base_inputs(base_inputs,
                      col_valores,
                      etapas_conversao,
                      nome_do_arquivo):
  
  mensagem = ''
  erro = 0


  # Checar conteúdo específico das colunas:
  col_conversao = list(np.unique(base_inputs['conversão'].values))
  try:
    col_conversao.remove('Coincident')
  except:
    pass
  try:
    col_conversao = [int(x) for x in col_conversao]
  except:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', a coluna '+colored('Conversão','red')+' na base '+colored(base_inputs.name,'yellow')+' possui valores que não são conversões.\nO único valor aceito nessa coluna que não seja um número inteiro é "Coincident".\nOs valores encontrados na coluna foram: '+colored(str(col_conversao),'red')
    erro += 1 

  col_aplicacao = list(np.unique(base_inputs['aplicação'].values))
  valores_errados = list(set(col_aplicacao) - set(['Pontual','Permanente']))
  if len(valores_errados) > 0:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', a coluna '+colored('Aplicação','red')+' na base '+colored(base_inputs.name,'yellow')+' possui valores não permitidos. Os únicos valores permitidos são "Pontual" e "Permanente".\nOs valores encontrados na coluna foram: '+colored(str(col_aplicacao),'red')
    erro += 1  

  col_etapa = list(np.unique(base_inputs['etapa'].values))
  col_etapa = [e.lower() for e in col_etapa]
  valores_errados = list(set(col_etapa) - set(etapas_conversao))
  if len(valores_errados) > 0:
    mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', a coluna '+colored('Etapa','red')+' na base '+colored(base_inputs.name,'yellow')+' possui valores não permitidos. Os únicos valores permitidos são: '+colored(str(etapas_conversao),'red')+'\nOs valores encontrados na coluna foram: '+colored(str(col_etapa),'red')+f'\n Atenção aos valores:\033[1;31m{list(set(col_etapa)-set(etapas_conversao))}\033[0;0;0m'
    erro += 1  

  # Checar valores inferiores a -100%
  cb = list(base_inputs.columns.values)
  idx_valores = cb.index('etapa')+1
  for i in range(idx_valores,len(cb)):
    valor_minimo = np.min(base_inputs[cb[i]].values)
    if valor_minimo < -1:
      mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', a coluna '+colored(str(cb[i]),'red')+' na base '+colored(base_inputs.name,'yellow')+' possui valores inferiores a -100%'
      erro += 1 

  # Checar linhas de inputs sem valores:
  for i in range(len(base_inputs)):
    soma_valores = np.sum(base_inputs.iloc[i,idx_valores:].values)
    if soma_valores == 0:
      mensagem = mensagem+'\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+', a linha '+colored(str(i+2),'red')+' na base '+colored(base_inputs.name,'yellow')+' não possui nenhum input'
      erro += 1 

  return mensagem,erro



# @title Def check_geral

'''
Função que aplica os seguintes checks e formatações numa lista de dataframes:

1 - existência de colunas obrigatórias (ordena as colunas e retorna colunas de valores)
2 - formatação de valores (formata os valores das bases)
3 - verifica existência de valores negativos
4 - verifica compatibilidade das chaves das aberturas entre as bases
5 - verifica formatação das datas (retorna bases com as datas formatadas)

'''

def check_geral(lista_de_bases,                 # Lista de bases que vamos verificar (dataframes)
                lista_de_colunas_obrigatorias,  # Lista das colunas obrigatórias correspondentes de cada base
                lista_colunas_de_valores,       # Lista com as colunas de valores pré-definidas de cada base
                lista_do_retorno_de_valores,    # Lista indicando quais bases possuem colunas de valores a serem inferidas e retornadas por este check
                lista_mantem_formatacao_original = True,
                lista_check_vazios,             # Lista indicando quais bases devem fazer uma checagem de valores vazios (fórmulas arrastadas até o final)
                lista_verifica_valores,         # Lista indicando quais bases devem fazer uma checagem de valores vazios (fórmulas arrastadas até o final)
                lista_de_bases_checar_chaves,   # Lista indicando quais bases devem fazer uma checagem de valores vazios (fórmulas arrastadas até o final)
                lista_comparacao_parcial,       # Lista indicando quais bases devem fazer uma checagem de valores vazios (fórmulas arrastadas até o final)
                lista_comparacao_a_mais = [],
                aberturas_especificas,          # Lista com as aberturas específicas que não são chaves de comparação mas não são colunas com valores
                lista_lista_colunas_datas,      # Lista com as aberturas específicas que não são chaves de comparação mas não são colunas com valores
                lista_lista_frequencia,         # Lista com as aberturas específicas que não são chaves de comparação mas não são colunas com valores
                chaves_ignoradas,               # Lista com as aberturas específicas que não são chaves de comparação mas não são colunas com valores
                aberturas_das_bases,            # Lista com as aberturas comuns às bases, definidas no painel de controle
                coluna_de_conversoes,
                Nome_do_arquivo_sheets,
                tipo_de_tof = '',
                dict_renames):
  

  contador_de_erros = 0
  mensagens = ''
    
  # Para cada base, vamos verificar se ela contém as colunas obrigatórias. Cada base pode conter diferentes
  # colunas obrigatórias, por isso devemos checar cada uma individualmente.
  #___________________________________________________________________________________________________
  erros_ate_o_momento = contador_de_erros
  flag_erro_colunas = False 
  mensagens_locais = ''  
  cabecalho_dos_erros = '\n\n\nErros de Colunas:\n__________________________________________________________________________________________________________________________________________\n'                          

  # Para cada base na lista de bases, vamos realizar a verificação das colunas obrigatórias:
  for b in range(len(lista_de_bases)):

    if len(lista_de_bases[b]) > 0:
           
      lista_de_bases[b],colunas_de_valores,mensagem_local,erro_local = check_colunas(df = lista_de_bases[b],     # DataFrame
                                                                                     lista_df = lista_de_bases,
                                                                                     aberturas = aberturas_das_bases,                                                                                    
                                                                                    colunas_obrigatorias = lista_de_colunas_obrigatorias[b],  # lista com as colunas obrigatórias e a ordem
                                                                                    retorna_col_valores = lista_do_retorno_de_valores[b],                 # booleano que determina se a função vai retornar as colunas de valores
                                                                                    mantem_formatacao_original = lista_mantem_formatacao_original[b],
                                                                                    dict_renames = dict_renames,
                                                                                    coluna_de_conversoes = coluna_de_conversoes,
                                                                                    colunas_datas = lista_lista_colunas_datas[b],
                                                                                    nome_do_arquivo = Nome_do_arquivo_sheets[b])
      
      if lista_do_retorno_de_valores[b]:
        lista_colunas_de_valores[b] = colunas_de_valores
      
      contador_de_erros = contador_de_erros+erro_local
      mensagens_locais = mensagens_locais+mensagem_local


  if mensagens_locais != '':
    mensagens = mensagens+cabecalho_dos_erros+mensagens_locais
  if contador_de_erros > erros_ate_o_momento:
    flag_erro_colunas = True


  # Verificar a formatação dos valores.
  #___________________________________________________________________________________________________

  erros_ate_o_momento = contador_de_erros
  flag_erro_valores = False
  mensagens_locais = ''
  cabecalho_dos_erros = '\n\n\nErros na formatação de valores:\n__________________________________________________________________________________________________________________________________________\n'                          

  if not flag_erro_colunas: # nao conseguimos verificar os valores se a coluna de valores não existir na base

    # Para cada base na lista de bases, vamos realizar a verificação das colunas obrigatórias:
    for b in range(len(lista_de_bases)):

      if len(lista_de_bases[b]) > 0:

        lista_de_bases[b], mensagem_local, erro_local = check_valores(df = lista_de_bases[b],                   # DataFrame já deve ter checado a existencia das colunas de valores
                                                                        colunas_de_valores = lista_colunas_de_valores[b],    # lista com as colunas que contém valores
                                                                        check_valores_vazios = lista_check_vazios[b],                # Boleano que, caso seja verdadeiro, verifica se existem blocos de valores vazios nas colunas
                                                                        mantem_formatacao_original = lista_mantem_formatacao_original[b],
                                                                      nome_do_arquivo = Nome_do_arquivo_sheets[b])


        contador_de_erros = contador_de_erros+erro_local
        mensagens_locais = mensagens_locais+mensagem_local

    if mensagens_locais != '':
      mensagens = mensagens+cabecalho_dos_erros+mensagens_locais

    if contador_de_erros > erros_ate_o_momento:
      flag_erro_valores = True



  # Verificar a existência de valores negativos.
  #___________________________________________________________________________________________________

  erros_ate_o_momento = contador_de_erros
  mensagens_locais = ''
  cabecalho_dos_erros = '\n\n\nErros de valoes negativos:\n__________________________________________________________________________________________________________________________________________\n'                          

  if not flag_erro_valores and not flag_erro_colunas: # nao conseguimos verificar os valores negativos se não conseguimos formatar os valores


    # Para cada base na lista de bases, vamos realizar a verificação das colunas obrigatórias:
    for b in range(len(lista_de_bases)):
      if not lista_verifica_valores[b] or len(lista_de_bases[b]) == 0:
        pass
      else:
        erro_local_aux, mensagem_local_aux = check_valores_negativos(dataframe = lista_de_bases[b],
                                                          lista_colunas_numericas = lista_colunas_de_valores[b])
        erro_local = erro_local + erro_local_aux
        mensagem_local = mensagem_local + mensagem_local_aux

      contador_de_erros = contador_de_erros+erro_local
      mensagens_locais = mensagens_locais+mensagem_local

    if mensagens_locais != '':
      mensagens = mensagens+cabecalho_dos_erros+mensagens_locais


  # Verificar as chaves das aberturas entre as bases.
  #___________________________________________________________________________________________________

  erros_ate_o_momento = contador_de_erros
  flag_erro_chaves = False
  mensagens_locais = ''
  cabecalho_dos_erros = '\n\n\nErros na comparação de chaves entre as bases:\n__________________________________________________________________________________________________________________________________________\n'                          


  if not flag_erro_valores and not flag_erro_colunas: 


    lista_de_bases_chaves = list(compress(lista_de_bases, lista_de_bases_checar_chaves))
    lista_comparacao_parcial = list(compress(lista_comparacao_parcial, lista_de_bases_checar_chaves))
    lista_nomes_arquivos_chaves = list(compress(Nome_do_arquivo_sheets, lista_de_bases_checar_chaves))
    if len(lista_comparacao_a_mais) > 0:
      lista_comparacao_a_mais = list(compress(lista_comparacao_a_mais, lista_de_bases_checar_chaves))

    lista_de_bases_chaves, mensagem_local, erro_local = check_chaves(lista_df = lista_de_bases_chaves,                   # lista de DataFrames já devem ter as colunas de valores formatadas
                                                                    aberturas_compartilhadas = aberturas_das_bases,   # lista com as aberturas que devem estar presentes em todas as bases da lista de dataframes
                                                                    aberturas_especificas = aberturas_especificas,      # lista com as aberturas que não precisam estar presentes em todas as bases
                                                                    lista_comparacao_parcial = lista_comparacao_parcial,   # lista de booleanos indicando quais bases serão checadas se contém todas as aberturas de todas as bases ou se contém aberturas que outras bases não tem
                                                                    lista_comparacao_a_mais = lista_comparacao_a_mais,
                                                                    chaves_ignoradas = chaves_ignoradas,           # lista de chaves a serem ignoradas se encontradas, como chaves globais "Todos" por exemplo                                                                 
                                                                    tipo_de_tof = tipo_de_tof,
                                                                    nome_do_arquivo = lista_nomes_arquivos_chaves,
                                                                    agrupar_duplicados = True)

    contador_de_erros = contador_de_erros+erro_local
    mensagens_locais = mensagens_locais+mensagem_local

    if mensagens_locais != '':
      mensagens = mensagens+cabecalho_dos_erros+mensagens_locais

    if contador_de_erros > erros_ate_o_momento:
      flag_erro_chaves = True



  # Vamos verificar a formatação das datas.
  #___________________________________________________________________________________________________

  erros_ate_o_momento = contador_de_erros
  mensagens_locais = ''
  cabecalho_dos_erros = '\n\n\nErros de formatação de datas:\n__________________________________________________________________________________________________________________________________________\n'                          

  if not flag_erro_colunas: # nao conseguimos verificar as datas se a coluna de datas não existir na base



    # Para cada base na lista de bases
    for b in range(len(lista_de_bases)):

      if len(lista_de_bases[b]) > 0:

        lista_de_bases[b], erro_local, mensagem_local = ajusta_formato_data(nome_do_arquivo= Nome_do_arquivo_sheets[b],
                                                                            dataframe = lista_de_bases[b],
                                                                              lista_colunas_datas = lista_lista_colunas_datas[b])

        contador_de_erros = contador_de_erros+erro_local
        mensagens_locais = mensagens_locais+mensagem_local


        lista_de_bases[b], erro_local, mensagem_local = verifica_frequencia_datas(nome_do_arquivo= Nome_do_arquivo_sheets[b],
                                                                                  dataframe = lista_de_bases[b],
                                                                                  lista_colunas_datas = lista_lista_colunas_datas[b],
                                                                                  lista_frequencia = lista_lista_frequencia[b])


        contador_de_erros = contador_de_erros+erro_local
        mensagens_locais = mensagens_locais+mensagem_local

    if mensagens_locais != '':
      mensagens = mensagens+cabecalho_dos_erros+mensagens_locais

    if contador_de_erros > erros_ate_o_momento:
      flag_erro_datas = True


  return lista_de_bases, lista_colunas_de_valores, mensagens, contador_de_erros




#@title Def check_transforma_base_demanda

def check_transforma_base_demanda(base_df,
                                  aberturas,
                                  col_week_origin):
  
  cb_df = list(base_df.columns.values)

  aberturas_minusculo = [x.lower() for x in aberturas]

  col_week_origin = col_week_origin.lower()

  if 'tier' in cb_df:
    aberturas_minusculo = aberturas_minusculo+['tier']

  if 'vc' in cb_df and col_week_origin in cb_df and 'vc2os' in cb_df:

    mensagem = ''

    nome = base_df.name
    col_data = cb_df[0]
    index_conversion = cb_df.index(col_week_origin)
    col_valores = cb_df[index_conversion+1:]
    base_df[col_data] = pd.to_datetime(base_df[col_data], infer_datetime_format=True)

    # Vamos somar os volumes das colunas vc e vb2vc e comparar as diferenças:
    #---------------------------------------------------------------------------------------------------

    base_df[['vc2os','vc']] = base_df[['vc2os','vc']].astype(float)

    base_vol = base_df.groupby([col_data]+aberturas_minusculo, as_index=False)[['vc2os','vc']].sum()
    base_vol['delta'] = base_vol['vc'] - base_vol['vc2os']


    # Vamos unir a base com a diferença nos volumes com a base original
    base_merged = pd.merge(base_df,base_vol[[col_data]+aberturas_minusculo+['delta']],how='left',on=[col_data]+aberturas_minusculo)


    # Vamos corrigir a diferença nos volumes na conversão 'Não Convertido':
    base_merged.loc[(base_merged[col_week_origin] == 'Não Convertido'),['vc2os']] =\
    base_merged.loc[(base_merged[col_week_origin] == 'Não Convertido'),['vc2os']].values +\
    base_merged.loc[(base_merged[col_week_origin] == 'Não Convertido'),['delta']].values


    # Vamos criar a conversão de ajuste:
    #---------------------------------------------------------------------------------------------------

    col_valores.remove('vc')
    base_merged = base_merged.drop(columns=['vc'])

    col_valores_delta = [x + "_d" for x in col_valores][:-1]
    col_valores_x = [x + "_x" for x in col_valores]
    col_valores_y = [x + "_y" for x in col_valores]

    base_merged[col_valores] = base_merged[col_valores].replace('',0)

    base_merged[col_valores] = base_merged[col_valores].astype(float)

    base_vol_ajuste = base_merged.groupby([col_data]+aberturas_minusculo, as_index=False)[col_valores].sum()

    base_diagonal = base_merged.loc[(base_merged[col_week_origin] != 'Não Convertido') & (base_merged[col_week_origin] != '5')]
    base_diagonal[col_week_origin] = base_diagonal[col_week_origin].astype(int)
    base_diagonal[col_week_origin] = base_diagonal[col_week_origin].apply(lambda x: pd.Timedelta(x*7, unit='D'))
    base_diagonal['shifted'] = base_diagonal[col_data] + base_diagonal[col_week_origin]
    
    base_diagonal = base_diagonal.groupby(['shifted']+aberturas_minusculo, as_index=False)[col_valores].sum()
    base_diagonal = base_diagonal.rename(columns={'shifted':col_data})

    base_vol_merged = pd.merge(base_vol_ajuste,base_diagonal,how='left',on=[col_data]+aberturas_minusculo)

    base_vol_merged[col_valores_delta] = base_vol_merged[col_valores_x[1:]].values - base_vol_merged[col_valores_y[:-1]].values

    base_vol_merged[col_data] = base_vol_merged[col_data] - pd.Timedelta(5*7, unit='D')
    base_vol_merged[col_week_origin] = 'Coincident'

    base_vol_merged = base_vol_merged.drop(columns=col_valores_y+col_valores_x)
    base_vol_merged = base_vol_merged.rename(columns=dict(zip(col_valores_delta, col_valores[:-1])))

    base_merged = pd.concat([base_merged,base_vol_merged])
    base_merged = base_merged.fillna(0)


    # Retornamos a base final
    base_df = base_merged.copy()
    if 'vc' in base_df.columns.values:
      base_df = base_df.drop(columns=['vc'])
    if 'delta' in base_df.columns.values:
      base_df = base_df.drop(columns=['delta'])
    #base_df[col_data] = base_df[col_data].apply(lambda x: x.strftime('%m/%d/%Y'))


    cb_df = list(base_df.columns.values)
    index_conversion = cb_df.index(col_week_origin)
    col_valores = cb_df[index_conversion+1:]
    base_df = base_df.groupby([col_data]+aberturas_minusculo+[col_week_origin], as_index=False)[col_valores].sum()
    base_df.name = nome
    

  return base_df


#@title Def check_building_blocks

def check_building_blocks(base_tof_semanal,
                          base_tof_mensal,
                          base_inputs,
                          aberturas,
                          coluna_de_semanas,
                          nome_do_arquivo,
                          chaves_ignoradas):
  
  mensagem_bb = ''
  erro_bb = 0

  # Primeiro, vamos comparar as aberturas entre as bases de ToF e inputs incluindo a coluna de building blocks tof:

  lista_df_tofs,mensagem_bb,erro_bb = check_chaves(lista_df = [base_tof_mensal,base_tof_semanal,base_inputs],                   # lista de DataFrames já devem ter as colunas de valores formatadas
                                                        aberturas_compartilhadas = ['building block tof']+aberturas,   # lista com as aberturas que devem estar presentes em todas as bases da lista de dataframes
                                                        aberturas_especificas = [coluna_de_semanas,'mês','ano']+['building block cohort','Conversão','Aplicação','Etapa'],      # lista com as aberturas que não precisam estar presentes em todas as bases
                                                        lista_comparacao_parcial = [False,False,True],   # lista de booleanos indicando quais bases serão checadas se contém todas as aberturas de todas as bases ou se contém aberturas que outras bases não tem
                                                        chaves_ignoradas = chaves_ignoradas,           # lista de chaves a serem ignoradas se encontradas, como chaves globais "Todos" por exemplo
                                                        agrupar_duplicados = True,
                                                        nome_do_arquivo = nome_do_arquivo)   

  # Verificamos o conteúdo das colunas de building blocks tof
  if erro_bb == 0:

    list_building_blocks_tof = list(np.unique(base_tof_mensal['building block tof'].values))

    baselines = ['Baseline','baseline','BASELINE']

    lista_baselines = [i for i in list_building_blocks_tof if i in baselines]

    if len(lista_baselines) == 0:
      mensagem_bb = mensagem_bb+'\n\nNo arquivo '+colored(nome_do_arquivo[0],'blue')+' a base '+colored(base_tof_mensal.name,'yellow')+' não contém "baseline" na coluna "Building Block ToF".\nPara rodar o planning é necessária a existência de volumes de ToF de baseline, que pode ser indicados pelos seguintes nomes: '+colored(str(baselines),'blue')+'\nPorém, foi encontrada somente esta lista de Building Blocks ToF: '+colored(str(list_building_blocks_tof),'red')
      erro_bb += 1

    if len(lista_baselines) > 1:
      mensagem_bb = mensagem_bb+'\n\nNo arquivo '+colored(nome_do_arquivo[0],'blue')+' a base '+colored(base_tof_mensal.name,'yellow')+' possui mais de um "baseline" na coluna "Building Block ToF".\nPara rodar o planning é necessária a existência de um único baseline, que pode ser indicados pelos seguintes nomes: '+colored(str(baselines),'blue')+'\nForam encontrados os seguintes Building Blocks ToF que podem ser interpretados como "baseline": '+colored(str(lista_baselines),'red')
      erro_bb += 1

  # Formatando o conteúdo das colunas de Building Blocks ToF:
  if erro_bb == 0:

    base_tof_semanal['building block tof'] = base_tof_semanal['building block tof'].str.replace(lista_baselines[0],'baseline')
    base_tof_mensal['building block tof'] = base_tof_mensal['building block tof'].str.replace(lista_baselines[0],'baseline')
    base_inputs['building block tof'] = base_inputs['building block tof'].str.replace(lista_baselines[0],'baseline')

  return base_tof_semanal,base_tof_mensal,base_inputs,mensagem_bb,erro_bb
  
  
  
#@title def confere_datas_tof_travado

def confere_datas_tof_travado(data_piso, data_teto, format="%Y/%m/%d"):
  mensagem = ''
  cont_erros = 0

  try:
    data_piso = pd.to_datetime(data_piso,format=format)
    data_teto = pd.to_datetime(data_teto,format=format)
  except:
    mensagem = mensagem + '\n Atenção! Não foi possível converter para o formato de data as datas de piso e/ou teto da base histórica.'
    cont_erros += 1

  if cont_erros == 0:
    if data_piso > data_teto:
      mensagem = 'Atenção! A data piso para recorte da base histórica de act e respectivo cálculo de Topo Travado deve ser anterior à data de teto'
      cont_erros += 1
  
  return cont_erros, mensagem, data_piso, data_teto


#@title Def check_bases_realizadas

def check_bases_realizadas(df_realizado_mensal,
                           df_realizado_cohort,
                           df_realizado_tp,
                           df_metas_tp,
                           df_modelo,
                           aberturas_da_base,
                           df_dicionario_mensal,
                           df_dicionario_cohort,
                           col_extra,
                           nome_coluna_week_origin,
                           col_volumes,
                           col_conversoes,
                           nome_do_arquivo):
  
  erros = 0
  mensagem = ''

  
  if len(df_realizado_mensal) > 0 and len(df_realizado_cohort) > 0:

    colunas_obrigatorias_mensal = list(df_dicionario_mensal['Nome Original'].values)
    colunas_novas_mensal = list(df_dicionario_mensal['Nome Novo'].values)
    colunas_novas_mensal = [c.lower() for c in colunas_novas_mensal]

    colunas_obrigatorias_cohort = list(df_dicionario_cohort['Nome Original'].values)
    colunas_novas_cohort  = list(df_dicionario_cohort['Nome Novo'].values)
    colunas_novas_cohort  = [c.lower() for c in colunas_novas_cohort]


    col_extra = [c.lower() for c in col_extra]
    aberturas_oficiais = [c.lower() for c in aberturas_da_base]

    nome_mensal = df_realizado_mensal.name
    nome_cohort = df_realizado_cohort.name

    lista_de_bases = [df_modelo,df_realizado_mensal,df_realizado_cohort,df_realizado_tp]


    df_realizado_mensal,colunas_extra,mensagem_local,erro_local = check_colunas(df = df_realizado_mensal,     # DataFrame
                                                                                lista_df = lista_de_bases,
                                                                                aberturas = aberturas_da_base,
                                                                                  colunas_obrigatorias = colunas_obrigatorias_mensal,  # lista com as colunas obrigatórias e a ordem
                                                                                  retorna_col_valores = True,
                                                                                dict_renames = dict_renames,                 # booleano que determina se a função vai retornar as colunas de valores
                                                                                  
                                                                                  nome_do_arquivo = nome_do_arquivo)

    erros = erros+erro_local
    mensagem = mensagem + mensagem_local

    df_realizado_cohort,colunas_extra,mensagem_local,erro_local = check_colunas(df = df_realizado_cohort,     # DataFrame
                                                                                lista_df = lista_de_bases,
                                                                                aberturas = aberturas_da_base,
                                                                                  colunas_obrigatorias = colunas_obrigatorias_cohort,  # lista com as colunas obrigatórias e a ordem
                                                                                  retorna_col_valores = True,  
                                                                                dict_renames = dict_renames,               # booleano que determina se a função vai retornar as colunas de valores
                                                                                  nome_do_arquivo = nome_do_arquivo)

    erros = erros+erro_local
    mensagem = mensagem + mensagem_local



    if erros == 0:

      colunas_atuais_mensal = list(set(list(df_realizado_mensal.columns.values))-set(colunas_extra))
      colunas_atuais_mensal = [c for c in df_realizado_mensal.columns.values if c in colunas_atuais_mensal]

      df_realizado_mensal = df_realizado_mensal.rename(columns=dict(zip(colunas_atuais_mensal ,colunas_obrigatorias_mensal)))
      df_realizado_mensal = df_realizado_mensal.rename(columns=dict(zip(colunas_obrigatorias_mensal,colunas_novas_mensal)))

      df_realizado_mensal = df_realizado_mensal[colunas_novas_mensal]

      aberturas_realizado = list(set(colunas_novas_mensal)-set(col_volumes+col_extra+['período']))
      etapas_realizado = list(set(colunas_novas_mensal)-set(aberturas_oficiais+col_extra+['período']))

      diff_aberturas_1 = list(set(aberturas_oficiais)-set(aberturas_realizado))
      diff_aberturas_2 = list(set(aberturas_realizado)-set(aberturas_oficiais))

      diff_etapas_1 = list(set(col_volumes)-set(etapas_realizado))
      diff_etapas_2 = list(set(etapas_realizado)-set(col_volumes))

      if len(diff_aberturas_1) > 0 or len(diff_aberturas_2) > 0:
        mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+' as colunas de aberturas da base '+colored(str(nome_mensal),'yellow')+' encontradas são: '+colored(str(aberturas_realizado),'red')+'. \nEssas aberturas não batem com as aberturas definidas no Painel de Controle: '+colored(str(aberturas_oficiais),'red')
        erros = erros+1

      if len(diff_etapas_1) > 0 or len(diff_etapas_2) > 0:
        mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+' as colunas de etapas do funil da base '+colored(str(nome_mensal),'yellow')+' encontradas são: '+colored(str(etapas_realizado),'red')+'. \nEssas etapas não batem com as aberturas definidas no Painel de Controle: '+colored(str(col_volumes),'red')
        erros = erros+1


      colunas_atuais_cohort = list(set(list(df_realizado_cohort.columns.values))-set(colunas_extra))
      colunas_atuais_cohort = [c for c in df_realizado_cohort.columns.values if c in colunas_atuais_cohort]

      df_realizado_cohort = df_realizado_cohort.rename(columns=dict(zip(colunas_atuais_cohort ,colunas_obrigatorias_cohort)))
      df_realizado_cohort = df_realizado_cohort.rename(columns=dict(zip(colunas_obrigatorias_cohort,colunas_novas_cohort)))

      df_realizado_cohort = df_realizado_cohort[colunas_novas_cohort]

      aberturas_realizado = list(set(colunas_novas_cohort)-set(col_conversoes+col_extra+['período',nome_coluna_week_origin]))
      etapas_realizado = list(set(colunas_novas_cohort)-set(aberturas_oficiais+col_extra+['período',nome_coluna_week_origin]))

      diff_aberturas_1 = list(set(aberturas_oficiais)-set(aberturas_realizado))
      diff_aberturas_2 = list(set(aberturas_realizado)-set(aberturas_oficiais))

      diff_etapas_1 = list(set(col_conversoes)-set(etapas_realizado))
      diff_etapas_2 = list(set(etapas_realizado)-set(col_conversoes))


      if diff_aberturas_2 == ['vc']:
        diff_aberturas_2 = []
      if diff_etapas_2 == ['vc']:
        diff_etapas_2 = []

      if len(diff_aberturas_1) > 0 or len(diff_aberturas_2) > 0:
        mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+' as colunas de aberturas da base '+colored(str(nome_cohort),'yellow')+' encontradas são: '+colored(str(aberturas_realizado),'red')+'. \nEssas aberturas não batem com as aberturas definidas no Painel de Controle: '+colored(str(aberturas_oficiais),'red')
        erros = erros+1

      if len(diff_etapas_1) > 0 or len(diff_etapas_2) > 0:
        mensagem = mensagem + '\n\nNo arquivo '+colored(nome_do_arquivo,'blue')+' as colunas de etapas do funil da base '+colored(str(nome_cohort),'yellow')+' encontradas são: '+colored(str(etapas_realizado),'red')+'. \nEssas etapas não batem com as aberturas definidas no Painel de Controle: '+colored(str(col_conversoes),'red')
        erros = erros+1

    if erros == 0:

      df_realizado_mensal.name = nome_mensal
      df_realizado_cohort.name = nome_cohort


      df_realizado_mensal,mensagem_local,erro_local = check_valores(df = df_realizado_mensal,                    # DataFrame já deve ter checado a existencia das colunas de valores
                                                                    colunas_de_valores = col_volumes,    # lista com as colunas que contém valores
                                                                    check_valores_vazios = False,  # Boleano que, caso seja verdadeiro, verifica se existem blocos de valores vazios nas colunas
                                                                    nome_do_arquivo = nome_do_arquivo)
      erros = erros+erro_local
      mensagem = mensagem + mensagem_local

      df_realizado_cohort,mensagem_local,erro_local = check_valores(df = df_realizado_cohort,                    # DataFrame já deve ter checado a existencia das colunas de valores
                                                                    colunas_de_valores = col_conversoes,    # lista com as colunas que contém valores
                                                                    check_valores_vazios = False,  # Boleano que, caso seja verdadeiro, verifica se existem blocos de valores vazios nas colunas
                                                                    nome_do_arquivo = nome_do_arquivo)
      
      erros = erros+erro_local
      mensagem = mensagem + mensagem_local





      df_realizado_mensal.name = nome_mensal
      df_realizado_cohort.name = nome_cohort



      df_realizado_mensal,erro_local,mensagem_local = ajusta_formato_data(nome_do_arquivo = nome_do_arquivo, 
                                                                          dataframe = df_realizado_mensal,
                                                                          lista_colunas_datas = ['período'])
      
      erros = erros+erro_local
      mensagem = mensagem + mensagem_local

      df_realizado_cohort,erro_local,mensagem_local = ajusta_formato_data(nome_do_arquivo = nome_do_arquivo, 
                                                                          dataframe = df_realizado_cohort,
                                                                          lista_colunas_datas = ['período'])
      
      erros = erros+erro_local
      mensagem = mensagem + mensagem_local



      df_realizado_cohort = check_transforma_base_demanda(base_df = df_realizado_cohort,
                                                          aberturas = aberturas_da_base,
                                                          col_week_origin = nome_coluna_week_origin)
    

  if len(df_metas_tp) > 0 and len(df_realizado_tp) > 0:

    colunas_obrigatorias_tp = ['tp']

    col_data_realizado_tp = list(df_realizado_tp.columns.values)[0]
    col_data_metas_tp = list(df_metas_tp.columns.values)[0]

    nome_actual_tp = df_realizado_tp.name
    nome_meta_tp = df_metas_tp.name

    lista_df_check = [df_modelo,df_realizado_mensal,df_realizado_cohort,df_realizado_tp,df_metas_tp]

    df_realizado_tp,colunas_extra,mensagem_local,erro_local = check_colunas(df = df_realizado_tp,     # DataFrame
                                                                            lista_df = lista_df_check,
                                                                            aberturas = aberturas_da_base,
                                                                                  colunas_obrigatorias = colunas_obrigatorias_tp,  # lista com as colunas obrigatórias e a ordem
                                                                                  retorna_col_valores = True,  
                                                                            dict_renames = dict_renames,               # booleano que determina se a função vai retornar as colunas de valores
                                                                                  nome_do_arquivo = nome_do_arquivo)

    erros = erros+erro_local
    mensagem = mensagem + mensagem_local


    df_metas_tp,colunas_extra,mensagem_local,erro_local = check_colunas(df = df_metas_tp,     # DataFrame
                                                                            lista_df = lista_df_check,
                                                                            aberturas = aberturas_da_base,
                                                                            colunas_obrigatorias = colunas_obrigatorias_tp,  # lista com as colunas obrigatórias e a ordem
                                                                            retorna_col_valores = True,      
                                                                        dict_renames = dict_renames,           # booleano que determina se a função vai retornar as colunas de valores
                                                                            nome_do_arquivo = nome_do_arquivo)

    erros = erros+erro_local
    mensagem = mensagem + mensagem_local

    if erros == 0:

      df_realizado_tp.name = nome_actual_tp
      df_metas_tp.name = nome_meta_tp


      df_realizado_tp,mensagem_local,erro_local = check_valores(df = df_realizado_tp,                    # DataFrame já deve ter checado a existencia das colunas de valores
                                                                    colunas_de_valores = ['tp'],    # lista com as colunas que contém valores
                                                                    check_valores_vazios = False,  # Boleano que, caso seja verdadeiro, verifica se existem blocos de valores vazios nas colunas
                                                                    nome_do_arquivo = nome_do_arquivo)
      
      erros = erros+erro_local
      mensagem = mensagem + mensagem_local


      df_metas_tp,mensagem_local,erro_local = check_valores(df = df_metas_tp,                    # DataFrame já deve ter checado a existencia das colunas de valores
                                                                    colunas_de_valores = ['tp'],    # lista com as colunas que contém valores
                                                                    check_valores_vazios = False,  # Boleano que, caso seja verdadeiro, verifica se existem blocos de valores vazios nas colunas
                                                                    nome_do_arquivo = nome_do_arquivo)
      
      erros = erros+erro_local
      mensagem = mensagem + mensagem_local


      df_realizado_tp.name = nome_actual_tp
      df_metas_tp.name = nome_meta_tp

      
      df_realizado_tp,erro_local,mensagem_local = ajusta_formato_data(nome_do_arquivo = nome_do_arquivo, 
                                                                          dataframe = df_realizado_tp,
                                                                          lista_colunas_datas = [col_data_realizado_tp])
      
      erros = erros+erro_local
      mensagem = mensagem + mensagem_local


      df_metas_tp,erro_local,mensagem_local = ajusta_formato_data(nome_do_arquivo = nome_do_arquivo, 
                                                                          dataframe = df_metas_tp,
                                                                          lista_colunas_datas = [col_data_metas_tp])
      
      erros = erros+erro_local
      mensagem = mensagem + mensagem_local


      df_metas_tp = df_metas_tp.rename(columns={col_data_metas_tp:'período'})
      df_realizado_tp = df_realizado_tp.rename(columns={col_data_realizado_tp:'período'})

      df_realizado_tp = df_realizado_tp.groupby(['período'],as_index=False)['tp'].sum()
      df_metas_tp = df_metas_tp.groupby(['período'],as_index=False)['tp'].sum()


  return df_realizado_mensal,df_realizado_cohort,df_realizado_tp,df_metas_tp,erros,mensagem

def check_datas_base_share_diario(df_parametro,
                             data_inicio,
                             data_fim,
                             coluna_semanas,
                             coluna_datas
):

  # Definições iniciais
  #-------------------------------------------------------------------------------------------------
  mensagem = ''


  # Caso não encontremos a coluna de semanas na base, vamos criar uma:
  
  if coluna_semanas not in df_parametro.columns.values:
    coluna_semanas = 'week_start'
    df_parametro[coluna_semanas] = df_parametro[coluna_datas].dt.to_period('W').apply(lambda r: r.start_time)
  
  data_inicio = datetime.strptime(data_inicio, '%Y-%m-%d') #conversão que em tese já está feita no código do planning
  data_fim = datetime.strptime(data_fim, '%Y-%m-%d') #conversão que em tese já está feita no código do planning

  erro = 0


  #Convertendo as datas para depois filtrar
  df_parametro[coluna_semanas] = pd.to_datetime(df_parametro[coluna_semanas], infer_datetime_format = True)

  df_parametro[coluna_datas] = pd.to_datetime(df_parametro[coluna_datas], infer_datetime_format = True)


  #Filtra o DataFrame com base nas datas
  df_parametro_filtrado = df_parametro[(df_parametro[coluna_semanas] >= data_inicio) & (df_parametro[coluna_semanas] <= data_fim)]


  #Check de qtd de semanas. Se forem 2 ou menos não rodar.
  if len(df_parametro_filtrado[coluna_semanas].unique()) < 2:
    mensagem = mensagem+colored("\nA quantidade de semanas na base definida é menor que três.\nEscolha um mínimo de três semanas para que o share tenha um melhor espaço amostral.\n", 'red')
    erro = erro+1


  #Criar dois dfs separados somente a data início e data fim
  df_parametro_filtrado_inicio = df_parametro_filtrado[df_parametro_filtrado[coluna_semanas] == data_inicio]
  df_parametro_filtrado_fim = df_parametro_filtrado[df_parametro_filtrado[coluna_semanas] == data_fim]


  #Verificar se a data de piso é segunda-feira
  if data_inicio.weekday() == 0:
  # Verifique se há 7 datas nos DataFrames resultantes
    if len(df_parametro_filtrado_inicio[coluna_datas].unique()) == 0:
      mensagem = mensagem+colored("\nO week_start de piso preenchido no painel de controle não existe na base.\nPreencha um novo piso ou alimente com uma base com a semana de piso completa.\n", 'red')
      erro = erro+1
      #print("O week_start de piso preenchido no painel de controle não existe na base.\nPreencha um novo piso ou alimente com uma base com a semana de piso completa.")
    elif len(df_parametro_filtrado_inicio[coluna_datas].unique()) != 7:
      mensagem = mensagem+colored("\nO week_start de piso de corte não contém todos os 7 dias da semana.\nPreencha um novo piso ou alimente com uma base com a semana de piso completa.\n",'red')
      erro = erro+1
      #print("O week_start de piso de corte não contém todos os 7 dias da semana.\nPreencha um novo piso ou alimente com uma base com a semana de piso completa.")
  else:
    mensagem = mensagem+colored('\nA data de piso preenchida no painel de controle não é uma segunda-feira.\nAltere no painel para uma data que seja segunda-feira.\n','red')
    erro = erro+1
    #print(colored("A data de piso preenchida no painel de controle não é uma segunda-feira.\nAltere no painel para uma data que seja segunda-feira.",'red'))

  #Verificar se a data de teto é segunda-feira
  if data_fim.weekday() == 0:
  # Verifique se há 7 datas nos DataFrames resultantes
    if len(df_parametro_filtrado_fim[coluna_datas].unique()) == 0:
      mensagem = mensagem+colored("\nO week_start de teto preenchido no painel de controle não existe na base.\nPreencha um novo teto ou alimente com uma base com a semana de teto completa.",'red')
      erro = erro+1
      #print("O week_start de teto preenchido no painel de controle não existe na base.\nPreencha um novo teto ou alimente com uma base com a semana de teto completa.")
    elif len(df_parametro_filtrado_fim[coluna_datas].unique()) != 7:
      mensagem = mensagem+colored("\nO week_start de teto de corte não contém todos os 7 dias da semana.\nPreencha um novo teto ou alimente com uma base com a semana de teto completa.",'red')
      erro = erro+1
      #print("O week_start de teto de corte não contém todos os 7 dias da semana.\nPreencha um novo teto ou alimente com uma base com a semana de teto completa.")
  else:
    mensagem = mensagem+colored("\nA data de teto preenchida no painel de controle não é uma segunda-feira.\nAltere no painel para uma data que seja segunda-feira.",'red')
    erro = erro+1
    #print("A data de teto preenchida no painel de controle não é uma segunda-feira.\nAltere no painel para uma data que seja segunda-feira.")

  return mensagem,erro
