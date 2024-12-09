#@title Def abertura_do_arquivo

'''
Essa função serve para abrir um arquivo sheets.
A função tenta remover erros no nome do arquivo ants de abrir.
Se não conseguir abrir, retorna uma mensagem de erro
'''
# Importando funções auxiliares:
from colored import colored
import pandas as pd
from IPython.display import clear_output
# A autorização "client" já deve ter sido configurada no notebook

def abertura_do_arquivo(nome_do_arquivo, # string com o nome do arquivo sheets a ser aberto
                       sheets_aberto,    # Booleano indicando se o sheets já foi aberto
                       sheets,           # Arquivo sheets já aberto
                       nome_da_aba,      # string com o nome da aba a ser aberta
                       client = 'client'):          # Autorização de acesso
  
  mensagem = ''

  # Tentativa de abrir o arquivo sheets
  flag_abriu = True
  if not sheets_aberto:
    try:
      nome_do_arquivo_original = nome_do_arquivo

      if "https://" in nome_do_arquivo:
        id_do_arquivo = nome_do_arquivo.split("/")[5]

        # Abrindo o arquivo pelo ID
        dados_sheets = client.open_by_key(id_do_arquivo)

      else:
        if "'" in nome_do_arquivo:
          nome_do_arquivo = nome_do_arquivo.replace("'",'')
        
        if len(nome_do_arquivo) < len(nome_do_arquivo_original):
          mensagem = mensagem+'\n\nO nome do arquivo sheets "'+colored(nome_do_arquivo_original,'blue')+'" possui aspas simples.\nAs mesmas foram removidas e será aberto o arquivo "'+colored(nome_do_arquivo,'blue')+'"'
          nome_do_arquivo_original = nome_do_arquivo

        if "\"" in nome_do_arquivo:
          nome_do_arquivo = nome_do_arquivo.replace("\"",'')

        if len(nome_do_arquivo) < len(nome_do_arquivo_original):
          mensagem = mensagem+'\n\nO nome do arquivo sheets "'+colored(nome_do_arquivo_original,'blue')+'" possui aspas duplas.\nAs mesmas foram removidas e será aberto o arquivo "'+colored(nome_do_arquivo,'blue')+'"'
          nome_do_arquivo_original = nome_do_arquivo

        nome_do_arquivo = nome_do_arquivo.strip()
        
        if len(nome_do_arquivo) < len(nome_do_arquivo_original):
          mensagem = mensagem+'\n\nO nome do arquivo sheets "'+colored(nome_do_arquivo_original,'blue')+'" possui espaços em branco.\nOs mesmos foram removidos e será aberto o arquivo "'+colored(nome_do_arquivo,'blue')+'"'
        
        # Abrindo o arquivo pelo nome
        dados_sheets = client.open(nome_do_arquivo)
      
      mensagem = mensagem+'\n\n'+colored('Sheets Aberto','green')
    except Exception as e:
      if str(e) == '':
        e = 'SpreadsheetNotFound'
      flag_abriu = False
      dados_sheets = []
      mensagem = mensagem+'\n\nNão foi possível abrir o arquivo sheets '+colored(nome_do_arquivo,'blue')+'.\nCertifique-se de que escreveu o nome do arquivo corretamente e também se seu usuário possui acesso ao arquivo.\nA tentativa de abrir o arquivo resultou no seguinte erro:\n'+colored(str(e),'red')
  else: 
    dados_sheets = sheets


  # Tentativa de abrir a aba:
  base = []
  if (flag_abriu and not sheets_aberto and nome_da_aba != '') or sheets_aberto:
    try:
      base = dados_sheets.worksheet(nome_da_aba)
    except:
      mensagem = mensagem+'\n\nNão foi encontrada a aba '+colored(nome_da_aba,'yellow')+' no arquivo '+colored(nome_do_arquivo,'blue')+'.\nCertifique-se que escreveu o nome da aba corretamente.'
      flag_abriu = False

  # retornamos, além do arquivo sheets aberto, a base da aba (se houver), um booleano indicando se a abertura deu certo e o nome do arquivo corrigido
  return dados_sheets,base,mensagem,flag_abriu,nome_do_arquivo






#@title Def abertura_das_bases

'''
Essa função serve para abrir e transformar em DataFrame uma lista de bases.

A lista de bases podem estar em vários arquivos sheets diferentes ou na forma de CSV em alguma pasta do Google Drive
'''

def abertura_das_bases(lista_de_nomes_das_bases,  # lista de strings com os nomes das bases (nomes dos arquivos csv ou das abas no sheets)
                       lista_de_arquivos_das_bases, # lista de strings com os nomes dos arquivos onde estão as bases (nome dos sheets ou o caminho onde se encontra o csv)
                       colunas_para_dropar,       # lista de listas de inteiros, indicando a posição das colunas a serem excluídas de cada base (lista vazia caso nenhuma coluna a ser excluída)
                       lista_obrigatoria,         # lista de booleanos indicando se a existência daquela base é obrigatória
                       nome_do_painel_de_controle,           # string com o nome do arquivo sheets onde se encontra o painel de controle (somente para imprimir mensagens de erro)
                       sheets_painel_de_controle,           # arquivo sheets do painel de controle aberto
                       client = 'client',                  # Autorização de acesso
                       nomes_teoricos_das_bases = []):                  # nomes_teoricos_das_bases
  

  numero_de_bases = len(lista_de_nomes_das_bases)
  lista_de_bases = []
  
  qtd_bases_existentes = 0
  for x in range(len(lista_de_nomes_das_bases)):
    if lista_de_nomes_das_bases[x] != '':
      qtd_bases_existentes+=1


  # Para cada base na lista de bases, vamos tentar abrir o arquivo:
  #_________________________________________________________________________________________________
  i=0
  qtd=0
  lista_bases_abertas = []
  for nome_da_base in lista_de_nomes_das_bases:

    nome_aba = nome_da_base

    if len(nomes_teoricos_das_bases) == len(lista_de_nomes_das_bases):
      nome_da_base = str(nomes_teoricos_das_bases[i])+' - ('+nome_aba+')'
    else:
      nome_da_base = nome_aba
    

    if not lista_obrigatoria[i] and nome_aba == '':

      base_df = pd.DataFrame()

      base_df.name = nome_da_base

      lista_de_bases = lista_de_bases+[base_df]

      i+=1
    
    else:
      string_print = "Abrindo "+colored(nome_da_base,'y')+": "
      empty_string = " "*(80-len(string_print))
      print("Abrindo "+colored(nome_da_base,'y')+": "+empty_string+str(qtd+1)+'/'+str(qtd_bases_existentes),end="\r")
      lista_bases_abertas = lista_bases_abertas+[nome_da_base]
      qtd+=1


      # Caso exista "/" no nome do arquivo, vamos considerar que se trata de um caminho para
      # uma pasta do google drive onde se encontra um csv que vamos tentar abrir:
      #-----------------------------------------------------------------------------------------------
      if '/' in lista_de_arquivos_das_bases[i] and "https://" not in lista_de_arquivos_das_bases[i]:
        
        # iniciamos uma flag para indicar se conseguimos abrir a base
        flag_abriu = False

        # adicionamos os diretórios obrigatórios no caminho original fornecido
        path = '/content/drive/'+lista_de_arquivos_das_bases[i]+'/'

        # removemos a terminação ".csv" do nome do arquivo caso o usuário tenha inputado dessa forma
        nome_aba = nome_aba.replace('.csv','')

        # tentamos abrir o arquivo
        try:
          base_df=pd.read_csv(path+nome_aba+'.csv')
          flag_abriu = True
        except Exception as e:
          print('\n\nNão foi possível abrir o arquivo '+colored(nome_da_base,'blue')+'.\nFoi identificado que esta é uma tentativa de abrir um arquivo csv localizado na seguinte pasta do Google Drive:\n'+colored(lista_de_arquivos_das_bases[i],'red')+'\nA tentativa de abrir o arquivo resultou no seguinte erro:\n'+colored(str(e),'red'))

        # caso o arquivo tenha sido aberto, transformamos a base em DataFrame
        if flag_abriu:
          
          # removemos a coluna com o número da linha
          base_df = base_df.drop(base_df.index[0])

          # caso existam colunas para serem removidas, removemos elas aqui:
          if len(colunas_para_dropar) != 0:
            cb = base_df.columns.values
            dropar = cb[colunas_para_dropar[i]]
            base_df = base_df.drop(columns=dropar)

          # definimos o nome da base
          base_df.name = nome_da_base

          # adicionamos a base na nossa lista de bases
          lista_de_bases = lista_de_bases+[base_df]

        i+=1


      # Caso o nome do arquivo não seja um caminho, tentamos abrir um sheets com o nome indicado:
      #-----------------------------------------------------------------------------------------------
      else:
        
        # Caso o nome do arquivo seja "Atual", vamos procurar a base no sheets do painel de controle
        if lista_de_arquivos_das_bases[i] == 'Atual' or lista_de_arquivos_das_bases[i] == 'atual':

          dados_sheets,base,mensagem,flag_abriu,nome_do_arquivo = abertura_do_arquivo(nome_do_arquivo = nome_do_painel_de_controle,
                                                                                      sheets_aberto = True,
                                                                                      sheets = sheets_painel_de_controle,
                                                                                      nome_da_aba = nome_aba,
                                                                                      client = client)
        # Caso contrário, procuramos um outro sheets:
        else:

          dados_sheets,base,mensagem,flag_abriu,nome_do_arquivo = abertura_do_arquivo(nome_do_arquivo = lista_de_arquivos_das_bases[i],
                                                                                      sheets_aberto = False,
                                                                                      sheets = [],
                                                                                      nome_da_aba = nome_aba,
                                                                                      client = client)
        
        # Caso conseguimos abrir o sheets e a aba dentro dele, vamos transformar em DataFrame:
        if flag_abriu:
          rows = base.get_all_values()
          base_df = pd.DataFrame.from_records(rows)
          try:
            base_df.columns = base_df.iloc[0]
            base_df = base_df.drop(base_df.index[0])

            if len(colunas_para_dropar) != 0:
              cb = base_df.columns.values
              dropar = cb[colunas_para_dropar[i]]
              base_df = base_df.drop(columns=dropar)
          
          except:
            base_df = pd.DataFrame()

          base_df.name = nome_da_base

          lista_de_bases = lista_de_bases+[base_df]

          i+=1

        else:
          if lista_obrigatoria[i]:
            print(mensagem)
            break
          else:
            base_df = pd.DataFrame()
            
            base_df.name = 'base_vazia'
  
            lista_de_bases = lista_de_bases+[base_df]
  
            i+=1

                         
            


  clear_output(wait=True)
  if len(lista_de_bases) == numero_de_bases:
    flag_abriu = True
    print("\r", end="")
    if len(lista_de_bases) > 1:
      string_print = colored("Bases Abertas: ","green")
      empty_string = " "*(80-len(string_print))
      print(string_print+empty_string,str(qtd_bases_existentes)+'/'+str(qtd_bases_existentes),end="")
      print('\n')
      for base in lista_bases_abertas:
        print(colored(base,'b'))
    else:
      print(colored(nome_da_base+' aberto',"green"),end="")
  else:
    flag_abriu = False
  
  return lista_de_bases,flag_abriu

