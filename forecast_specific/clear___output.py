#@title Definições de Manutenção
from IPython.display import clear_output
flag_clear = True # Apaga ou mantém todos os prints. Deixar como "False" quando quiser fazer manutenção do código e imprimir coisas internas
def clear___output(flag):
  if flag:
    clear_output(wait=True)
