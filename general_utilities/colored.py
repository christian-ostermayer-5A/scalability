# Função que retorna um texto colorido no print do colab

def colored(texto,cor):
  if cor == 'red' or cor == 'r':
    #return f'\033[1;31m{texto}\033[0;0;0m'
    return f'\x1b[31m{texto}\x1b[0m'
  elif cor == 'green' or cor == 'g':
    #return f'\033[1;32m{texto}\033[0;0;0m'
    return f'\x1b[32m{texto}\x1b[0m'
  elif cor == 'yellow' or cor == 'y':
    #return f'\033[1;33m{texto}\033[0;0;0m'
    return f'\x1b[33m{texto}\x1b[0m'
  elif cor == 'blue' or cor == 'b':
    #return f'\033[1;34m{texto}\033[0;0;0m'
    return f'\x1b[34m{texto}\x1b[0m'
  else:
    return texto
