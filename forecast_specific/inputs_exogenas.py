#@title Def inputs_exogenas

def inputs_exogenas(df_inputs_exogenas,abertura,etapa,endogenous,exog_list):

  cb_inputs = list(df_inputs_exogenas.columns.values)
  cb_inputs.pop(-3)

  df_i_exo = df_inputs_exogenas.copy()

  # Vamos substituir a aplicação em 'Todos' pela abertura que estamos analisando:
  i=0
  for c in cb_inputs[:-1]:
    if i < len(abertura):
      replace = abertura[i]
    elif i == len(abertura):
      replace = etapa
    else:
      replace = endogenous
    i=i+1
    df_i_exo[c] = df_i_exo[c].replace('Todos',replace)

  # Vamos substituir o nome das exógenas na base de inputs pelos nomes que usamos:
  df_i_exo['exógena'] = df_i_exo['exógena'].replace('Feriado','Feriado (dummy)')

  # Vamos substituir o nome das endógenas na base de inputs pelos nomes que usamos:
  df_i_exo['endógena'] = df_i_exo['endógena'].replace('Cohort Aberta','%__Volume Aberta')
  for r in range(5): # @ Aqui substituir por automático de max origin
    orginal = "Share W"+str(r)
    replace = "s__"+str(r)
    df_i_exo['endógena'] = df_i_exo['endógena'].replace(orginal,replace)


  # Agora vamos filtrar apenas as aberturas, etapas e endógenas que iremos usar:
  i=0
  for c in cb_inputs[:-2]:
    if i < len(abertura):
      filtro = abertura[i]
    elif i == len(abertura):
      filtro = etapa
    else:
      filtro = endogenous
    i=i+1
    df_i_exo = df_i_exo.loc[df_i_exo[c] == filtro]

  # Vamos adicionar a informação de lag ou diferenciação nas séries exógenas:
  lista_exogenas = list(df_i_exo['exógena'].values)
  lista_lag = list(df_i_exo['lag ou diff'].values)
  for l in range(len(lista_exogenas)):
    if lista_lag[l] == "0" or lista_lag[l] == "":
      lista_exogenas[l] = lista_exogenas[l]
    elif lista_lag[l] == "d":
      lista_exogenas[l] = lista_exogenas[l]+"___d"
    elif lista_lag[l] == "log":
      lista_exogenas[l] = lista_exogenas[l]+"___log"
    else:
      lista_exogenas[l] = lista_exogenas[l]+"___l___"+lista_lag[l]
  df_i_exo['exógena'] = lista_exogenas

  # Vamos excluir as séries exógenas
  df_excluir = df_i_exo.loc[df_i_exo['ação'] == 'Excluir']
  if len(df_excluir) != 0:
    exog_list_ex = list(df_excluir['exógena'].values)
    if 'Todos' in exog_list_ex:
      exog_list = []
    else:
      exog_list = list(set(exog_list) - set(exog_list_ex))


  # Vamos incluir as séries exógenas
  df_incluir = df_i_exo.loc[df_i_exo['ação'] == 'Incluir']
  if len(df_incluir) != 0:
    exog_list_in = list(df_incluir['exógena'].values)
    exog_list = exog_list+exog_list_in
    exog_list = list(dict.fromkeys(exog_list)) # remove duplicados

  # Vamos isolar as séries exógenas
  df_isolar = df_i_exo.loc[df_i_exo['ação'] == 'Isolar']
  if len(df_isolar) != 0:
    exog_list = list(df_isolar['exógena'].values)
    try:
      exog_list.remove('Todos')
    except:
      exog_list = exog_list


  return exog_list
