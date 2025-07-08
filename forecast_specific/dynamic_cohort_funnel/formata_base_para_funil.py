#@title Def formata_base_para_funil
import pandas as pd

'''
formata a base de forecast que vem no seguinte formato:

[week_start	city_group	mkt_channel	lead	Etapa	Volume	Volume Aberta	0	1	2	3	4	5	Coincident	%__0	%__5	%__1	%__2	%__3	%__4	%__Coincident	%__Volume Aberta	s__0	s__1	s__2	s__3	s__4	s__5	s__Coincident	GT Aluguel	GT 5A	Meta OP	ordem_semana	Feriado_1	Feriado_2	Feriado_3	Feriado_4	Feriado_5	Feriado_6	Feriado_7	Feriado]

No formato da base que vai para o funil dinâmico:

['week_start' 'city_group' 'mkt_channel' 'lead' 'Week Origin' 'Opp2FL' 'Q'
 'coh_vol_OP2Q' 'OP2Q' 'Q2Opp' 'Opp' 'OP' 'coh_vol_Q2Opp' 'Shifted'
 'coh_vol_Opp2FL' 'FL']

'''

def formata_base_para_funil(df,           # DataFrame completo com a etapa filtrada
                            max_origin,
                            etapa):

  cb_df = list(df.columns.values)

  # Vamos definir as chaves que serão as mesmas em ambas as bases
  chaves = cb_df[:cb_df.index('Etapa')]

  # Vamos definir as conversões cohort da base: 0	1	2	3	4	5	Coincident
  #conversoes_n = cb_df[cb_df.index('0'):cb_df.index('%__0')]
  conversoes_n = [str(x) for x in range(max_origin+1)]+['Coincident']
                             
  # Vamos definir as conversões cohort da base: %__0	%__5	%__1	%__2	%__3	%__4	%__Coincident
  #conversoes_p = cb_df[cb_df.index('%__0'):cb_df.index('%__Volume Aberta')]
  conversoes_p = ['%__'+x for x in conversoes_n]
                             
  # A base final vai ter que conter 1 coluna com os volumes coincident, 1 coluna com os volumes
  # da cohort e outra com as cohorts em %. Assim, vamos começar criando a base com volume coincident
  # e conversões em % e depois criamos uma base no mesmo formato com as conversões em volume e
  # mergeamos as duas:

  # Criando a base com volume coincident e conversões em %
  #--------------------------------------------------------------------------------------------

  # Vamos selecionar apenas as colunas que importam da base completa:
  df_transformada_1 = df[chaves+['Volume']+conversoes_p]

  # Vamos renomear as colunas de conversão:

  df_transformada_1.columns = chaves+['Volume']+conversoes_n

  # Vamos dar um 'melt' nas colunas de conversões e transformar uma coluna só de 'Week Origin':
  df_transformada_1 = df_transformada_1.melt(id_vars = chaves+['Volume'], value_vars = conversoes_n, var_name='Week Origin', value_name = etapa)

  # Renomeamos a coluna de volume para o nome da etapa de volume do funil:
  etapa_vol = etapa.split('2')[0]
  df_transformada_1 = df_transformada_1.rename(columns={'Volume':etapa_vol})


  # Criando a base com volume coincident e conversões em volume
  #--------------------------------------------------------------------------------------------

  # Vamos selecionar apenas as colunas que importam da base completa:
  df_transformada_2 = df[chaves+conversoes_n]

  # Vamos dar um 'melt' nas colunas de conversões e transformar uma coluna só de 'Week Origin':
  df_transformada_2 = df_transformada_2.melt(id_vars = chaves, value_vars = conversoes_n, var_name='Week Origin', value_name = "coh_vol_"+etapa)


  # Unindo as bases:
  #--------------------------------------------------------------------------------------------
  df_transformada = pd.merge(df_transformada_1,df_transformada_2,how='left',on=chaves+['Week Origin'])

  return df_transformada








