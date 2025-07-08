#@title Def formatacao_output_forecast
import pandas as pd

# Retorna a base final no formato de base única, com volume e cohort
def formatacao_output_forecast(base,etapas_coh,etapas_vol,chaves,col_data):

  base= base.fillna(0.0)


  cb_base = list(base.columns.values)

  base[col_data] = base[col_data].apply(lambda x: x.strftime('%Y/%m/%d'))

  etapas_vol_cohort = ["coh_vol_"+i for i in etapas_coh]
  etapas_vol_nao_conv = ["nao_conv_"+i for i in etapas_coh]

  cb_output_cohort = chaves+['Week Origin']+etapas_vol_cohort
  cb_output_coincident = chaves+etapas_vol

  output_cohort = base[cb_output_cohort]
  output_coincident = base.loc[base['Week Origin'] == '0'][cb_output_coincident]

  output_vol_parcial = output_cohort.loc[output_cohort['Week Origin'] != 'Coincident']
  output_vol_parcial = output_vol_parcial.groupby(chaves, as_index=False)[etapas_vol_cohort].sum()
  #print(output_coincident.columns.values)
  #print(output_vol_parcial.columns.values)
  nao_convertido = pd.merge(output_coincident,output_vol_parcial,how='outer',on=chaves)
  nao_convertido = nao_convertido.fillna(0)
  #print(output_cohort.loc[output_cohort[col_data] == '2022/03/14'][[col_data,'Week Origin','coh_vol_os2oa']])
  #print(nao_convertido[[col_data,'os']])
  #print(nao_convertido[[col_data,'coh_vol_os2oa']])


  nao_convertido[etapas_vol_nao_conv] = nao_convertido[etapas_vol[:-1]].values - nao_convertido[etapas_vol_cohort].values
  #print(nao_convertido[[col_data,'nao_conv_os2oa']])
  nao_convertido['Week Origin'] = 'Não Convertido'
  nao_convertido = nao_convertido[chaves+['Week Origin']+etapas_vol_nao_conv+[etapas_vol[-1]]]
  nao_convertido = nao_convertido.rename(columns=dict(zip(etapas_vol_nao_conv, etapas_vol_cohort)))


  output = pd.concat([output_cohort,nao_convertido])


  output = output[chaves+['Week Origin']+etapas_vol_cohort+[etapas_vol[-1]]]

  output = output.rename(columns=dict(zip(etapas_vol_cohort, etapas_coh)))
  cb_final = list(output.columns.values)

  # Remove linhas zeradas:
  output = output.fillna(0)
  output['Soma'] = output[etapas_coh+[etapas_vol[-1]]].sum(axis=1)
  output = output.loc[output['Soma'] != 0,cb_final]


  return output
