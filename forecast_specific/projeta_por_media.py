#@title Def projeta_por_media
import numpy as np
# o df tem que estar ordenado por data com a mais antiga no topo
def projeta_por_media(df,
                      endogenous,
                      qtd_semanas_projetadas,
                      qtd_semanas_media):

  if endogenous == '%__Volume Aberta':
    #serie_aberta = df['Volume Aberta'].astype('float').values
    serie_volume = df['Volume'].astype('float').values
    # Devemos multiplicar a cohort aberta % pelos volumes, pois,
    # caso tenha sido feita uma projeção das cohorts não-maturadas
    # ou uma remoção de outliers,
    # precisamos incluir essa correção nos volumes.
    serie_aberta = df['%__Volume Aberta'].astype('float').values * serie_volume
    

    if len(serie_aberta) >= qtd_semanas_media:
      serie_aberta = serie_aberta[-qtd_semanas_media:]
      serie_volume = serie_volume[-qtd_semanas_media:]
    
    avg = np.sum(serie_aberta)/np.sum(serie_volume)
  
  elif endogenous == 's__0':
    serie_aberta = df['Volume Aberta'].astype('float').values
    serie_0 = df['0'].astype('float').values    

    if len(serie_aberta) >= qtd_semanas_media:
      serie_aberta = serie_aberta[-qtd_semanas_media:]
      serie_0 = serie_0[-qtd_semanas_media:]

    avg = np.sum(serie_0)/np.sum(serie_aberta)

  else:

    serie = df[endogenous].astype('float').values

    if len(serie) >= qtd_semanas_media:
      serie = serie[-qtd_semanas_media:]
      
    avg = np.mean(serie)

    # Precisamos gerar um número irrisório de volume caso o histórico venha zerado
    # para conseguir gerar um baseline:
    if endogenous == 'Volume' and (len(serie) < qtd_semanas_media or avg == 0):
      avg = 0.00000123
    
                        
  forecast = np.zeros(shape=(int(qtd_semanas_projetadas)))
  forecast[:] = avg

  return forecast
