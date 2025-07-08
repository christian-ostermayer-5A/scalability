#@title Def transforma_dummy
import numpy as np

'''
Algumas séries exógenas servem para avaliar impactos de momentos no tempo na variável endógena.
Por exemplo, feriados, não possuem um valor exógeno específico, são apenas marcadores temporais de
eventos. Para mensurar o impacto de feriados proporcional ao valor da endógena, não podemos considerar
todos os marcadores com o mesmo valor (=1), pois isso resulta que em qualquer momento da séria, o impacto
vai sempre modificar a série pelo mesmo valor (coef * 1). Para contornar isso, fazemos a variável "dummy"
de marcador temporar ao invés de ser =1, ser uma média dos valores anteriores da variável endógena:
'''
def transforma_dummy(df, # Dataframe filtrado (somente uma etapa e uma abertura especifica, e sem as colunas das mesmas. Somente data e valores)
                     exogenous_dummy, # Lista com o nome das variaveis exogenas que são dummy
                     endogenous, # variável endógena
                     n_avg, # número de pontos passados usados na média móvel da endógena
                     col_data): # String com o nome da coluna de datas

    # Vamos remover "_t" dos nomes das exógenas dummy para serem recalculadas caso tenham vindo de
    # uma base de parâmetros já calculados:
    exogenous_dummy = [d.replace("_t","") for d in exogenous_dummy]

    if len(exogenous_dummy) == 0:
      return df
    else:
      c_df = df.copy()

      # Vamos checar se a coluna de datas já está ordenada:
      #-----------------------------------------------------------------------------------------------
      c_df['Teste'] = c_df[col_data].astype(int)
      teste = c_df['Teste'].astype(int).values[1:]-c_df['Teste'].astype(int).values[:-1]

      # Se não estiver, ordenamos pelas datas
      if c_df['Teste'].values[0] > c_df['Teste'].values[-1] or np.average(teste) != 604800000000000:
        c_df = c_df.sort_values(by=col_data)

      c_df = c_df.drop(columns='Teste')
      #-----------------------------------------------------------------------------------------------

      for exo_d in exogenous_dummy:

        # Vamos garantir que os valores dummy sejam sempre 0 ou 1:
        c_df[exo_d] = c_df[exo_d].astype(float)
        c_df.loc[c_df[exo_d] != 0,exo_d] = 1

        # Multiplicamos o valor dummy (0 ou 1) pela média dos últimos 4 pontos da variável endógena afetada
        # Criamos uma coluna especial somente com a serie dummy transformada
        c_df[exo_d+"_t"] = np.append(np.array([0.]) , c_df[endogenous].rolling(n_avg).mean().astype(float).values[:-1]) * c_df[exo_d].astype(float).values
        c_df[exo_d+"_t"] = c_df[exo_d+"_t"].fillna(0)


      return c_df
