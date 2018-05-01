import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing  

 
data = pd.read_csv('/home/valdirluiz/Documents/traballho-dm/dados_selecionados.csv',  sep='|',low_memory=False)


#histograma do tipo de atendimento
data["tipo_atendimento"].value_counts().plot(kind="bar")
#plt.show()

#histograma da classificação da gravidade
data["classificacao_gravidade"].value_counts().plot(kind="bar")
#plt.show()

#histograma da classificação da gravidade
data["manifestacao_clinica"].value_counts().plot(kind="bar")
#plt.show()

#histograma caso_revisado
data["caso_revisado"].value_counts().plot(kind="bar")
#plt.show()

#histograma caso_validado
data["caso_validado"].value_counts().plot(kind="bar")
#plt.show()

#histograma sexo
data["sexo_tratado"].value_counts().plot(kind="bar")
#plt.show()


#histograma obito
data["obito"].value_counts().plot(kind="bar")
#plt.show()

#histograma gestação
data["gestacao"].value_counts().plot(kind="bar")
#plt.show()

#histograma gestação
data["grupo_agente"].value_counts().plot(kind="bar")
#plt.show()


#histograma raca_etnia
data["raca_etnia"].value_counts().plot(kind="bar")
#plt.show()
 
#histograma desfecho
data["desfecho"].value_counts().plot(kind="bar")
#plt.show()

 

#histograma parte_corpo
data["parte_corpo"].value_counts().plot(kind="bar")
#plt.show()


#histograma parte_corpo
data["classe_agente"].value_counts().plot(kind="bar")
#plt.show()

 
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# 12-13 peso normalizado 
#peso = data[data.columns[12:13]] 
#peso_scaled = min_max_scaler.fit_transform(peso)
#peso_normalizado = pd.DataFrame(peso_scaled) 
#peso_normalizado.plot(legend=None)
#plt.xlabel('Identificador do caso')
#plt.ylabel('Peso normalizado')
#plt.show()


# 12-13 peso normalizado  
idade = data[data.columns[19:20]] 
idade_scaled = min_max_scaler.fit_transform(idade)
idade_normalizado = pd.DataFrame(idade_scaled)
print(idade.max()) 
idade_normalizado.plot(legend=None)
plt.xlabel('Identificador do caso')
plt.ylabel('Idade normalizada')
plt.show()
 
 
