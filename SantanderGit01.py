# Passo 1: Importar bibliotecas

import pandas as pd #Para manipulação de dados
from sklearn.model_selection import train_test_split #para dividir os dados em conjuntos de treino
from sklearn.preprocessing import StandardScaler, OrdinalEncoder #Standard para normalizar os dados/ Ordinal para codificação das variaveias categoricas
from sklearn.linear_model import LogisticRegression #Para construir o modelo de regressão lógica
from sklearn.metrics import classification_report, roc_auc_score #para avaliar o desempenho do modelo

# Passo 2: Carregar os dados de treino e teste
train_data = pd.read_csv('/train.csv') #Não esqueça de colocar o endereçamento certo do arquivo, tem o link no README.md
test_data = pd.read_csv('/test.csv')#Não esqueça de colocar o endereçamento certo do arquivo, tem o link no README.md

# Passo 3: Separar os dados! Separar Features (X) e Target(y)
X = train_data.drop(['target'], axis=1)
y = train_data['target']

# Passo 4: identificar as colunas categoricas
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Passo 5: Codificar colunas categóricas usando OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

# Passo 6: Tratar os dados ausentes(Por via das dúvidas)
X.fillna(X.mean(), inplace=True) #Colocando a média das colunas nos dados ausentes.

# Passo 7: Dividir o conjunto de treino em treino e validação (Procedimento padrão)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# test_size=0.2 significa que 20% dos dados são usados para a validação
#random_state=42 Garante que a divisão seja reprodutível

# Passo 7: Normalizar os dados(por via das dúvidas)
scaler = StandardScaler() #Normaliza os dados para que cada feature tenha media 0 e desvio padrão 1
X_train = scaler.fit_transform(X_train) #fit_transform é aplicado ao conjunto de treino para calcular a média e desvio padrão e transformar os dados
X_val = scaler.transform(X_val) #transform é aplicado ao conjunto de validação usando a mesma media e desvio padrão que calculamos a partir do conjunto de treino

# Passo 8: Treinar e avaliar o modelo
# treinando o modelo de regressão logica
model = LogisticRegression()
model.fit(X_train, y_train)
# Inicializamos o modelo de regressão logicae treinamos o modelo com os dados de treino

# Passo 9: Fazendo previsões no conjunto de validação e avaliando o desempenho do modelo
y_pred = model.predict(X_val)

# Avaliando o desempenho do modelo
print("Classification Report:")
print(classification_report(y_val, y_pred))

print("\nAUC-ROC Score:", roc_auc_score(y_val, y_pred))

#  Passo 10: Visualizando os resultados
