# Classificador binário - Random Forest
### Objetivo
Predizer se uma pessoa é diabética ou não.
### Bibliotecas
Scikit-learn (sklearn), pandas e matplotlib
### Origem dos dados
Os dados vieram do dataset feito por Alex Teboul, no kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset. Vale ressaltar que este dataset é feito sobre outro dataset bem maior, do Centers for Disease Control and Prevention. O nome do dataset é Behavioral Risk Factor Surveillance System: https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system. O caderno contendo os métodos de normalização, limpeza de dados entre outros, está em: https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook
### Metodologia
Visto que o classificador Random Forest não precisa necessariamente de dados normalizados, as únicas preocupações são de campos nulos ou desbalanceamento das classes. Como Alex já fez isso por mim, não preciso me preocupar, pois este dataset é balanceado. Idealmente, a normalização é feita apenas no conjunto de treino, para evitar contaminação de dados, mas como os registros foram feitos em perguntas de sim ou não, os registro acabam virando binários e, de certa forma, normalizado. Para a escolha de hiperparâmetros, validação cruzada em 5 dobras (folds) foi feita no conjunto de treino. Utilizei o método GridSearch que procura exaustivamente a melhor combinação de parâmetros possível. Depois disso, o modelo foi testado através do conjunto teste, o qual ele não conhece e não deve conhecer, caso contrário isso se enquadraria em contaminação de dados. Ou seja, seu modelo deixa de ser confiável.
### Parâmetros utilizados - Random Forest
Parâmetros definidos pro cross-validantion utilizando GridSearch.
max_depth: 11, min_samples_split: 3, n_estimators: 500. O resto é padrão da biblioteca. Observe que não coloquei muitas combinações de parâmetros, pois como o GridSearch busca exaustivamente, significa que ele tenta todas as combinações possíveis, o que pode ser muito pesado e demorar horas para chegar a um resultado. Portanto, tome cuidado ao definir o escopo de parâmetros para serem testados.
