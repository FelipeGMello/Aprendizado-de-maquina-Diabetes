# Classificador binário - Random Forest
### Objetivo
predizer se uma pessoa é diabética ou não
### Bibliotecas
Scikit-learn (sklearn), pandas, 
### Metodologia
Cross-validation em 5 folds com GridSearch para a escolha dos hiperparâmetros
### Parâmetros utilizados - Random Forest
Parâmetros definidos pro cross-validantion utilizando GridSearch.
max_depth: 11, min_samples_split: 3, n_estimators: 500. O resto é padrão da biblioteca.
