import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

dados = pd.read_csv('datasets/diabetes_binary_5050.csv')

X = dados.drop(['NoDocbcCost', 'Education', 'Diabetes_binary'], axis=1)
y = dados['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

param_grid = [
    {'n_estimators': [100, 300, 500, 700],
      'max_depth': ['None', 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
]

modelo = RandomForestClassifier(n_jobs=10, random_state=42, criterion='gini', max_features='sqrt')
grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train,y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

y_pred = grid_search.best_estimator_.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(ConfusionMatrixDisplay(cm, display_labels=["FALSE", "TRUE"]).plot())