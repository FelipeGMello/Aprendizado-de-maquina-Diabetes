import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

dados = pd.read_csv('datasets/diabetes_binary_5050.csv')

X = dados.drop(['Diabetes_binary'], axis=1)
y = dados['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_jobs=10, random_state=42, max_depth=11, min_samples_split=3, n_estimators=500)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de confusão - Random Forest')
plt.show()