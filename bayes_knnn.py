# Importar librerías necesarias
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Cargar datasets
iris = load_iris(as_frame=True)
wine = load_wine(as_frame=True)
cancer = load_breast_cancer(as_frame=True)
data = pd.read_csv('bezdekIris.data', header = None)

datasets = {
    "Iris": (iris['data'], iris['target']),
    "Wine": (wine['data'], wine['target']),
    "Cancer": (cancer['data'], cancer['target'])
}

# Separar características (X) y etiquetas (y)
X1 = data.iloc[:, :-1].values  # Todas las columnas menos la última
y1 = data.iloc[:, -1].values   # La última columna

# Codificar las etiquetas en valores numéricos (si es necesario)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y2 = label_encoder.fit_transform(y1)

# Configurar clasificadores
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5)  # Usamos K=5 para KNN

# Función para Hold Out 70/30
def hold_out_70_30(X, y, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm

# Función para 10-Fold Cross-Validation
def k_fold_cv(X, y, classifier, k=10):
    kf = KFold(n_splits=k, random_state=42, shuffle=True)
    scores = cross_val_score(classifier, X, y, cv=kf)
    return scores.mean()

# Función para Leave-One-Out
def loo_cv(X, y, classifier):
    loo = LeaveOneOut()
    scores = cross_val_score(classifier, X, y, cv=loo)
    return scores.mean()

# Evaluar clasificadores en todos los datasets y métodos de validación
results = {}

# Evaluar clasificadores
results1 = {
    "Naive Bayes": {
        "Hold Out": hold_out_70_30(X1, y2, gnb),
        "10-Fold CV": k_fold_cv(X1, y2, gnb),
        "Leave-One-Out": loo_cv(X1, y2, gnb)
    },
    "KNN": {
        "Hold Out": hold_out_70_30(X1, y2, knn),
        "10-Fold CV": k_fold_cv(X1, y2, knn),
        "Leave-One-Out": loo_cv(X1, y2, knn)
    }
}

for name, (X, y) in datasets.items():
    results[name] = {
        "Naive Bayes": {
            "Hold Out": hold_out_70_30(X, y, gnb),
            "10-Fold CV": k_fold_cv(X, y, gnb),
            "Leave-One-Out": loo_cv(X, y, gnb)
        },
        "KNN": {
            "Hold Out": hold_out_70_30(X, y, knn),
            "10-Fold CV": k_fold_cv(X, y, knn),
            "Leave-One-Out": loo_cv(X, y, knn)
        }
    }

# Mostrar resultados
for dataset_name, classifiers in results.items():
    print(f"Dataset: {dataset_name}")
    for clf_name, methods in classifiers.items():
        print(f"  Classifier: {clf_name}")
        print(f"    Hold Out - Accuracy: {methods['Hold Out'][0]:.4f}, Confusion Matrix: \n{methods['Hold Out'][1]}")
        print(f"    10-Fold CV - Accuracy: {methods['10-Fold CV']:.4f}")
        print(f"    Leave-One-Out - Accuracy: {methods['Leave-One-Out']:.4f}")

# Mostrar resultados
print('KNN y BAYES con datos de bezdekiris.data')
for clf_name, methods in results1.items():
    print(f"Classifier: {clf_name}")
    print(f"  Hold Out - Accuracy: {methods['Hold Out'][0]:.4f}, Confusion Matrix: \n{methods['Hold Out'][1]}")
    print(f"  10-Fold CV - Accuracy: {methods['10-Fold CV']:.4f}")
    print(f"  Leave-One-Out - Accuracy: {methods['Leave-One-Out']:.4f}")