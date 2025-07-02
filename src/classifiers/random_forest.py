from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_random_forest(X_train, Y_train, X_test, Y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    print("Random Forest")
    print(f"Acurácia: {accuracy_score(Y_test, y_pred):.4f}")
    print("Relatório de Classificação:")
    print(classification_report(Y_test, y_pred, digits=2, target_names=["Fake News", "Real News"]))
    print("Matriz de Confusão:")
    print(confusion_matrix(Y_test, y_pred))
