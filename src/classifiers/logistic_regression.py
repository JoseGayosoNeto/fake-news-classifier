from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_logistic_regression(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    
    print("Logistic Regression")
    print(f"Acurácia: {accuracy_score(Y_test, y_pred):.4f}")
    print("Relatório de Classificação:")
    print(classification_report(Y_test, y_pred, digits=2, target_names=["Fake News", "Real News"]))
    print("Matriz de Confusão:")
    print(confusion_matrix(Y_test, y_pred))
