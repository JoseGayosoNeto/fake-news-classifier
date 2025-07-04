from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_naive_bayes(X_train, Y_train, X_test, Y_test):
    model = MultinomialNB()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    print("Naive Bayes")
    print(f"Acurácia: {accuracy_score(Y_test, y_pred):.4f}")
    print("Relatório de Classificação:")
    print(classification_report(Y_test, y_pred, digits=2, target_names=["Fake News", "Real News"]))
    print("Matriz de Confusão:")
    cm =confusion_matrix(Y_test, y_pred)

    return cm
