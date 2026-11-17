import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mlflow.set_experiment('iris')
with mlflow.start_run():
    n = 100
    clf = RandomForestClassifier(n_estimators=n, random_state=42).fit(Xtr, ytr)
    mlflow.log_param('n_estimators', n)
    mlflow.log_metric('accuracy', clf.score(Xte, yte))
    mlflow.sklearn.log_model(
       clf,
       name='model',
       skops_trusted_types=['sklearn.tree._tree.Tree']
    )
    joblib.dump(clf, 'model.pkl')
    print('Accuracy:', clf.score(Xte, yte))
