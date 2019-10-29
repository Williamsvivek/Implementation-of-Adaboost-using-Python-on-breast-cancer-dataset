from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
cancer = datasets.load_breast_cancer()
print("Features:", cancer.feature_names)
print("Labels:", cancer.target_names)
print(cancer.data.shape)
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3)
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model = abc.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("##############################################################")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Kappa Stats:",metrics.cohen_kappa_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Mean Absolute Error:",metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:",metrics.mean_squared_error(y_test, y_pred))
print("F-Measure:",metrics.recall_score(y_test, y_pred))
print("##############################################################")