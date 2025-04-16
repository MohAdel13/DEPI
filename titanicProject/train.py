from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import utils

# Loading and preparing data
df = utils.loading_data('titanic_cleaned_data.csv')

x_train, x_test, y_train, y_test = utils.splitting_data(df)


# Logistic Model
logistic = LogisticRegression(random_state=42)

logistic.fit(x_train, y_train)

y_train_pred = logistic.predict(x_train)
y_test_pred = logistic.predict(x_test)

metrics = utils.evaluate_model(y_train_pred, y_test_pred, y_train, y_test)

utils.log_model(logistic,
                {'random_state': 42},
                metrics, 'logistic')


# Decision Tree
tree = DecisionTreeClassifier(random_state=42)

param_grid = {
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 5]
}
tree_grid = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

tree_grid.fit(x_train, y_train)


y_train_pred = tree_grid.best_estimator_.predict(x_train)
y_test_pred = tree_grid.best_estimator_.predict(x_test)
metrics = utils.evaluate_model(y_train_pred, y_test_pred, y_train, y_test)


utils.log_model(tree_grid.best_estimator_,
                tree_grid.best_params_,
                metrics, 'decision tree')


