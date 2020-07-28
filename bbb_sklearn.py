from math import sqrt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, tree
import xgboost as xgb
import graphviz
from pyrsm import gains, gains_plot, lift, lift_plot, confusion, profit_max, ROME_max

bbb = pd.read_pickle("data/bbb.pkl")

bbb["buyer_yes"] = (bbb["buyer"] == "yes").astype(int)
bbb["gender_male"] = (bbb["gender"] == "M").astype(int)

# adding a random number that should be dropped by lasso
bbb["rnd"] = np.random.randn(bbb.shape[0])
bbb.head()

rvar = "buyer_yes"
evar = [
    "gender_male",
    "last",
    "total",
    "child",
    "youth",
    "cook",
    "do_it",
    "reference",
    "art",
    "geog",
    "rnd",
]
idvar = "acctnum"
lev = "yes"

# not required - just an example of how you would do this in python
bbb.drop(columns=["training"])
train, test = train_test_split(
    bbb, test_size=0.3, random_state=1234, stratify=bbb["buyer_yes"]
)

# stratification insures the proportion of buyers is (almost) identical
train["buyer_yes"].sum() / train.shape[0]
test["buyer_yes"].sum() / test.shape[0]

# adding the (new) training variable to the dataset
train.loc[:, "training"] = 1
test.loc[:, "training"] = 0

# storing results for evaluation of different models
eval_dat = pd.concat([train, test], axis=0)
eval_dat = eval_dat[[idvar, rvar, "training"]]

X_train = train[evar]
y_train = train[rvar]
X_test = test[evar]
y_test = test[rvar]

# scaling the training data
scaler = StandardScaler()
scaler.fit(X_train)

# apply transformation to training data
Xs_train = scaler.transform(X_train)

# apply same transformation to test data
Xs_test = scaler.transform(X_test)

# combining so we can predict for the full dataset as well
Xs = np.concatenate((Xs_train, Xs_test), axis=0)

# Logistic regression with strong L1 regularization
clf = LogisticRegression(
    random_state=1234, max_iter=1000, solver="saga", penalty="l1", C=0.01
)

clf.fit(Xs_train, y_train)
coef = clf.fit(Xs_train, y_train).coef_
coef = pd.DataFrame({"labels": evar, "coefficients": coef[0], "OR": np.exp(coef[0])})

# Effect of the 'rnd' variable set to 0
coef.query("coefficients == 0")
lr_proba = clf.predict_proba(Xs)
eval_dat["y_lr"] = lr_proba[:, 1]

# Logistic regression with LASSO
clf = LogisticRegression(random_state=1234, max_iter=1000, solver="saga", penalty="l1")
param_grid = dict(
    C=[1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025], class_weight=["None"]
)
lr_lasso_search = GridSearchCV(clf, param_grid, cv=3, scoring="roc_auc")
lr_lasso_search.fit(Xs_train, y_train)

results = pd.DataFrame(lr_lasso_search.cv_results_)
results = results.sort_values(by=["rank_test_score"])
results

lr_lasso_best = LogisticRegression(
    random_state=1234,
    max_iter=1000,
    solver="saga",
    penalty="l1",
    C=results.loc[0, "param_C"],
    class_weight="None",
)

lr_lasso_best.fit(Xs_train, y_train)
lr_lasso_best.fit(Xs_train, y_train).intercept_
coef = lr_lasso_best.fit(Xs_train, y_train).coef_
pd.DataFrame({"labels": evar, "coefficients": coef[0], "OR": np.exp(coef[0])})

lasso_proba = lr_lasso_best.predict_proba(Xs)
eval_dat["y_lasso"] = lasso_proba[:, 1]

# Neural net from SKLEARN
clf = MLPClassifier(
    solver="adam",
    learning_rate_init=0.01,
    alpha=0.01,
    hidden_layer_sizes=(2, 2),
    random_state=1234,
    max_iter=1000,
)
clf.fit(Xs_train, y_train)

nn_proba = clf.predict_proba(Xs)
eval_dat["y_nn"] = nn_proba[:, 1]
eval_dat

# CV for NN
nr_hnodes = range(4, 6)
hls = (
    list(zip(nr_hnodes))
    + list(zip(nr_hnodes, nr_hnodes))
    + list(zip(nr_hnodes, nr_hnodes, nr_hnodes))
)
hls

param_grid = {"hidden_layer_sizes": hls, "alpha": [0.001, 0.01, 0.05]}
scoring = {"AUC": "roc_auc"}

clf_cv = GridSearchCV(
    clf, param_grid, scoring=scoring, cv=5, n_jobs=4, refit="AUC", verbose=5
)
clf_cv.fit(Xs_train, y_train)

clf_cv.best_params_
clf_cv.best_score_
results = pd.DataFrame(clf_cv.cv_results_)
results = results.sort_values(by=["rank_test_AUC"])
results

nn_proba = clf_cv.best_estimator_.predict_proba(Xs)
eval_dat["y_nn_cv"] = nn_proba[:, 1]
eval_dat

# prediction on training set
pred = clf_cv.predict_proba(Xs_train)
fpr, tpr, thresholds = metrics.roc_curve(y_train.values, pred[:, 1])
auc_rf = metrics.auc(fpr, tpr)
auc_rf

# prediction on test set
pred = clf_cv.predict_proba(Xs_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test.values, pred[:, 1])
auc_rf = metrics.auc(fpr, tpr)
auc_rf

# estimating and tuning a tree model
clf = tree.DecisionTreeClassifier(max_depth=2)
clf.fit(X_train, y_train)

ret = tree.plot_tree(clf.fit(X_train, y_train))

# alternative plotting option
dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=X_train.columns,
    class_names=["yes", "no"],
    proportion=True,  # show probabilities
    rounded=True,
    filled=True,
)
graph = graphviz.Source(dot_data)
graph.render("bbb_tree")

# re-estimate a larger tree
clf = tree.DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)

# predict for the entire dataset
tree_proba = clf.predict_proba(pd.concat((X_train, X_test), axis=0))
eval_dat["y_tree"] = tree_proba[:, 1]

# CV for trees
param_grid = {"max_depth": list(range(2, 20))}
scoring = {"AUC": "roc_auc"}

clf_cv = GridSearchCV(
    clf, param_grid, scoring=scoring, cv=5, n_jobs=4, refit="AUC", verbose=5
)
clf_cv.fit(Xs_train, y_train)

clf_cv.best_params_
clf_cv.best_score_
results = pd.DataFrame(clf_cv.cv_results_)
results = results.sort_values(by=["rank_test_AUC"])
results

# predict for the entire dataset
tree_proba = clf_cv.best_estimator_.predict_proba(pd.concat((X_train, X_test), axis=0))
eval_dat["y_tree_cv"] = tree_proba[:, 1]

# Random Forest
# how many features to include? use sqrt(N) as an approximation
sqrt(X_train.shape[1])
clf = RandomForestClassifier(
    n_estimators=100, max_features=3, oob_score=True, random_state=1234
)
clf.fit(X_train, y_train)

# *not* using OOB values here and check out the AUC value!!!
pred = clf.predict_proba(X_train)
fpr, tpr, thresholds = metrics.roc_curve(y_train.values, pred[:, 1])
auc_rf = metrics.auc(fpr, tpr)
auc_rf

# using OOB scores instead
pred = clf.oob_decision_function_
fpr, tpr, thresholds = metrics.roc_curve(y_train.values, pred[:, 1])
auc_rf = metrics.auc(fpr, tpr)
auc_rf

# prediction on test set
pred = clf.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test.values, pred[:, 1])
auc_rf = metrics.auc(fpr, tpr)
auc_rf

# predict for the entire dataset
rf_proba = clf.predict_proba(pd.concat((X_train, X_test), axis=0))
eval_dat["y_rf"] = rf_proba[:, 1]

# Random Forest with cross validation and grid search
clf = RandomForestClassifier()
param_grid = {
    "n_estimators": list(range(100, 501, 100)),
    "max_features": np.arange(3, 5),
}
scoring = {"AUC": "roc_auc"}

# class_weight="balanced_subsample", n_estimators=20, max_depth=5, random_state=1
clf_cv = GridSearchCV(
    clf, param_grid, scoring=scoring, cv=5, n_jobs=4, refit="AUC", verbose=5
)
clf_cv.fit(X_train, y_train)
clf_cv.best_params_
clf_cv.best_score_

# *not* using OOB values here, again, note the AUC value!!!
pred = clf_cv.predict_proba(X_train)
fpr, tpr, thresholds = metrics.roc_curve(y_train.values, pred[:, 1])
auc_rf = metrics.auc(fpr, tpr)
auc_rf

# using OOB scores instead ... but have to re-estimate because it does not seem
# possible to pass the 'oob_score' option when using GridSearchCV
clf_cv.best_params_
clf = RandomForestClassifier(
    n_estimators=clf_cv.best_params_["n_estimators"],
    max_features=clf_cv.best_params_["max_features"],
    oob_score=True,
    random_state=1234,
)
clf.fit(X_train, y_train)
pred = clf.oob_decision_function_
fpr, tpr, thresholds = metrics.roc_curve(y_train.values, pred[:, 1])
auc_rf = metrics.auc(fpr, tpr)
auc_rf

# prediction on test set
pred = clf_cv.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test.values, pred[:, 1])
auc_rf = metrics.auc(fpr, tpr)
auc_rf

# predict for the entire dataset
rf_proba = clf_cv.best_estimator_.predict_proba(pd.concat((X_train, X_test), axis=0))
eval_dat["y_rf_cv"] = rf_proba[:, 1]

# XGBoost
clf = xgb.XGBClassifier(
    max_depth=2,
    n_estimators=500,
    early_stopping_rounds=10,
    eval_metric="auc",
    random_state=1234,
)
clf.fit(X_train, y_train, verbose=True)

# prediction on training set
pred = clf.predict_proba(X_train)
fpr, tpr, thresholds = metrics.roc_curve(y_train.values, pred[:, 1])
auc_xgb = metrics.auc(fpr, tpr)
auc_xgb

# prediction on test set
pred = clf.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test.values, pred[:, 1])
auc_xgb = metrics.auc(fpr, tpr)
auc_xgb

# predict for the entire dataset
xgb_proba = clf.predict_proba(pd.concat((X_train, X_test), axis=0))
eval_dat["y_xgb"] = xgb_proba[:, 1]

# XGBoost with cross-validation and grid search
clf = xgb.XGBClassifier()
param_grid = {
    "max_depth": list(range(1, 3)),
    "n_estimators": list(range(100, 301, 100)),
}
scoring = {"AUC": "roc_auc"}

clf_cv = GridSearchCV(
    clf, param_grid, scoring=scoring, cv=5, n_jobs=4, refit="AUC", verbose=5
)
clf_cv.fit(X_train, y_train)

print(clf_cv.best_params_)
print(clf_cv.best_score_)

# evaluation on training data
pred = clf_cv.predict_proba(X_train)
fpr, tpr, thresholds = metrics.roc_curve(y_train.values, pred[:, 1])
auc_xgb = metrics.auc(fpr, tpr)
auc_xgb

# prediction on test set
pred = clf_cv.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test.values, pred[:, 1])
auc_xgb = metrics.auc(fpr, tpr)
auc_xgb

# predict for the entire dataset
xgb_proba = clf_cv.best_estimator_.predict_proba(pd.concat((X_train, X_test), axis=0))
eval_dat["y_xgb_cv"] = xgb_proba[:, 1]

# performance evaluations

models = eval_dat.columns.values[3:]
models

# calculate gains for all models
[[m, gains(eval_dat.query("training == 0"), "buyer_yes", 1, m, qnt=10)] for m in models]

# calculate lift for all models
[[m, lift(eval_dat.query("training == 0"), "buyer_yes", 1, m, qnt=10)] for m in models]

gains_plot(eval_dat.query("training == 0"), "buyer_yes", 1, list(models[0:4]), qnt=10)

gains_plot(eval_dat.query("training == 0"), "buyer_yes", 1, list(models[4:]), qnt=10)

gains_plot(
    eval_dat.query("training == 0"), "buyer_yes", 1, ["y_tree", "y_tree_cv"], qnt=10
)

gains_plot(
    eval_dat.query("training == 0"),
    "buyer_yes",
    1,
    ["y_rf", "y_rf_cv", "y_xgb", "y_xgb_cv"],
    qnt=10,
)

performance_df = pd.DataFrame(
    {
        "models": models,
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0,
        "contact": 0,
        "profit": 0,
        "ROME": 0,
        "AUC": 0,
    }
)
performance_df = performance_df.set_index("models")

for m in models:
    performance_df.loc[m, 0:5] = confusion(
        eval_dat.query("training == 0"), "buyer_yes", 1, m, cost=0.5, margin=6
    )
    performance_df.loc[m, "profit"] = profit_max(
        eval_dat.query("training == 0"), "buyer_yes", 1, m, cost=0.5, margin=6
    )
    performance_df.loc[m, "ROME"] = ROME_max(
        eval_dat.query("training == 0"), "buyer_yes", 1, m, cost=0.5, margin=6
    )
    fpr, tpr, thresholds = metrics.roc_curve(
        y_test.values, eval_dat.query("training == 0")[m]
    )
    performance_df.loc[m, "AUC"] = metrics.auc(fpr, tpr)

performance_df.sort_values(by="profit", ascending=False)
