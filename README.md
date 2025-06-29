# Machine-Learning-Model-Skills

****Supervised Learning:****
**Linear Regression:**
1. **Polynomial Regression** - degree, include_bias (from sklearn.preprocessing import PolynomialFeatures) (from sklearn.pipeline import Pipeline)

2. **Ridge, Lasso and ElasticNet**
   (from sklearn.linear_model import Ridge, from sklearn.linear_model import Lasso, from sklearn.linear_model import ElasticNet)
   Cross Validations:

   **LassoCV** - cv, alpha, mse_path
   **RidgeCV** - cv, getparams
   **ElasticCV** - l1norm

3. **Logistic Regression:**
  Sample Dataset - from sklearn.datasets import make_classification - n_samples,n_features, n_classes
  penalty_term, class_weight (from sklearn.linearmodel import Logistic Regression)

  Multi-Class Classification: logistic=LogisticRegression(multi_class='ovr')
  X,y=make_classification(n_samples=10000,n_features=2,n_clusters_per_class=1,
                   n_redundant=0,weights=[0.99],random_state=10)

  class_weight=[{0:w,1:y} for w in [1,10,50,100] for y in [1,10,50,100]]


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
penalty=['l1', 'l2', 'elasticnet']
c_values=[100,10,1.0,0.1,0.01]
solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
class_weight=[{0:w,1:y} for w in [1,10,50,100] for y in [1,10,50,100]]

params=dict(penalty=penalty,C=c_values,solver=solver,class_weight=class_weight)

**AUC ROC Curve:**
## calculate ROC Curves
dummy_fpr, dummy_tpr, _ = roc_curve(y_test, dummy_model_prob)
model_fpr, model_tpr, thresholds = roc_curve(y_test, model_prob)


 

1. **Random Forst** - n_estimators, max_features,criterion (from sklearn.ensemble import RandomForestClassifier)


**HyperParameter Tuning:**
**Grid Search CV:**
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
