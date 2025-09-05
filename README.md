# Machine-Learning-Model-Skills

##### Very Important: Decision Tree doesn't require any kind of Standardization or Normalization. We can directly apply the Algorithm; Ensemble Techniques no need that

#### For Linear Regression use Standardization, as the Gradient Descent can converge faster

#### For Distance based models, use Standardization

#### Always check for y.value_counts if it is an Imbalanced Dataset; Ensemble Models such as Random Forest, XGBoost,etc.. perform well in Imbalanced Datasets

#### Random Forest Works better than AdaBoost, because AdaBoost works on the concept of Stumps

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

2. **AdaBoost**

3. **Gradient Boosting:** Fk​(x)=F0​(x)+LR⋅PR1​(x)+LR⋅PR2​(x)+⋯+LR⋅PRk​(x)


**PCA:**

First principal component will capture the more variance compared to second; And in the same way it will decrease as we move from third, fourth, etc..

If we didn't mention n_components in "pca=PCA(n_components=2)", it will find that many components to those many in the features;If we mention it will reduce, say 2 here;

Once we reduced, we can also visualize using matplotlibe

**4. K-Means Clustering:**

**Creating our own dataset** - X,y=make_blobs(n_samples=1000,centers=3,n_features=2)

### If we create 5,10 features, we can also reduce to 2,3 using PCA

**We won't use y here, as we are doing unsupervised learning; Clustering - It will cluster**

**In real world, we have to find how many clusters needed; Here we will verify if we get 3**

### We will use Silhoutte scoring to check if whatever cluster we got is valid or not

plt.scatter(X[:,0],X[:,1],c=y) - Visualizing 3 clusters, with y as color type

**Splitted into Train and test and doing Standard Scaling** - X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#### Finding Cluster using Elblow Method:

wcss - Within cluster sum of squares

wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(X_train_scaled)
    wcss.append(kmeans.inertia_)

**wcss value increases as k value increases**

#### Plotting Elbow curve with help of WCSS to find k value:

plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.xlabel("Number of Clustrers")
plt.ylabel("WCSS")
plt.show()

#### Like our Elbow, we need to find the knuckel region, where the value changes abruptly and becomes zero

#### In our image, we found out that at k=3, it comes abruptly to 0 (We gave centre=3 and we also validated here)

**Creating k-means clustering using 3 Clusters**

kmeans=KMeans(n_clusters=3,init="k-means++")

kmeans.fit_predict(X_train_scaled)

y_pred=kmeans.predict(X_test_scaled)

#### Validating k value can be done using: (Because in Real world finding the optimal k-value might be challenging:

#### 1. kneelocator

#### 2. Silhoutee scoring

**(i) kneelocator** - !pip install kneed

from kneed import KneeLocator

kl=KneeLocator(range(1,11),wcss,curve="convex",direction="decreasing") **In our case, it is a Convex Curve and value is decreasing, so we gave it**

kl.elbow

**We got output as 3, which is right and we validated**

**(ii) Silhoutee scoring**

from sklearn.metrics import silhouette_score

silhouette_coefficients=[]
for k in range(2,11):
    kmeans=KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(X_train_scaled)
    score=silhouette_score(X_train_scaled,kmeans.labels_)
    silhouette_coefficients.append(score)

silhouette_coefficients

## plotting silhouette score
plt.plot(range(2,11),silhouette_coefficients)
plt.xticks(range(2,11))
plt.xlabel("Number of Cluters")
plt.ylabel("Silhoutte Coeffecient")
plt.show()

**Once we plotted it, we were able to see that Score at 3, increased abruptly and we confirmed, as Silhouette Score is highest**

**5. Hierarchical Clustering using Agglomerative Clustering:**

**(i) Using Iris Dataset without Target column (For Clustering)**

iris_data=pd.DataFrame(iris.data)

iris_data.columns=iris.feature_names

X_scaled=scaler.fit_transform(iris_data)

**(ii) Performing PCA**

pca=PCA(n_components=2)

pca_scaled=pca.fit_transform(X_scaled)

plt.scatter(pca_scaled[:,0],pca_scaled[:,1],c=iris.target)

**(iii) Constructing a Dendogram:**

To construct a dendogram

import scipy.cluster.hierarchy as sc

plt.figure(figsize=(20,7))

plt.title("Dendograms")

sc.dendrogram(sc.linkage(pca_scaled,method='ward'))

plt.title('Dendogram')

plt.xlabel('Sample Index')

plt.ylabel('Eucledian Distance')

**Refer to Notes on how you find number of clusters in this case; Creating a Horizontal line on the longest vertical line, where no other horizontal line passes**

from sklearn.cluster import AgglomerativeClustering

cluster=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')

cluster.fit(pca_scaled)

### From the previous code of y_target from iris group, changing the color to cluster labels

plt.scatter(pca_scaled[:,0],pca_scaled[:,1],c=cluster.labels_)

####  In our dataset we have 3 categories; But what this clustering is saying that you don't need three, you can do with 2 (Got value of 2 from Dendogram Image) itself

**6. Naive Bayes Machine Learning Algorithm:**

#### **In Bernoulli Naive Bayes, we will have features as Yes/No and convert as 1,0; This is known as "Sparse Matrix" (Maximum Number of 1's and 0's available in the dataset; This we will use in NLP too**

#### In NLP, we take text data and convert it to Numerical data using techniques; Output will be in 1's and 0's

#### Whenever we have Sparse Matrix, we can use Bernoulli or Multinomial Naive Bayes; For NLP, we can use Multinomial Naive Bayes or Bernoulli Naive Bayes

**Loading Iris Dataset:** X,y=load_iris(return_X_y=True) [Multi-Class Classification]

### We have only Numerical/Continuous values; So we will use Gaussian Naive Bayes

gnb=GaussianNB(); gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)

print(confusion_matrix(y_pred,y_test)); print(accuracy_score(y_pred,y_test)); print(classification_report(y_pred,y_test))

**7. k Nearest Neighbours - Classifier and Regressor**

By default n_neighbours=5

### Most Important Parameters (For both Regressor and Classifier) here are Weights, Algorithm, p - By Default 2 - Eucledian Distance, if we set to 1 then Manhattan Distance - We can select using Hyperparameter tuning using key-value pairs (Refer Documentation for These)

**Selecting the Type of kNN Method depends on Problem Statement**

**Codes are just simple, refer the Notebook**

**8. DBSCAN Clustering:**

**Shows Dataset in Moon Format** - from sklearn.datasets import make_moons

X,y=make_moons(n_samples=250,noise=0.05) **Noise is like Outliers**

plt.scatter(X[:,0],X[:,1]) - **Scatter Plot for Dataset**

### In DBSCAN we have "Epsilon" and using Epsilon, we will be able to get Clusters

X_scaled=scaler.fit_transform(X)

**Using DBSCAN with some epsilon values** - dbcan=DBSCAN(eps=0.3)

dbcan.fit(X_scaled)

dbcan.labels_ **Based on this datapoints it was able to categorize as two categories, 0 and 1**

**Plotting in a Scatter Plot** - plt.scatter(X[:,0],X[:,1],c=dbcan.labels_)

**We can later use Silhoutte Scoring to find optimum k-value**

**9. Decision Tree**

treeclassifier=DecisionTreeClassifier()

##### Very Important: Decision Tree doesn't require any kind of Standardization or Normalization. We can directly apply the algorithm

treeclassifier.fit(X_train,y_train)

### Visualizing the Decision Tree

from sklearn import tree

plt.figure(figsize=(15,10))

tree.plot_tree(treeclassifier,filled=True)

### If Gini=0, then it becomes a Leaf Node; Becomes a Pure Split; Anything Gini Near to 0.5, then it is Impure Split

### We may be getting Overfitting here; We can avoid Overfitting using: Pre-Prunning or Post-Prunning

### In Post-Pruning, if one category is much higher others, we can conclude that it is final [0,33,1], we can conclude it is the second which is the final category (Small Number won't play a greater role there)

### In Pre-Prunning, we will play with, Max-Depth, Gini or Entropy, Max Features; This is just like Hyper-Parameter Tuning

#### Post-Pruning always good for Smaller Datasets; Because if large dataset, we need to construct entire dataset, it takes more time; So do post-pruning for small datasets

#### Use Pre-Pruning for Large Datasets

#### Pre-Prunning and HyperParameter Tuning:

param={
    'criterion':['gini','entropy', 'log_loss'],
    'splitter':['best','random'],
    'max_depth':[1,2,3,4,5],
    'max_features':['auto','sqrt','log2']
}

from sklearn.model_selection import GridSearchCV

treemodel=DecisionTreeClassifier()

grid=GridSearchCV(treeclassifier,param_grid=param,cv=5,scoring='accuracy')

grid.best_params_

grid.best_score_

**Decision Tree Regression:**

df_diabetes=pd.DataFrame(dataset.data,columns=['age',
  'sex',
  'bmi',
  'bp',
  's1',
  's2',
  's3',
  's4',
  's5',
  's6'])
df_diabetes.head()

**HyperParameter Tuning** - param={
    'criterion':['squared_error','friedman_mse','absolute_error'],
    'splitter':['best','random'],
    'max_depth':[1,2,3,4,5,10,15,20,25],
    'max_features':['auto','sqrt','log2']
}

grid=GridSearchCV(regressor,param_grid=param,cv=5,scoring='neg_mean_squared_error')

grid.best_params_

**Predictions** - y_pred=grid.predict(X_test)

print(r2_score(y_test,y_pred)); print(mean_absolute_error(y_test,y_pred)); print(mean_squared_error(y_test,y_pred))

**10. Random Forest:**

By defauly n_estimators=100

#### Lot of Model codes are there; Learn from it; Notebook name "Holiday Package Prediction Project - Random Forest Classification"

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])

**Seeing ROC AUC Curve**

## Plot ROC AUC Curve
from sklearn.metrics import roc_auc_score,roc_curve
plt.figure()

# Add the models to the list that you want to view on the ROC plot
auc_models = [
{
    'label': 'Random Forest Classifier',
    'model': RandomForestClassifier(n_estimators=1000,min_samples_split=2,
                                          max_features=7,max_depth=None),
    'auc':  0.8325
},
    
]
# create loop through all model
for algo in auc_models:
    model = algo['model'] # select the model
    model.fit(X_train, y_train) # train the model
# Compute False postive rate, and True positive rate
   ; fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
# Calculate Area under the curve to display on the plot
   ; plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (algo['label'], algo['auc']))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("auc.png")
plt.show() 

**11. AdaBoost**

from sklearn.ensemble import AdaBoostClassifier

adaboost_param={
    "n_estimators":[50,60,70,80,90],
    "algorithm":['SAMME','SAMME.R']
}

**'SAMME','SAMME.R'** - For Boosting

HyperParameter Tuning:

Models list for Hyperparameter tuning
randomcv_models = [
                   ("RF", RandomForestClassifier(), rf_params),
    ("AB", AdaBoostClassifier(), adaboost_param)


#### Once you got the Best Models, Copy Paste here

models={
    "Random Forest":RandomForestClassifier(n_estimators=1000,min_samples_split=2,
                                          max_features=7,max_depth=None),
    "Adaboost":AdaBoostClassifier(n_estimators=80, algorithm='SAMME')
}           ]

#### We can also use a Pipeline, but this way of doing is better

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Adaboost Regressor":AdaBoostRegressor()

}

**AdaBoost Regression:**

ada_params={
    "n_estimators":[50,60,70,80],
    "loss":['linear','square','exponential']
}

**Model List for HyperParameter Tuning:**

randomcv_models = [('KNN', KNeighborsRegressor(), knn_params),
                   ("RF", RandomForestRegressor(), rf_params),
                   ("Adaboost",AdaBoostRegressor(),ada_params)                   ]


**Perform HyperParameter Tuning**

from sklearn.model_selection import RandomizedSearchCV

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])

**AdaBoost got only 60% in AUC ROC Curve, because of this stump; Random Forest performed well when compared to this**

**12. Gradient Boosting:**

1. Defining Hyperparameters for Hyperparameter Training
rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}
gradient_params={"loss": ['log_loss','deviance','exponential'],
             "criterion": ['friedman_mse','squared_error','mse'],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500],
              "max_depth": [5, 8, 15, None, 10]
                }

**2. Seeing the Gradient Boost Parameters** - gradient_params

3. Models for Hyperparameter Tuning:

randomcv_models = [
                   ("RF", RandomForestClassifier(), rf_params),
    ("GradientBoost", GradientBoostingClassifier(), gradient_params)
                   ]

**4. Seeing the best params for the Models**

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])

5. Updating the parameters to train with the best parameters:

models={
    "Random Forest":RandomForestClassifier(n_estimators=1000,min_samples_split=2,
                                          max_features=7,max_depth=None),
    "GradientBoostclassifier":GradientBoostingClassifier(n_estimators=500,
                                                        min_samples_split=20,
                                                        max_depth=15,
                                                        loss='exponential',
                                                        criterion='mse')
}

**Then we viewed AUC ROC Curve too and Gradient Boosting was able to cover around 90%**

**Gradient Boosting Regression:**

1. Giving Hyperparameters to tune:

rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}

gradient_params={"loss": ['squared_error','huber','absolute_error'],
             "criterion": ['friedman_mse','squared_error','mse'],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500],
              "max_depth": [5, 8, 15, None, 10],
            }

2. Model List for Hyperparameter Tuning:

randomcv_models = [
                   ("RF", RandomForestRegressor(), rf_params),
                   ("GradientBoost",GradientBoostingRegressor(),gradient_params)
                   ]

3. Performing HyperParameter Tuning:

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])

4. Retraining with Best Parameters:

models = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, min_samples_split=2, max_features='auto', max_depth=None, 
                                                     n_jobs=-1),
     "GradientBoost Regressor":GradientBoostingRegressor(n_estimators= 200,
                                                         min_samples_split=8, max_depth=10, loss= 'huber', criterion='mse')
    
}

**13. XGBoost**

#### In all these Models learnings, Krishg= first trained Normally like this; Then only defined Parameters and did HyperParameter Tuning

models={
    "Logisitic Regression":LogisticRegression(),
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "Gradient Boost":GradientBoostingClassifier(),
    "Adaboost":AdaBoostClassifier(),
    "Xgboost":XGBClassifier()
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test
    # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score
    model_train_precision = precision_score(y_train, y_train_pred) # Calculate Precision
    model_train_recall = recall_score(y_train, y_train_pred) # Calculate Recall
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)
    # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
    model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
    model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recall
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred) #Calculate Roc
    print(list(models.keys())[i]
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))
    print('----------------------------------')
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score)
    print('='*35)
    print('\n')

**1. Defining Parameters**

## Hyperparameter Training
rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20, 30],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]}

2. Seeing the Parameters - xgboost_params

3. Model List for HyperParameter Tuning:

randomcv_models = [
                   ("RF", RandomForestClassifier(), rf_params),
    ("Xgboost", XGBClassifier(), xgboost_params)
                   ]

**4. Performing RandomSearchCV with HyperParameter Tuning:**

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])

**5. Training with Best HyperParameters:**

models={
    "Random Forest":RandomForestClassifier(n_estimators=1000,min_samples_split=2,
                                          max_features=7,max_depth=None),
    "Xgboost":XGBClassifier(n_estimators=200,max_depth=12,learning_rate=0.1,
                           colsample_bytree=1)
}

**6. Plotting ROC AUC Curver:**

auc_models = [
{
    'label': 'Xgboost',
    'model':XGBClassifier(n_estimators=200,max_depth=12,learning_rate=0.1,
                           colsample_bytree=1),
    'auc':  0.8882
}

**XG Boost covered 89%**

**Similar Codes for XGBoost Regressor Too, Refer Notebook**
                   

**14. Support Vector Machines:**

**A) Support Vector Machines - Classification:**

In this video we are going to implement a support vector classifier with the help of sklearn. And for this we'll create our own data set. We'll see some visualization. And uh probably we'll also be discussing about SVM kernel as we go ahead with respect to all these videos okay. So first of all I'm just going to import some important libraries. First of all let's go ahead and import "import pandas as pd" "import numpy as np" okay. Uh, import Seaborn. As a CNS. And finally you can import matplotlib, which I am also going to use "dot pyplot as plt". And I can also import seaborn write "seaborn as sns". So these are some of the basic library that I'm going to use it for creating our data set. The first step is that we will try to create some data set, but in the upcoming videos we'll be seeing a full fledged projects also with respect to support vector classifier.

So now let's create let's create a synthetic or data points okay data points. And here we'll try to solve both classification problem and for multiclass also we can basically use that over concept in SVM SVM okay. So let's create some uh data points uh synthetic data points. And in order to create a synthetic data points, all you have to basically do is that "import from sklearn.datasets import make_classification". So I hope everybody knows about this. Miss classification. We have already used this. This "make_classification" will be giving me x and y axis. And probably if I use "make_classification" here the "number of samples will be 1000". And the "number of features that I'm probably going to consider, let's say the number of features, I can basically say two, just to see that whether my visualization will look good or not". And here the "number of classes also I'll be trying to take it as two". And there is also like "number of cluster per class". I will go ahead and assign this value as one okay. And, uh, any specific parameter. Let's see whether this will work or not.

So here it gives some warning. Uh, number of informatic redundant repeated feature. Must be some. Okay, this is the same warning that we got earlier. So here I'm just going to make it as "number of redundant features = 0". Okay. So now this is my x coordinates with respect to my input features. And this is my y which is with respect to my output features. Now quickly let's, uh plot the scatter plot. So I'll be using "sns.scatterplot". And on this particular scatter plot you will be seeing, I will be using my x and y coordinate okay. So x and y coordinate and there is will be a parameter which is called as wi okay. And here I can definitely see. The thing is that in my x I have two axes right x one. Uh, this this is my first column. This can be my second column. Right. So we will give in our x and y axis with respect to that same column. Okay.

So probably I can do one thing. I can just write "df = pd.DataFrame(x)". I can basically convert this into data frame. And if I probably write "x[0]" here you will be able to see this is how we'll be getting. If I want to get the first column I can write off zero. So this will basically give me the first column. Right. So I'll go and paste it over here. And then I will go and paste over here and basically write one which will give me my y axis. And this we will basically be given by my y parameter. Now once I plot it here. Uh, let's see, "Seaborn has no attribute scatter plot". Okay then this will be "sns.scatterplot". And then we try to execute it. Now here you can definitely see that, uh, my data points are very clearly separable. And obviously I can actually create a best fit line with the help of SVC. And for this, the SVC that we will be using, it is called as "linear SVC". And let's see how the accuracy of our uh, data set will basically come with respect to this okay. So this is how my data set looks like.

Now in the next step, all I have to do is that go to sklearn and use SVC. So if I write a skeleton as VC that is SVM classifier. This is where the library is basically present "from sklearn.svm import SVC". So all I will do is that I'll say "from sklearn.svm import SVC". Right. So once I execute this, this will be my SVC. And then I am basically writing my SVC over here. By default there are many parameters which we can basically play with hyper parameter tuning like kernels, degree, gamma and all. We'll try to play with hyper parameter tuning, but right now by default, you will be able to see that we are having a kernel which is called as RBF kernel. But I as I know that. Right? Because since this is two dimensional points, I definitely know I can see that, okay, it is clearly separable. I can definitely use a "kernel='linear'". Okay. So when your data points is clearly separable, you can use the kernel as linear and start fitting the entire data set.

Then I will go and do "svc.fit(X_train, y_train)" okay. So once I do this. So okay x train and train is still not done. So let me do after this. Right. I will just try to do my train test split for. So I'll say "from sklearn.model_selection import train_test_split". And then, uh, what I'm actually going to do over here is that I'm just going to create my "X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)". Okay. And this will basically give me my train test split okay. So this has got executed. Now I will go and do this. I will go and fit with my Xtrain and Ytrain. And now let's go ahead and do my predictions. Okay. Now for this I will do the prediction as "y_pred = svc.predict(X_test)". Now obviously I know that this will give us a very good accuracy because we definitely have a good separation between these two points. Okay.

So uh, now what we are going to do over here is that, uh, "from sklearn.metrics import classification_report, confusion_matrix". I think this both will be more than sufficient. Then I can "print(classification_report(y_test, y_pred))". And then I can also "print(confusion_matrix(y_test, y_pred))". So once I execute it here you can see absolutely 100% accuracy. And obviously because my data points looks clearly separable. Right. So this is one uh, amazing thing that we have done. And with the help of linear, as we see, yes, we are able to get it. Probably if I apply logistic regression also I will be able to get a good accuracy. Okay. The reason is that my points are very clearly separable.

Now, uh, let's say if I don't give this specific parameter and "n_clusters_per_class", you know, what will basically happen is that you just see it. Okay. So let's say if I press shift tab and. Over here. "n_clusters_per_class" is two by default, right? Let me just give it as two. Now you will be seeing if I probably plot my data set. Now this has a lot of overlappings. Now if I probably use linear SVC, I will be getting a bad accuracy. You can see over here. So obviously my accuracy is decreased to 85%. Right. Why. Because this data points are not that linearly separable. But uh, usually uh, we try to find out a data set which is trying to get if it is linearly separable, we can definitely use this linear SVC.

Now there may be situation that we may get this kind of overlap data set also. And for this kind of data sets we can use different different kernels okay. The kernels like RBF kernel they are kernels like polynomial kernels. Sigmoid kernel. Let's say by default if there is RBF. And for the same data point I go ahead and use RBF kernel, right? So if I say "svc = SVC(kernel='rbf')" and then "svc.fit(X_train, y_train)" okay. Now I have also done the fit. Now I can do the prediction for doing the prediction. Let's say this is "y_pred1 = svc.predict(X_test)". And once I execute it, I'm probably copy and paste this entire thing and try to print it with respect to y pred one. Now you can see that the accuracy has improved. Right before I was getting somewhere around 85%. Now because of the RBF kernel. What may have happened internally, some transformation of this lines has happened. Uh, this points has happened and it is probably created a new dimension. Okay. And because of that, you will be able to see that you are able to get better accuracy, right?

Similarly, if you try with other kernels, even like polynomial, suppose let's say if I try with a polynomial over here, I'll copy everything over here and paste it. Let's see. This will be my prediction. Okay. And this will be my classification report. And the confusion matrix. So let's say if this is "SVC(kernel='poly')" okay. And with the help of polynomial also there are chances because of the mathematical equation or transformation that will basically happen. It may improve the accuracy okay. So here it is polynomial Y1Y2. This will be y_pred2. This will also be y_pred2. So once I probably plotted "poly" is not in the list. Okay, so I think it is "poly". Okay. You can probably press shift tab and you'll be able to see that. What are the possible options in the kernel. Right. So in kernel you have poly rbf linear sigmoid and all. So suppose if I say I'll be using "poly" I'm probably trying to predict it. Now I'm getting 82%. So definitely poly transformation will not work because it is giving bad accuracy and compared to the linear part also.

Similarly I can basically do it for sigmoid also, right? "SVC(kernel='sigmoid')". And this I will just make it as sigmoid. Because these are all different, different kernels, which we will basically be using. And each and every kernel applies a different kind of formula and transformation on your data set. Okay. So once I predict this "y_pred3 = svc.predict(X_test)" and let's see whether we get a better accuracy when compared to the previous one. We can also do that. So here you can see that with the help of sigmoid again bad accuracy is coming. So out of this if you tell me that which kernel I may be probably using, I'll be using RBF kernel for this kind of data set okay. Because it will do some kind of transformation, create a higher dimension based on the formula of RBF that is radial basis kernel. Okay.

So this is how the kernel basically happens now along with the kernel. Once you are able to decide, uh, the kernel that you are probably going to use, you know, you can also perform something called as hyperparameter tuning. Now with respect to hyperparameter tuning, what we are basically going to do is that we're going to play with different, different parameters. Okay. So I have probably made a list of parameter with respect to "param_grid". Okay. Now, as I know with respect to my particular problem statement, I'm going to use RBF because it has given better accuracy with respect to RBF uh, over here, right, with respect to this specific data set. But there are some more parameters in SVC that you will be able to see. And this parameters are nothing but C value, kernel value, degree value, gamma value, shrinking probability total. All these values are there and we can definitely play with this parameter. We can select this right kind of parameters. And it completely depends on the data set.

So here I'll just show you one example how we can perform "GridSearchCV". So once I execute this all I have to do is that go ahead and train my grid search CV. So here you'll be able to see I'll be using "GridSearchCV(SVC(), param_grid=param_grid, cv=5, verbose=3, refit=True)". And once I probably implement this I will save it in grid. Okay. There is also a parameter which is called as refit in this okay. So now if I probably write "grid.fit(X_train, y_train)" now you'll be seeing that it'll take some amount of time for training. Now this entire training has happened with respect to different different parameters. And now here you can see that I will be able to get the best parameter. So if I write probably "grid.best_params_" you'll be able to see these are the parameters that are got selected. "C=1000, gamma=1, kernel='rbf'". Now if I probably do the predictions here, you'll be able to see it. So this will be my predictions. And my predictions will be done with the help of grid. Right. So "y_pred4 = grid.predict(X_test)". And finally you'll be able to see what is the output result that I'm actually getting. So overall a good accuracy of 93%.

Now see this is the basic difference. What will happen if you try to select the right C value gamma value? I've just played with two parameters over here like that in SVC. There are so many different different parameters, right. So I hope you got an idea. With respect to Gridsearchcv, uh, and you have also understood how we can basically perform hyper parameter tuning. So I'm just make a cell over here and basically create hyper parameter tuning with SVC. Okay. Super important, super easy, uh, technique to basically see the accuracy. And now you can see that the accuracy is increased to 93% before it was somewhere around. If you if you probably see it, it was somewhere around uh, 88%. Right, with the help of RBF. But still we use some different different parameters like see gamma kernel. So here you should not stop over here. Still, you should basically try out with different different parameters right now like class weight. If you have an imbalanced data set, definitely use class weight. You can also use gamma right gamma with respect to gamma different different values you can assign with respect to Coef or you can assign different different values. And you can basically play.

Now this uh, you know, whenever you probably calculate with respect to SVC, SVC also gives you different uh, coefficients and intercept. So suppose if I probably see for sigmoid if I write "sigmoid.intercept_" here you'll be able to see this intercept comma if I probably write "sigmoid.coef_" so here also you'll be able to see the coef value. Coef is only used when using a linear kernel. Okay, perfect. So for this it will not be there. But if you probably see over here for a linear kernel, which we have actually seen in this SVC, here you will be also able to see the coefficient. Obviously it is because for a curvy line you will not get coefficient. For a straight line only you will be getting okay. So "svc.coef_" and once I execute this here, you can see the coefficients that you are able to get right. Because in my X train I have two features. Right. So yes, this was just a basic implementation of SVC. I hope you are able to understand this. Um, now what we'll do is that we'll try to create more complicated data set. And let's see by if I do the transformation manually. Right. Like polynomial kernel transformation manually, you know, uh, what what things will happen. And are we able to change the data or clearly separate the data. You know, uh, once we find out a linear separable data itself by using a plane so that all things will have a look onto my next video. So yes, I will see you all in the next video. Thank you. With SVM kernels.

**B) SVM Kernels:**

In this video we are going to see the power of SVM kernels, right? In our previous video we have seen some SVM implementation. And here also I showed you that when we have this kind of data set right, and when we use RBF, you know, we basically get a very good accuracy, right? Uh, and along with that, if we perform hyperparameter tuning, still more better accuracy. But today in this video I'm going to focus more on SVM kernels. Now let's say if we are using a linear SVC that is support vector classifier, if my data points are linearly separable, then only it will give us good accuracy. What if I have data points which looks like this. Now here you can see my data points are completely overlapped, right? I cannot create a linear best fit line which looks like this. Right? I cannot do that. I cannot create a linear best fit line, right? I cannot create something like this line. Okay. Sorry for the line. Like, uh, I cannot create like this, right? Obviously it will give me an error because 50 percentage of the points are correctly classified, 50% will be incorrectly classified. So this is a big blunder. So can we use SVM kernel, or probably increase the number of dimension with respect to this particular features, and then probably see how we can, uh, easily classify this again by using linear SVC.

Okay. So first of all I have just created this data points. So here you can see these are my data points. Uh here I've created my x axis y axis I've stacked all my values with respect to. Now see this if I probably execute it. And if I see this my x axis. Now here, this will be what will be my x value. It says that it is going to. It is basically present between minus five to plus five. And the number of points are 100. Right. So these are all the 100 points right. Similarly if you see why I've just done the "np.sqrt" over here. And finally you'll be able to see this okay. Now if I go over here this is my other coordinates. Let's say uh this is this this x and y are my outer coordinates over here. Right. The inner coordinates I have actually plotted over here between five -5 to 500 and "np.sqrt" with five square and x one square. Okay. So once we do this plotting then y comma x and y1 x1 I will be getting this kind of data points. Now this is perfectly fine. Now what I will do is that with respect to this outer data points I will label it as zero. Okay. So here you can see that I'm doing the vertical stacking of y comma x. Vertical stacking means one after the other okay of y comma x okay. Probably. Let me just show you what will happen if I probably do this and probably transpose this particular value. So if I probably make a cell above here, you'll be able to see that I have vertically stacked this and I have made a transpose transpose like this. Right. So how we do a transpose in the matrix. So here you can see this is my first feature with two features. I have these values. Then this is my second data point. Third data point fourth data point like this. Right. So that and I have specifically taken two data points over here because I can visually see this. Okay. Now here I've given my column names X1 X2 my df1 of y. This is my output feature for all this outer data points. I'm saying that all my data points is basically given as zero the output feature. And uh, with respect to y1 x1 when I'm actually, uh, putting it in data frame two, this is uh, the output will basically be one. And then I'm appending df1 with df2. That's simple right. And this is my final df you know X1 X2 I have. Then after that this will get appended y1 x1. Along with this. And this will be my y value which will have zero and one. Right. So if I probably see "df.tail" you will be able to see. Finally I'll be getting all the ones values like this. Okay, now after this, as usual, I will try to divide my features into independent and dependent features. So x1 and x2 will be all my independent features. Y will basically be my dependent feature. So here it is. This is my y value. Then I do "train_test_split". This is super super simple. So I'm going to go a little bit quickly. So here is my y train.

Now there are different different types of kernels that you can apply. See one is something called as polynomial kernel okay. Polynomial kernel SVM formula. If I probably go and search for this and search in Wikipedia right, you'll be able to see that I will be getting a different definition which is looking like this. Right. So this is the transformation that I can apply on, on the on the feature one and feature two. So basically on two points not feature one and feature two, but two points x and y. Right. And if I probably consider x they will be feature one and feature two and y also they will be feature one and feature two two separate points. If you do the transpose plus C to the dimension of T, uh, sorry to the dimension of T, D is nothing but the polynomial degree. So in this particular case here I'm specifically applying polynomial kernel. And this is the transformation that I am going to apply for any points okay. And we have already discussed in theory based on this transformation we are going to get what all key important points that is x1 square x2 square and x1 multiplied by x2. So this three key points I will try to find it out from this particular equation. And I'll try to plot it over here when I try to plot this by considering x1 square, x2 square and df of x1 and x2, I'll be able to see that my points will be linearly separable. Okay, so from this equation, the three key important components that we will be getting is x1 square, x2 square and x1 multiplied by x2. Since all my values are in data frame, so I'm just going to create new features like x1 square, x2 square, and x1 comma multiplied by x2. So once I execute this these are how my points will look like. Now see uh let's take all my independent features from here. Independent features. These all are my independent features. Right. So I'll, uh, then update my independent feature. And my Y will basically have my output feature. Now again, let me go ahead and do "train_test_split". Now see one magical thing will happen now okay. So once I do the "train_test_split". And now if I try to plot it now see what I am doing while plotting, I will take x1. I will take x2 and then I'll take x1 multiplied by x2. Okay. And probably this is a scatter 3D plot. So in plotly.express you have something called a "scatter_3d" plot. You have to give three coordinates your entire data frame, your x coordinate y and your z. So x I am taking it as x1, x2 and x1 multiplied by x2. So if I probably plot this and uh you'll be able to see this. So once I plot this see this is the same thing I plotted, right. This is the same thing I plotted X1 X2 X1 comma x2. Right. So this are the three features I plotted X1 X2 and x1 multiplied by x2. So if I plot it here, you'll be able to see that I will be getting a data points which will be separated like this. Obviously, from this particular data point you can see that it is not clearly linearly separable. I cannot just create a simple SVM line, right? I cannot create a SBC line, a marginal plane onto this. Right. Because again, half of the points are here, half of the points are here. But now see what happens if I try to plot this same 3D plot by using x1 square, x2 square and x1 multiplied by x2. Now see what will happen. So here I am taken x1 square, x2 square and x1 multiplied by x2. Once I execute it. Now you see the magic guys. My data points has now got transformed into a higher dimension of three. Now in this particular points is my is my other points and this are my other points. Now here you can see easily. Initially my data points was something like this right. How was my data points. My data points were like this. Now after applying this polynomial transformation, what did we do in polynomial transformation? We just created this additional feature x1 square, x2 square and x1 multiplied by x2. And I took this three features. I plotted it in a 3D plot. Now here you can see that I am able to get two groups of plot very much clearly with respect to different different data points. Now all I have to do is that create a hyper plane within this. And how do I create a hyper plane? Because if I'm able to get this kind of transformation, all then I can do is that use an SVC kernel with linear and just try to do the prediction. Now you see what will be the accuracy. Okay.

So let me import seaborn. So here I'll just go into "import seaborn". From "sklearn.svm import SVC". Now, if I probably try to execute okay, accuracy score is also not there. Let's say "from sklearn.metrics import accuracy_score". Now if I try to execute I'm getting one as my accuracy score right. But here you can see that with respect to polynomial kernel right. What did I do I manually created this I manually created this features. And then I applied a linear kernel uh linear kernel over here with respect to SVC. But instead of that if I could have just written like this and over here, if I would have written my kernel as polynomial and executed this, here, you'll be able to see, okay, the polynomial, it is not polynomial, it is poly. Okay. If I had just directly written "kernel='poly'" and executed, I could have got the same accuracy. So poly internally what it does polynomial kernel. It internally creates this feature x1 square, x2 square and x1 multiplied by x2. You don't have to do it manually. If you do it manually, then you can definitely apply linear kernel, right because it is clearly separable.

Now similarly, you can also solve this problem with the help of RBF kernel that is radial basis kernel. So if I probably search for RBF kernel I'll see this RBF kernel. Right. This is radial basis function. Right. And in this kernel we are okay. Let me just open Wikipedia. Okay. Now over here the transformation formula that gets applied to create more dimensions. Is this formula right X and x dash are different different coordinates. And the sigma is basically initialized with some values. Okay. If we apply this exponential formula in creating new transformation feature, then automatically RBF will also be applicable in solving this problem. Now see, instead of writing poly, all I have to do is that write "rbf" and then try to do the prediction here. Also, you can see that I'm getting one right. So with the help of RBF kernel, how does RBF kernel looks like? What how how RBF will basically make the changes. So here if I probably change, uh see the images. Right. So here if I probably go and search for the images, uh, let me see one example with respect to RBF kernel, okay. How the RBF kernel will look like. So my RBF kernel will basically create this kind of transformation. That basically means all the points that are present below, right? It will become like this. It will have this kind of curve. So all the elements that are present in the central right, this will become like this. It will go up. Okay. Probably I can show you some more things, some more examples. Yeah. This is the right thing. Now see guys here you can see one curve right. This curve okay. Yeah. This is the perfect curve. So this curve is basically created by this RBF kernel right. So here you can see the central points is going up. Now all you have to do is that probably create a plane in between like this probably to separate the points because all your major classes will be going up and this will be your other set of classes. Right. So this is what RBF kernel does by using this simple transformation technique. All the points from here will go in the higher dimension. Right. So this is what RBF kernel will do.

And similarly over here you can see that I have this kind of points right? I have this kind of points over here. Now what will RBF kernel do in this particular case. It is very simple. RBF kernel will make sure that all these points, once it is converted into 3D dimension, it should look something like this. Let's say these are my points. Okay, these are my below orange points. So let's say these are my blue points. And top of it I may create these points which may look like this. Okay. And then probably I can create my best fit line. Okay, so this is what it basically does with the help of RBF. So we can solve this kind of problem statement with both linear or polynomial and RBF kernels. Right. And similarly there is also another kernel which is called as sigmoid. You can also try it out with sigmoid. So let's see again this particular problem statement with sigmoid also. So here I will just go and copy and paste it. And probably instead of RBF I'll write "sigmoid". Now let's see I don't know whether we'll get or not but sigmoid also we are able to get as 1.0. That is 100% accuracy with respect to test data. So. Whenever you get bad accuracy with respect to SVM, try to use SVM kernels. Try to play with this kind of hyperparameter tuning with respect to linear poly or RBF or sigmoid. And then you try to select what parameter is best for it. So that is what I had actually shown you in my previous example. So this was the power behind the SVM kernel. I hope you liked this particular video. Please make sure that you understand things. Understand what is the transformation technique that is used? Some mathematical formula to create some more additional functional features internally. I showed you an example of polynomial. Similarly, you can uh assume with respect to radial basis function that is RBF and similarly with respect to sigmoid. Okay. So yes, I will see you all in the next video. Thank you.

**C) Support Vector Machines - Regression**

So in this video we are going to implement the support vector regression machine learning algorithm. We will take up a use case and try to solve it. We have already seen the example of support vector classifier, and we have also seen SVM kernels. Now let’s go ahead and try to explore support vector regression. For this, first of all, we will start with the data set, and the data set that we are going to take is about the Tips data set. Again this data set is available in a library called seaborn, so if you say "import seaborn as sns" and then "sns.load_dataset('tips')" this is the data set that we will be working with. You can see this data set has features like total bill, tip, sex, smoker, day, time and size. Based on this, let’s consider that in this particular data set the total bill will be my dependent feature and the remaining features will be my independent features. With respect to all these features, we will try to predict the total bill, and for this we will be using support vector regression.

Now if I make the dataset as a DataFrame and write "df.head()" you will be able to see the first few records. In this data set, our problem statement is that we need to predict the total bill, so total bill becomes the dependent feature, and all the others are independent features. One thing to notice is that features like sex, smoker, day and time are categorical features. If I write "df.info()" you will be able to see that sex, smoker, day and time are categorical. That means we have fixed categories. For example, if I write "df['sex'].value_counts()" you will see the number of males are 157 and the number of females are 87. Similarly, "df['smoker'].value_counts()" shows 93 are smokers and 151 are non-smokers. If you check "df['day'].value_counts()" you will see Saturday 87, Sunday 76, Thursday 62, Friday 19. So we have four categories there. And for "df['time'].value_counts()" you will see it is either lunch or dinner.

Now sex, smoker, and time are binary categories, meaning they have only two categories like male or female, yes or no, dinner or lunch. Day is different because it has four categories. For categorical features we need feature encoding, and we have two approaches: label encoding and one hot encoding. Label encoding is used when we have just two categories, replacing them with zeros and ones. For example, sex male/female, smoker yes/no, time lunch/dinner can all be encoded with label encoding. For day which has four categories, we apply one hot encoding, which creates separate columns for each category with 0 or 1 values.

Before doing feature encoding, we need to split the dataset into independent and dependent features. For that we write "X = df[['tip','sex','smoker','day','time','size']]" and "y = df['total_bill']". Then we do a train test split by writing "from sklearn.model_selection import train_test_split" and then "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)". Now "X_train.head()" shows our training features.

Now let’s go ahead with feature encoding. It is important to do the split first and then encoding, otherwise there can be data leakage, meaning the model gets information from test data. For label encoding we import "from sklearn.preprocessing import LabelEncoder". Then we create three encoders "le1 = LabelEncoder()", "le2 = LabelEncoder()", "le3 = LabelEncoder()". Now we apply them: "X_train['sex'] = le1.fit_transform(X_train['sex'])", "X_train['smoker'] = le2.fit_transform(X_train['smoker'])", and "X_train['time'] = le3.fit_transform(X_train['time'])". Now "X_train.head()" shows sex, smoker and time encoded as zeros and ones.

For the test data we must apply the same transformation but only using transform, not fit, to avoid leakage. So "X_test['sex'] = le1.transform(X_test['sex'])", "X_test['smoker'] = le2.transform(X_test['smoker'])", and "X_test['time'] = le3.transform(X_test['time'])". Now "X_test.head()" also shows the encoded values.

Next we need to encode the day feature using one hot encoding. For this we import "from sklearn.compose import ColumnTransformer" and "from sklearn.preprocessing import OneHotEncoder". We define a column transformer "ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')" where [2] is the index of the day column in our current X. Then we transform the training data "X_train = ct.fit_transform(X_train)" and transform the test data "X_test = ct.transform(X_test)". Now day is replaced by one hot encoded columns, and the remaining features are passed through.

At this point, encoding is complete and we are ready to apply support vector regression. For this we write "from sklearn.svm import SVR", then "svr = SVR()", and then "svr.fit(X_train, y_train)". After fitting, we can predict with "y_pred = svr.predict(X_test)". Now to evaluate, we import "from sklearn.metrics import r2_score, mean_absolute_error" and then print "r2_score(y_test, y_pred)" and "mean_absolute_error(y_test, y_pred)". You will get the R2 value and mean absolute error.

Now we can also do hyperparameter tuning to improve results. For that we use grid search. First we import "from sklearn.model_selection import GridSearchCV". Then we define "param_grid = {'kernel': ['linear','poly','rbf','sigmoid'], 'C': [0.1,1,10], 'gamma':['scale','auto']}". Then "grid = GridSearchCV(SVR(), param_grid, refit=True)". Fit it with "grid.fit(X_train, y_train)". Now "grid.best_params_" will show the best hyperparameters. Then we can predict with "grid_predictions = grid.predict(X_test)". Finally evaluate again with "r2_score(y_test, grid_predictions)" and "mean_absolute_error(y_test, grid_predictions)". You will notice R2 score improves and mean absolute error reduces compared to before.

So in this video we have implemented support vector regression. Along with that you saw how we handled categorical variables with label encoding and one hot encoding, which is very important. We also did hyperparameter tuning with GridSearchCV to improve model performance. In upcoming videos we will also see how to handle missing values and other preprocessing steps. So yes, I will see you all in the next video. Thank you.
