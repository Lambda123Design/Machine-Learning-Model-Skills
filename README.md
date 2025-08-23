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
                   
         
