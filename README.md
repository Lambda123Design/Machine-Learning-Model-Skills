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

**7. k Nearest Neighbours - Classifier**

By default n_neighbours=5

### Most Important Parameters (For both Regressor and Classifier) here are Weights, Algorithm, p - By Default 2 - Eucledian Distance, if we set to 1 then Manhattan Distance - We can select using Hyperparameter tuning using key-value pairs (Refer Documentation for These)

**Selecting the Type of kNN Method depends on Problem Statement**

**Codes are just simple, refer the Notebook**
