# Rain Prediction in Australia

## About the Data Set
The data set used for this project originates from the Australian Government's Bureau of Meteorology and can be accessed [here](http://www.bom.gov.au/climate/dwo/). The dataset used in this project includes additional columns such as 'RainToday,' and the target variable is 'RainTomorrow,' which was obtained from [Rattle](https://bitbucket.org/kayontoga/rattle/src/master/data/weatherAUS.RData).

## Import the Required Libraries
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
```

## Importing the Dataset
```python
df = pd.read_csv("Weather_Data.csv")
df.head()
```

## Data Preprocessing
### One-Hot Encoding
First, we perform one-hot encoding to convert categorical variables to binary variables.

```python
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
```

Next, we replace the values in the 'RainTomorrow' column, changing them from categorical to binary (0 for 'No' and 1 for 'Yes').

```python
df_sydney_processed.replace(['No', 'Yes'], [0, 1], inplace=True)
```

### Training Data and Test Data
We split the data into training and testing sets.

```python
df_sydney_processed.drop('Date', axis=1, inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)

features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']
```

## Linear Regression
We use the Linear Regression model and evaluate its performance.

```python
x_train, x_test, Y_train, Y_test = train_test_split(features, Y, test_size=.2, random_state=10)

LinearReg = LinearRegression()
LinearReg.fit(x_train, Y_train)

predictions = LinearReg.predict(x_test)
```

## K-Nearest Neighbors (KNN)
We use the K-Nearest Neighbors classifier and assess its performance.

```python
k = 4
KNN = KNeighborsClassifier(n_neighbors=k).fit(x_train, Y_train)
predictions2 = KNN.predict(x_test)
```

## Decision Tree Classifier
We employ the Decision Tree classifier and evaluate its performance.

```python
Tree = DecisionTreeClassifier()
Tree = Tree.fit(x_train, Y_train)
predictions3 = Tree.predict(x_test)
```

## Logistic Regression
Lastly, we use Logistic Regression and assess its performance.

```python
x_train2, x_test2, Y_train2, Y_test2 = train_test_split(features, Y, test_size=.2, random_state=1)

LR = LogisticRegression(C=1.0, solver='liblinear').fit(x_train2, Y_train2)
predictions4 = LR.predict(x_test2)
```

## Model Evaluation Metrics
We calculate various evaluation metrics for each model:

- Linear Regression:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared (R2)

- K-Nearest Neighbors:
  - Accuracy
  - Jaccard Index
  - F1 Score

- Decision Tree Classifier:
  - Accuracy
  - Jaccard Index
  - F1 Score

- Logistic Regression:
  - Coefficients
  - Predictions
  - Accuracy
  - Jaccard Index
  - F1 Score

## Summary of Results
Here are the results for each model:

### Linear Regression
- MAE: 0.2563
- MSE: 0.1157
- R2: 0.3402

### K-Nearest Neighbors (KNN)
- Accuracy: 0.8183
- Jaccard Index: 0.7901
- F1 Score: 0.5966

### Decision Tree Classifier
- Accuracy: 0.7588
- Jaccard Index: 0.7122
- F1 Score: 0.5730

### Logistic Regression
- Coefficients: [coefficients_list]
- Accuracy: [accuracy_score]
- Jaccard Index: [jaccard_index]
- F1 Score: [f1_score]

You can use these results to assess the performance of each model for your specific problem.