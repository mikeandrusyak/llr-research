# LLR Cheat Sheet

## Import

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
```

## Read csv

```python
df = pd.read_csv('autos.csv')
# df = pd.read_csv('titanic.csv')
# df = pd.read_csv('creditcard.csv')
```

## Encode

```python
df['Sex_code'] = df['Sex'].map({'female': 1, 'male': 0})
```

## Linear

```python
X = df[['Speed']]
y = df['BrakingDistance']

X_const = sm.add_constant(X)
sm_model = sm.OLS(y, X_const).fit()
print(sm_model.summary())

lin_model = LinearRegression().fit(X, y)
print(lin_model.intercept_)
print(lin_model.coef_[0])
print(lin_model.score(X, y))
```

## Logistic

```python
X = df[['Sex_code']]
y = df['Survived']

# X = df[['Age']]
# y = df['CreditCard']

log_model = LogisticRegression(penalty=None).fit(X, y)
print(log_model.intercept_[0])
print(log_model.coef_[0][0])
```

## Probability

```python
new_data = np.array([[1]])
# new_data = np.array([[80]])

prob = log_model.predict_proba(new_data)[0][1]
print(prob)
```

## Linear plot

```python
plt.scatter(df['Speed'], df['BrakingDistance'])
plt.plot(df['Speed'], lin_model.predict(X), color='red')
plt.xlabel('Speed')
plt.ylabel('BrakingDistance')
plt.show()
```

## Logistic plot

```python
x_range = np.linspace(df['Age'].min(), df['Age'].max(), 100).reshape(-1, 1)
y_prob = log_model.predict_proba(x_range)[:, 1]

plt.scatter(df['Age'], df['CreditCard'], alpha=0.5)
plt.plot(x_range, y_prob, color='red')
plt.xlabel('Age')
plt.ylabel('Probability')
plt.show()
```

## Residuals

```python
residuals = y - lin_model.predict(X)

plt.scatter(lin_model.predict(X), residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()

plt.hist(residuals, bins=15)
plt.show()
```

## Confusion matrix

```python
y_pred = log_model.predict(X)
cm = confusion_matrix(y, y_pred)
print(cm)

cm_df = pd.DataFrame(
    cm,
    index=['Actual: 0', 'Actual: 1'],
    columns=['Pred: 0', 'Pred: 1']
)
print(cm_df)
```

## Accuracy

```python
accuracy = cm.diagonal().sum() / cm.sum()
print(accuracy)
```

## Short

```python
X = df[['Speed']]
y = df['BrakingDistance']
m = LinearRegression().fit(X, y)
print(m.intercept_, m.coef_[0], m.score(X, y))
```

```python
X = df[['Age']]
y = df['CreditCard']
m = LogisticRegression(penalty=None).fit(X, y)
print(m.intercept_[0], m.coef_[0][0])
print(m.predict_proba(np.array([[80]]))[0][1])
```

```python
pred = m.predict(X)
cm = confusion_matrix(y, pred)
print(cm)
print(cm.diagonal().sum() / cm.sum())
```