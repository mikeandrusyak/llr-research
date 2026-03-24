# Шпаргалка Python для LLR

## 1. Імпорт

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
```

## 2. Завантаження даних

```python
# Linear Regression
df = pd.read_csv('autos.csv')

# Logistic Regression
# df = pd.read_csv('titanic.csv')
# df = pd.read_csv('creditcard.csv')
```

Для Titanic:

```python
df['Sex_code'] = df['Sex'].map({'female': 1, 'male': 0})
```

## 3. Лінійна регресія

Модель:

$$
y = b_0 + b_1 x
$$

Типовий приклад:

```python
X = df[['Speed']]
y = df['BrakingDistance']
```

### Statsmodels

```python
X_const = sm.add_constant(X)
sm_model = sm.OLS(y, X_const).fit()
print(sm_model.summary())
```

У `summary()` шукати:

- `R-squared`
- `coef`
- `P>|t|`

### Sklearn

```python
lin_model = LinearRegression().fit(X, y)

b0 = lin_model.intercept_
b1 = lin_model.coef_[0]

print(b0, b1)
print(lin_model.score(X, y))  # R2
```

Рівняння:

```python
y = b0 + b1 * x
```

## 4. Логістична регресія

Модель:

$$
\pi(x) = \frac{1}{1 + e^{-(b_0 + b_1 x)}}
$$

Titanic:

```python
X = df[['Sex_code']]
y = df['Survived']
```

CreditCard:

```python
X = df[['Age']]
y = df['CreditCard']
```

Побудова моделі:

```python
log_model = LogisticRegression(penalty=None).fit(X, y)

b0 = log_model.intercept_[0]
b1 = log_model.coef_[0][0]

print(b0, b1)
```

Ймовірність для нового значення:

```python
new_data = np.array([[1]])      # Sex_code = 1
# new_data = np.array([[80]])   # Age = 80

prob = log_model.predict_proba(new_data)[0][1]
print(prob)
```

## 5. Графіки

### Лінійна регресія

```python
plt.scatter(df['Speed'], df['BrakingDistance'])
plt.plot(df['Speed'], lin_model.predict(X), color='red')
plt.xlabel('Speed')
plt.ylabel('BrakingDistance')
plt.show()
```

### Логістична крива

```python
x_range = np.linspace(df['Age'].min(), df['Age'].max(), 100).reshape(-1, 1)
y_prob = log_model.predict_proba(x_range)[:, 1]

plt.scatter(df['Age'], df['CreditCard'], alpha=0.5)
plt.plot(x_range, y_prob, color='red')
plt.xlabel('Age')
plt.ylabel('Probability')
plt.show()
```

Обов'язково:

- `xlabel`
- `ylabel`

## 6. Residual Analysis для linear

```python
residuals = y - lin_model.predict(X)

plt.scatter(lin_model.predict(X), residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()

plt.hist(residuals, bins=15)
plt.title('Distribution of Residuals')
plt.show()
```

Перевірити:

- середнє залишків приблизно `0`
- точки симетрично навколо `0`
- немає "воронки" -> гомоскедастичність
- немає патернів -> незалежність
- гістограма приблизно нормальна

## 7. Confusion Matrix і Accuracy

```python
y_pred = log_model.predict(X)
cm = confusion_matrix(y, y_pred)
print(cm)
```

Акуратний вивід:

```python
cm_df = pd.DataFrame(
    cm,
    index=['Actual: 0', 'Actual: 1'],
    columns=['Pred: 0', 'Pred: 1']
)
print(cm_df)
```

Accuracy:

```python
accuracy = cm.diagonal().sum() / cm.sum()
print(accuracy)
```

Розшифровка:

- `cm[0,0]` = TN
- `cm[0,1]` = FP
- `cm[1,0]` = FN
- `cm[1,1]` = TP

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

## 8. Найкоротші шаблони

### Linear

```python
X = df[['Speed']]
y = df['BrakingDistance']
m = LinearRegression().fit(X, y)
print(m.intercept_, m.coef_[0], m.score(X, y))
```

### Logistic

```python
X = df[['Age']]
y = df['CreditCard']
m = LogisticRegression(penalty=None).fit(X, y)
print(m.intercept_[0], m.coef_[0][0])
print(m.predict_proba(np.array([[80]]))[0][1])
```

### Confusion Matrix

```python
pred = m.predict(X)
cm = confusion_matrix(y, pred)
print(cm)
print(cm.diagonal().sum() / cm.sum())
```

## 11. Формули

Лінійна регресія:

$$
y = b_0 + b_1 x
$$

Логістична регресія:

$$
\pi(x) = \frac{1}{1 + e^{-(b_0 + b_1 x)}}
$$

Accuracy:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$