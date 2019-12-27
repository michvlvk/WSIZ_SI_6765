#ZADANIE 1
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

boston = load_boston()

boston_df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
boston_df['price'] = np.array(list(boston['target']))

crime_rate = boston['data'][:, np.newaxis, 0]
# plt.scatter(crime_rate, boston['target'])

linreg = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(crime_rate, boston['target'], test_size = 0.3)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

print(f'Metryka domyślna: {linreg.score(X_test, y_test)}')
print(f'Metryka r2: {r2_score(y_test, y_pred)}')
print(f'Współczynniki regresji:  {linreg.coef_}')


plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth = 2)
plt.show()


cv_score_r2 = cross_val_score(linreg, crime_rate, boston.target, cv=5, scoring='r2')
print()
print(f'r2: {cv_score_r2}')
print('Precyzja: %0.2f (+/-) %0.2f' % (cv_score_r2.mean(), cv_score_r2.std() * 2))

cv_score_ev = cross_val_score(linreg, crime_rate, boston.target, cv=5, scoring='explained_variance')
print()
print(f'explained_variance: {cv_score_ev}')
print('Precyzja: %0.2f (+/-) %0.2f' % (cv_score_ev.mean(), cv_score_ev.std() * 2))

cv_score_mse = cross_val_score(linreg, crime_rate, boston.target, cv=5, scoring='neg_mean_squared_error')
print()
print(f'neg_mean_squared_error: {cv_score_mse}')
print('Precyzja: %0.2f (+/-) %0.2f' % (cv_score_mse.mean(), cv_score_mse.std() * 2))

print()

plt.scatter(y_test, y_pred)
plt.xlabel("Ceny: $Y_i$")
plt.ylabel("Predyktowane ceny: $\hat{Y}_i$")
plt.title("Ceny vs Predyktowane ceny: $Y_i$ vs $\hat{Y}_i$ na podstawie parametru CRIM")
plt.show()




#ZADANIE2
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

diabetics = load_diabetes()

print('Klucze dostępne w zbiorze danych: ', diabetics.keys())
print("feature names:",diabetics.feature_names)
print(diabetics.DESCR)


# konwersja na obiekt pandas.DataFrame
diabetics_df = pd.DataFrame(diabetics['data'], columns=diabetics['feature_names'])

# doklejenie informacji o cenie do reszty dataframe
diabetics_df['target'] = np.array(list(diabetics['target']))

# wykres
#sns.pairplot(diabetics_df)
#plt.show()

# wybrana cecha: BMI
bmi_ind = diabetics['data'][:, np.newaxis, 4]
plt.scatter(bmi_ind, diabetics['target'])
plt.show()

# Stworzenie regresora liniowego
linreg = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(bmi_ind, diabetics['target'], test_size = 0.5)

linreg.fit(X_train, y_train)

# przewidywanie ceny
y_pred = linreg.predict(X_test)

# domyślna metryka
print('Metryka domyślna: ', linreg.score(X_test, y_test))

# wskaźnik (metryka) r^2
print('Metryka r2: ', r2_score(y_test, y_pred))

# współczynniki regresji
print('Współczynniki regresji:\n', linreg.coef_)

# Wykres regresji
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.show()

# Użyj walidacji krzyżowej.
cv_score_r2 = cross_val_score(linreg, bmi_ind, diabetics.target, cv=5, scoring='r2')
print("R^2 (coefficient of determination) regression score:",cv_score_r2)
cv_score_ev = cross_val_score(linreg, bmi_ind, diabetics.target, cv=5, scoring='explained_variance')
print("Explained variance regression score:",cv_score_ev)
cv_score_mse = cross_val_score(linreg, bmi_ind, diabetics.target, cv=5, scoring='neg_mean_squared_error')
print("Mean squared error regression loss:",cv_score_mse)
cv_score_mae = cross_val_score(linreg, bmi_ind, diabetics.target, cv=5, scoring='neg_mean_absolute_error')
print("Mean absolute error regression loss:",cv_score_mae)
cv_score_max_error = cross_val_score(linreg, bmi_ind, diabetics.target, cv=5, scoring='max_error')
print("MAX Error:", cv_score_max_error)