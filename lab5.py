
# Zadanie 1:
# Wyjaśnij w kilku zdaniach jaka cecha została wywnioskowana przez PCA i co ona intuicyjnie mogłaby oznaczać

from sklearn.decomposition import PCA
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

pca = PCA()
pca.fit(X)

print("Liczba komponentów: ", pca.n_components_)

print("Skład nowych cech:")
print(pca.components_)

print(pca.explained_variance_ratio_)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


import seaborn as sns

iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

targets = map(lambda x: iris['target_names'][x], iris['target'] )

iris_df['species'] = np.array(list(targets))

sns.pairplot(iris_df, hue='species')
plt.show()

pca_limit = PCA(n_components = 1)

X_new = pca_limit.fit_transform(X)

print("Liczba komponentów: ", print(pca_limit.n_components_))

print("Skład nowej cechy:")
print(pca_limit.components_)

print(pca_limit.explained_variance_ratio_) 

X_new[:5]


plt.scatter(X_new, y)
plt.show()

#po przeaalizowaniu wygenerowanych wykresów, mogę stwierdzić (w oparciu o wykres z najwiekszą wariancją
# danych), że cechą wywnioskowaną przez PCA jest 'petal width'


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from sklearn import datasets
import seaborn as sns

cars = fetch_openml('cars1')

#podział na cechy i etykiety
X = cars.data
y = cars.target


pca = PCA()
pca.fit(X)


#PCA tworzy n nowych cech, które odzwierciedlają zmiennosc zbioru
print("Liczba komponentów: ", pca.n_components_)
print("Skład nowych cech:")
print(pca.components_)


#okreslenie jaka cecha ma najwiekszy wplyw na zmiennosc zbioru
print(pca.explained_variance_ratio_)

cars_df = pd.DataFrame(cars['data'], columns=cars['feature_names'])


cars_df['target'] = np.array(list(cars['target'])) # przypisanie informacji o gatunku do dataframe

sns.pairplot(cars_df, hue='target')
plt.show()


pca_limit = PCA(n_components = 1)#redukowanie zbioru do najlepszej cechy

X_new = pca_limit.fit_transform(X)
X_new[:5]
print("Liczba komponentów: ", print(pca_limit.n_components_))

#oryginalne cechy a wywnioskowana cecha
print("Skład nowej cechy:")
print(pca_limit.components_)
print(pca_limit.explained_variance_ratio_)
plt.scatter(X_new, y)
plt.show()



