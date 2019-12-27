


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

neighbours_number_description = 'Number of neighbours'
scoring_description = 'Scoring'

file_name_for_chart = "Lab2_Exercise2_Sasiedzi.png"
file_name_for_chart2 = "Lab2_Exercise3_Wina.png"

iris = datasets.load_iris()





# Zadanie 1: sprawdź poniżej inne elementy wczytanego zbioru danych, w szczególności description.
# descriptionz w max 3 zdaniach swoimi słowami co zawiera zbiór danych

print('opis irysów w zbiorze to: ', iris['DESCR'])

# opis zawiera:
# - charakterystyki trzech klas irysów, po 50 instancji każda
# -- przypisaną wysokosc oraz szerokosc:
# --- dwóch cech
# - klasyfikację na podstawie gatunków irysów
# - odnośniki do bibliografii


#**Data Set Characteristics:**
#
#    :Number of Instances: 150 (50 in each of three classes)
#    :Number of Attributes: 4 numeric, predictive attributes and the class
#    :Attribute Information:
#        - sepal length in cm
#        - sepal width in cm
#        - petal length in cm
#        - petal width in cm
#        - class:
#                - Iris-Setosa
#                - Iris-Versicolour
#                - Iris-Virginica




# Zadanie 2:
# Stwórz listę kilku wybranych przez siebie wartości dla parametru n_neighbors
# Używamy funkcji do podzielenia zbioru na zbiór uczący i zbiór testowy

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

neighbours_number = [1, 2, 3, 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
scoring = []

for neighbors_count in neighbours_number:
    # W pętli 'for' użyj kolejnych wartości parametru do stworzenia klasyfikatora
    knn = KNeighborsClassifier(neighbors_count)

    # Następnie naucz go na danych uczących
    knn.fit(X_train, y_train)

    # Zapisz wynik scoringu na danych testowych do osobnej listy
    scoring.append(knn.score(X_test, y_test))

print("Liczba sasiadow, Dokladnosc skoringu")

for neighbours, scoring_ in zip(neighbours_number, scoring):
    print([neighbours_number, scoring])

# Wyświetl wykres zależności między liczbą sąsiadów a dokładnością.
# correlation = np.corrcoef(dokladnosci, lista_n)[0][1]
plt.plot(neighbours_number, scoring)
plt.title('Zaleznosc miedzy liczba sasiadow a dokladnoscia')
plt.xlabel(neighbours_number_description)
plt.ylabel(scoring_description)

# Zapisz do pliku obraz z wykresu
# plt.savefig(file_name_for_chart + '.png')


# Zadanie 3:
# wczytaj dane o winach za pomocą funkcji poniżej


wines = datasets.load_wine()


# Zbadaj zbiór danych. Stwórz wykresy obrazujące ten zbiór danych.


# Zobaczmy jakie dane mamy w zbiorze
print('Elementy zbioru win: ', list(wines.keys()))
# Etykiety które występują
print('Cechy win w zbiorze to: ', wines['feature_names'])

# konwersja na obiekt pandas.DataFrame
wines_df = pd.DataFrame(wines['data'], columns=wines['feature_names'])

# funkcja która nam zamieni wartości 0, 1, 2 na pełny description tekstowy dla gatunku
targets = map(lambda x: wines['target_names'][x], wines['target'])

# doklejenie informacji o gatunku do reszty dataframe
wines_df['species'] = np.array(list(targets))

# wykres
# sns.pairplot(wines_df, hue='species')
# plt.savefig(file_name_for_chart2)

# Podziel zbiór danych na uczący i testowy.
# Podzielmy zbiór na cechy oraz etykiety
X = wines.data
y = wines.target

# Używamy funkcji do podzielenia zbioru na zbiór uczący i zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)

# Tworzymy klasyfikator k-NN używając parametru 5 sąsiadów
knn = KNeighborsClassifier(n_neighbors = 5)

# Wytrenuj klasyfikator kNN
knn.fit(X_train, y_train)

# Przewidujemy wartości dla zbioru testowego
# Dokonaj predykcji na zbiorze testowym
y_pred = knn.predict(X_test)

# Sprawdzamy kilka pierwszych wartości przewidzianych
print(["Wartosci przewidziane: ", y_pred[:5]])

# Sprawdzamy dokładność klasyfikatora
print(["Dokładność klasyfikatora: ", knn.score(X_test, y_test)])

# Wypisz raport z uczenia: confusion_matrix oraz classification_report
print()
print("**** Raport z uczenia - classification_report ****")
print("Precision – What percent of your predictions were correct")
print("Recall – What percent of the positive cases did you catch")
print("The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall"
      "where an F-beta score reaches its best value at 1 and worst score at 0")
print("The support is the number of occurrences of each class in y_true")
print(classification_report(y_test, y_pred))

print()
print("**** Raport z uczenia - confusion_matrix ****")
print("x - Aktualna class")
print("y - Przewidziana class")
print('Klasy win w zbiorze to: ', wines['target_names'])
print(confusion_matrix(y_test, y_pred))

# Jak bardzo wyniki różnią się od prawdziwych wartości?
print("Raporty pokrywają się z rzeczywistością dla class_0")
print("Raporty nie pokrywają się z rzeczywistością dla class_1, class_2")