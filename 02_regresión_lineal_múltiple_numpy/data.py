import numpy as np
from typing import List
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Dataset:
    """
    Boston house prices
    ---------------------------

    :Numero de ejemplos: 506

    :Number de atributos: 13 numeric/categorical predictive.

    :Atributos (en orden):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
    """

    def __init__(self, columnas: List[str] = None):
        boston_ds = load_boston()
        self.df = pd.DataFrame(boston_ds.data, columns=boston_ds.feature_names)
        if columnas is not None:
            self.df = self.df.loc[:, [columnas]]
        self.df['price'] = boston_ds.target

    def obtener_datos(self):
        """
        Devolvemos `x` e `y` listo para usar en el modelo lineal

        Devuelve X_train, X_test, y_train, y_test
        """
        self.df.iloc[:, :-1] = self.normalizar(self.df.iloc[:, :-1])
        return train_test_split(self.df.iloc[:, :-1].to_numpy(),  # X
                                self.df.iloc[:, -1].to_numpy(),  # y
                                test_size=0.25,  # Dividimos nuestro train set en: 75% train 25% test
                                shuffle=True, random_state=1234)  # Mezclamos y seteamos seed para reproducir resultados

    @staticmethod
    def normalizar(X):
        """
        Normaliza los valores entre 0 y 1

        Con nuestro modelo de y = w1*x1 + w2*x2 + ... + b
        si x1 toma valores entre [0, 100000] y x2 toma valores entre [0, 1], vamos a querer
        actualizar w1 mucho mas que w2 por la magnitud de x1. Por ejemplo, para nuestro
        vector gradiente de v_l = [2*x1*s, 2*x2*s] la actualización de los pesos es
        proporcional al dato de entrada, por lo que vamos a dar más importancia a w1.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(X)
