import numpy as np
from data import Dataset

"""
Queremos buscar la ecuación de la
recta: y = w1*x1 + w2*x2 + ... + wn*xn + b
que mejor logren explicar/predecir el
precio de una casa (y) en base a las
características/features (x) de la misma

Vamos a buscar estos parámetros
mediante gradient descent y buscamos
minimizar el Error Cuadrático Medio (ECM)

El dataset que vamos a utilizar es el
de sklearn.datasets.load_boston
"""


class RegresionLinealMultiple:
    def __init__(self, n_features: int, lr: float = 0.05, epoch: int = 150):
        self.lr = lr
        self.epoch = epoch
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0

    def forward(self, X: np.ndarray):
        """
        Realiza la prediccion de nuestro modelo lineal
        y = w1*x1 + w2*x2 + ... + wn*xn + b

        Ejemplo:
        x =  [                          w = [
                [a11, a12, a13],          w1,
                [a21, a22, a23]           w2,
            ]                             w3,
                                        ]
        np.dot(x, w) = [
                        a11*w1 + a12*w2 + a13*w3,
                        a21*w1 + a22*w2 + a23*w3,
                    ]
        """
        return X.dot(self.w) + self.b

    @staticmethod
    def ecm(y_hat: np.ndarray, y: np.ndarray):
        """
        Función de coste (Error Cuadrático Medio)
        """
        s = y_hat - y
        avg_loss = np.mean(s**2)
        return np.expand_dims(s, axis=1), avg_loss

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entrenamos/Optimizamos el modelo lineal
        """
        for epoch in range(self.epoch):
            # Forward pass
            y_hat = self.forward(X)
            # Calcular las loss
            s, avg_loss = self.ecm(y_hat, y)
            # Calcular gradiente
            # dL/dw1, dL/dw2, dL/dwn y dL/db
            db = np.mean(2*s)
            dw = np.mean(2*X*s, axis=0)
            # Actualizamos param.
            self.b = self.b - self.lr * db
            self.w = self.w - self.lr * dw
            # Mostrar stats.
            if epoch % 20 == 0:
                print(f'Epoch {epoch} train avg_loss {avg_loss:.5f}')


if __name__ == "__main__":
    ds = Dataset()
    X_train, X_test, y_train, y_test = ds.obtener_datos()
    reg_lineal_mult = RegresionLinealMultiple(X_train.shape[1], lr=0.1, epoch=750)
    reg_lineal_mult.fit(X_train, y_train)
    # Vemos que tan bien esta performando en datos que nunca vió
    y_hat_test = reg_lineal_mult.forward(X_test)
    _, test_avg_loss = reg_lineal_mult.ecm(y_hat_test, y_test)
    print(f'test_avg_loss {test_avg_loss:.5f}')
