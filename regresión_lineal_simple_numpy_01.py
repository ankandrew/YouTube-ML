import numpy as np

"""
Queremos buscar los valores óptimos de w y b
de nuestro modelo lineal y=w*x+b que mejor
se ajusten a nuestros datos de entrenamiento

Datos de entrenamiento:
X=[0, 1, 2, 3]
Y=[1, 3, 5, 7]

Ecuación ideal -> y=2*x+1
Vamos a buscar estos parámetros
mediante gradient descent
"""


class RegresionLineal:
    def __init__(self, lr: float = 0.05, epoch: int = 150):
        self.lr = lr
        self.epoch = epoch
        self.w = 1
        self.b = 2

    def forward(self, x: np.ndarray):
        """
        Realiza la prediccion de nuestro modelo lineal (y=w*x+b)
        """
        return self.w * x + self.b

    @staticmethod
    def ecm(y_hat: np.ndarray, y: np.ndarray):
        """
        Función de coste (Error Cuadrático Medio)
        """
        s = y_hat - y
        loss = s ** 2
        return s, np.mean(loss)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Entrenamos/Optimizamos el modelo lineal
        """
        for epoch in range(self.epoch):
            # Forward
            y_hat = self.forward(x)
            # Calculamos loss
            s, avg_loss = self.ecm(y_hat, y)
            # Calculamos derivadas
            # dL/dw y dL/db
            dw = np.mean(2 * x * s)
            db = np.mean(2 * s)
            # Actualizamos parametros
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
            # Mostrar stats.
            if epoch % 20 == 0:
                print(f'Epoch {epoch} la loss {avg_loss:.4f}')


if __name__ == "__main__":
    x_train = np.array([0, 1, 2, 3], dtype=np.float32)
    y_train = np.array([1, 3, 5, 7], dtype=np.float32)
    linear_reg = RegresionLineal(lr=0.05, epoch=180)
    linear_reg.fit(x_train, y_train)
    # y=2*x+1
    print(f'w termino con valor de {linear_reg.w} y b termino con valor de {linear_reg.b}')
