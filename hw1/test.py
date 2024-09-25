from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
a = np.random.randn(3, 2)


scaler = MinMaxScaler()

a = scaler.fit_transform(a)


print(a)