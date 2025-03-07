import metagrad
from metagrad import Val
import numpy as np

def get_val2DArray(b: np.ndarray):
    assert b.ndim == 2, "b must be a 2D array"
    n, m = b.shape
    a = []
    for i in range(n):
        a.append([Val(j) for j in b[i, :]])
    return a

n, m, t = 3, 4, 5
np_a = np.random.randn(n, m)
np_b = np.random.randn(m, t)
a = get_val2DArray(np_a)
b = get_val2DArray(np_b)

loss = Val(0)
for i in range(n):
    for k in range(t):
        for j in range(m):
            loss = loss + a[i][j] * b[j][k]

print(a)
print(loss)
loss.backward()
print(a[0][0].grad)
