## numpy
* power
* mean
* numpy.matrix.A
* 数组元素过滤
* 插入一行/列

#### power
```python
"""
power(x1, x2, out=None)
x1 : array_like、The bases.(数组)
x2 : array_like、The exponents.(指数)
"""
x1 = range(6)
# [0, 1, 2, 3, 4, 5]
np.power(x1, 3)
# array([  0,   1,   8,  27,  64, 125])

x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
np.power(x1, x2)
# array([  0.,   1.,   8.,  27.,  16.,   5.])
```

#### mean
```python
"""
def mean(a, axis=None, dtype=None, out=None, keepdims=_NoValue):
a : array_like
axis : 计算均值的方向，0表示按行，1表示按列
"""
a = array([[9, 4], [13, 3], [5, 2]])
a_mean = mean(a, 0)  # 按行计算均值。(9+13+5)/3=9、(4+3+2)/3=3
print(a_mean)
# [ 9.  3.]

a_mean = mean(a, 1)  # 按列计算均值。(9+4)/2=6.5、(13+3)/2=8、(5+2)/2=3.5
print(a_mean)
# [ 6.5  8.   3.5]

```

#### numpy.matrix.A
```python
'''
matrix.A 将matrix类型转成narray类型
Return self as an ndarray object.
Equivalent to np.asarray(self).

'''
x = np.matrix(np.arange(12).reshape((3,4)))
'''
matrix([[ 0, 1, 2, 3],
[ 4, 5, 6, 7],
[ 8, 9, 10, 11]])
'''
x.getA()
'''
array([[ 0, 1, 2, 3],
[ 4, 5, 6, 7],
[ 8, 9, 10, 11]])
'''
```

#### 数组元素过滤
```python
'''
筛选n维数组
'''
a = array([[1, 2], [3, 4], [5, 6], [1, 7]])
print(a[:, 0] == 1)  # 判断第一列与1是否相同
# [ True False False  True]

print(nonzero(a[:, 0] == 1)[0])  # 返回与1相同的索引及索引类型，用[0]将索引数组取出
# [0 3]

print(a[nonzero(a[:, 0] == 1)[0]])  # 根据索引数组得到筛选后的a数组
# [[1 2]
 [1 7]]
```

#### 插入一行/列
```python
'''
column_stack(tup)
row_stack(tup)
参数类型为元组。
'''
# 追加一列
a = array([[1, 2], [2, 4], [3, 2], [1, 4], [3, 3]])
a = column_stack((a, [5, 5, 5, 5, 5]))
print(a)
'''
[[1 2 5]
 [2 4 5]
 [3 2 5]
 [1 4 5]
 [3 3 5]]
'''

# 追加一行
a = array([[1, 2], [2, 4], [3, 2], [1, 4], [3, 3]])
a = row_stack((a, [5, 5]))
print(a)
'''
[[1 2]
 [2 4]
 [3 2]
 [1 4]
 [3 3]
 [5 5]]
'''


```