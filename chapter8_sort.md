## 排序
* python自带排序
* numpy array排序
* numpy mat排序

#### python自带排序
1. 对元素为int类型的list排序
```python
a = [5,2,1,9,6]
sorted(a)  # 将a从小到大排序,不影响a本身结构
# [1, 2, 5, 6, 9]

sorted(a,reverse = True)  # 将a从大到小排序,不影响a本身结构
# [9, 6, 5, 2, 1]

a.sort()  # 将a从小到大排序,影响a本身结构
a
# [1, 2, 5, 6, 9]
```

2. 对元素类型为String类型的list排序
```python
b = ['aa','BB','bb','zz','CC']
sorted(b)  # 按列表中元素每个字母的ascii码从小到大排序,如果要从大到小,请用sorted(b,reverse=True)下同
# ['BB', 'CC', 'aa', 'bb', 'zz']

c =['CCC', 'bb', 'ffff', 'z']
sorted(c,key=len)  # 按列表的元素的长度排序
# ['z', 'bb', 'CCC', 'ffff']

f = [{'name':'abc','age':20},{'name':'def','age':30},{'name':'ghi','age':25}]  # 列表中的元素为字典
def age(s):
   return s['age']
ff = sorted(f,key = age)  # 自定义函数按列表f中字典的age从小到大排序
# [{'age': 20, 'name': 'abc'}, {'age': 25, 'name': 'ghi'}, {'age': 30, 'name': 'def'}]
```

#### numpy array排序

```python
"""
对数组a排序，排序后直接改变了a
axis：排序沿着数组的方向，0表示按行，1表示按列
kind：排序的算法，提供了快排、混排、堆排
order：不是指的顺序，以后用的时候再去分析这个
"""
a = array([[8, 4], [3, 3], [5, 1]])
aa = a.sort(axis=1)
# aa=None, 排序对a本身生效


"""
对数组a排序，返回一个排序后索引，a不变
"""
a = array([[9, 4], [13, 3], [5, 1]])
sort_index = argsort(a[:, 0])  # 对r*c维的数组排序后返回r*c维的索引数组
aa = a[sort_index]  # 对a按照索引排序，a不变
```

#### numpy mat排序
```python
# 第一种
a = array([[9, 4], [13, 3], [5, 1]])
a = mat(a)

sort_index = a.argsort(0)  # 按列排序，每一列都要排一次。如果想指定只对第一列排序：a[:, 0].argsort(0)
aa = a[sort_index]  # 按照排序索引将a排序
print(sort_index)
print(aa)

'''
结果：
[[2 2]
 [0 1]
 [1 0]]
[[[ 5  1]
  [ 5  1]]

 [[ 9  4]
  [13  3]]

 [[13  3]
  [ 9  4]]]
'''
# 第二种
a = array([[9, 4], [13, 3], [5, 1]])
a = mat(a)

sort_index = a[:, 0].argsort(0)  # 对第一列进行排序
aa = a[sort_index]  # 按照排序索引将a排序
print(sort_index)
print(aa)
'''
结果：
[[2]
 [0]
 [1]]
[[[ 5  1]]

 [[ 9  4]]

 [[13  3]]]
'''

# 第三种
a = array([[9, 4], [13, 3], [5, 1]])
a = mat(a)

sort_index = a[:, 0].argsort(0)  # 对第一列进行排序
aa = a[sort_index][:, 0, :]  # 按照排序索引将a排序并选出第三维
print(sort_index)
print(aa)
'''
结果：
[[2]
 [0]
 [1]]
[[ 5  1]
 [ 9  4]
 [13  3]]
'''

```