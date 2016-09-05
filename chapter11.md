* set


#### set
集合和dict其实是相通的：
* 都是用大括号来表示；
* set和dict类似，也是一组key的集合，但不存储value。由于key不能重复，所以，在set中，没有重复的key。
```python
s = set('abc')
d = {'a': 1, 'b': 2}
print(s)
print(d)
'''
{'a', 'c', 'b'}
{'a': 1, 'b': 2}
'''
```
set(可变集合)与frozenset(不可变集合)的区别：

set无序排序且不重复，是可变的，有add（），remove（）等方法。既然是可变的，所以它不存在哈希值。基本功能包括关系测试和消除重复元素. 集合对象还支持union(联合), intersection(交集), difference(差集)和sysmmetric difference(对称差集)等数学运算.
sets 支持 x in set, len(set),和 for x in set。作为一个无序的集合，sets不记录元素位置或者插入点。因此，sets不支持 indexing, 或其它类序列的操作。
frozenset是冻结的集合，它是不可变的，存在哈希值，好处是它可以作为字典的key，也可以作为其它集合的元素。缺点是一旦创建便不能更改，没有add，remove方法。

注：
set为可变集合：可以对无序集合进行操作。而frozenset不能对集合进行add、remove等操作。
存在哈希值：dict中的key必须存在hash值且不可变对象(list为可变对象)，因为set不存在hash值故set不能当作dict的key
```python
s = set('abc')
d = {s: 1, 'b': 2}
print(d)  # 报错，因为set是不能被hash的。可改用frozenset
# TypeError: unhashable type: 'set'

s = frozenset('abc')
d = {s: 1, 'b': 2}
print(d)  # frozenset是能够被hash的
# {frozenset({'a', 'c', 'b'}): 1, 'b': 2}

```

1. 集合创建
```python
    set()和 frozenset()工厂函数分别用来生成可变和不可变的集合。如果不提供任何参数，默认会生成空集合。如果提供一个参数，则该参数必须是可迭代的，即，一个序列，或迭代器，或支持
迭代的一个对象，例如：一个列表或一个字典。

>>> s=set('cheeseshop') 使用工厂方法创建
>>> s
{'h', 'c', 'o', 's', 'e', 'p'}
>>> type(s)
<type 'set'>
>>> s={'chessseshop','bookshop'}直接创建，类似于list的[]和dict的{}，不同于dict的是其中的值，set会将其中的元素转换为元组
>>> s
{'bookshop', 'chessseshop'}
>>> type(s)
<type 'set'>
不可变集合创建：
>>> t=frozenset('bookshop')
>>> t
frozenset({'h', 'o', 's', 'b', 'p', 'k'})
```

2. 更新可变集合
```python
用各种集合内建的方法和操作符添加和删除集合的成员:
>>> s.add('z') #添加
>>> s
set(['c', 'e', 'h', 'o', 'p', 's', 'z'])
>>> s.update('pypi') #添加,如果原集合已经存在就不再添加
>>> s
set(['c', 'e', 'i', 'h', 'o', 'p', 's', 'y', 'z'])
>>> s.remove('z') #删除
>>> s
set(['c', 'e', 'i', 'h', 'o', 'p', 's', 'y'])
>>> s -= set('pypi')#删除
>>> s
set(['c', 'e', 'h', 'o', 's'])
>>> del s #删除集合

只有可变集合能被修改。试图修改不可变集合会引发异常。
>>> t.add('z')
Traceback (most recent call last):
File "<stdin>", line , in ?
AttributeError: 'frozenset' object has no attribute 'add'
```

3. 子集/超集
```python
>>> set('shop') < set('cheeseshop')
True
>>> set('bookshop') >= set('shop')
True
```

4. 集合类型操作符（所有的集合类型）
1.联合( | )
 两个集合的联合是一个新集合，该集合中的每个元素都至少是其中一个集合的成员，即，属于两个集合其中之一的成员。联合符号有一个等价的方法，union().
 ```python
 >>> s | t
 set(['c', 'b', 'e', 'h', 'k', 'o', 'p', 's'])
 ```
2.交集( & )
 你可以把交集操作比做集合的 AND(或合取)操作。两个集合的交集是一个新集合，该集合中的每
 个元素同时是两个集合中的成员，即，属于两个集合的成员。交集符号有一个等价的方法，intersection()
  ```python
 >>> s & t
 set(['h', 's', 'o', 'p']
 ```
3.差补/相对补集( – )
 两个集合(s 和 t)的差补或相对补集是指一个集合 C，该集合中的元素，只属于集合 s，而不属
 于集合 t。差符号有一个等价的方法，difference().
  ```python
 >>> s - t
 set(['c', 'e'])
 ```
4.对称差分( ^ )
 和其他的布尔集合操作相似， 对称差分是集合的 XOR(又称"异或 ").
 两个集合(s 和 t)的对称差分是指另外一个集合 C,该集合中的元素，只能是属于集合 s 或者集合 t
 的成员，不能同时属于两个集合。对称差分有一个等价的方法，symmetric_difference().
  ```python
 >>> s ^ t
 set(['k', 'b', 'e', 'c'])
 ```
5.混合集合类型操作
 上面的示例中，左边的 s 是可变集合，而右边的 t 是一个不可变集合. 注意上面使用集合操作
 运算符所产生的仍然是可变集合,但是如果左右操作数的顺序反过来，结果就不一样了:
  ```python
 >>> t | s
 frozenset(['c', 'b', 'e', 'h', 'k', 'o', 'p', 's'])
 >>> t ^ s
 frozenset(['c', 'b', 'e', 'k'])
 >>> t - s frozenset(['k', 'b'])
 ```
如果左右两个操作数的类型相同， 既都是可变集合或不可变集合, 则所产生的结果类型是相同的，但如果左右两个操作数的类型不相同(左操作数是 set，右操作数是 frozenset，或相反情况)，则所产生的结果类型与左操作数的类型相同。
