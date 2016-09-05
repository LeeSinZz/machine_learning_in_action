## chapter5
1. Logistic Regression gradient ascent方法得到weights
缺点：
```python
# error为(row * 1)矩阵，data_matrix为(column * row)矩阵
# 计算复杂度很高，即：需要row * column次计算
weights += alpha * data_matrix.transpose() * error
```
2. stochastic gradient ascent
缺点：weights收敛速度很慢，意味着需要更多的迭代次数。

3. stochastic gradient ascent通过随机样本选择 + 动态减小alpha值提高weights收敛速度

```python
'''
subplot(numRows, numCols, plotNum)
subplot将整个绘图区域等分为numRows行 * numCols列个子区域，
然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1。
'''

plt.subplot(221)  # 第一行的左图
plt.subplot(222)  # 第一行的右图
plt.subplot(212)  # 第二整行
plt.show()


# +++++
# label : 给所绘制的曲线一个名字，此名字在图示(legend)中显示。只要在字符串前后添加"$"符号，matplotlib就会使用其内嵌的latex引擎绘制的数学公式。
# color : 指定曲线的颜色
# linewidth : 指定曲线的宽度
# "b--"指定曲线的颜色和线型
plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
plt.plot(x,z,"b--",label="$cos(x^2)$")


# +++++
plt.ylim(-1.2,1.2)  设置Y轴的范围
plt.legend()  # 显示图示


# +++++
weight0 = array(weight_arr)[:, 0]  # 返回二维数组的第一列

```

## 2 matplotlib-绘制精美的图表

matplotlib API包含有三层：
* backend_bases.FigureCanvas : 图表的绘制领域
* backend_bases.Renderer : 知道如何在FigureCanvas上如何绘图
* artist.Artist : 知道如何使用Renderer在FigureCanvas上绘图

FigureCanvas和Renderer需要处理底层的绘图操作，例如使用wxPython在界面上绘图，或者使用PostScript绘制PDF。Artist则处理所有的高层结构，例如处理图表、文字和曲线等的绘制和布局。通常我们只和Artist打交道，而不需要关心底层的绘制细节。
Artists分为简单类型和容器类型两种。简单类型的Artists为标准的绘图元件，例如Line2D、 Rectangle、 Text、AxesImage 等等。而容器类型则可以包含许多简单类型的Artists，使它们组织成一个整体，例如Axis、 Axes、Figure等。

直接使用Artists创建图表的标准流程如下：
* 创建Figure对象
* 用Figure对象创建一个或者多个Axes或者Subplot对象
* 调用Axies等对象的方法创建各种简单类型的Artist

下面首先调用pyplot.figure辅助函数创建Figure对象，然后调用Figure对象的add_axes方法在其中创建一个Axes对象，add_axes的参数是一个形如[left, bottom, width, height]的列表，这些数值分别指定所创建的Axes对象相对于fig的位置和大小，取值范围都在0到1之间：
```python
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.7, 0.3])
```
然后我们调用ax的plot方法绘图，创建一条曲线，并且返回此曲线对象(Line2D)。
```python
line, = ax.plot([1,2,3],[1,2,1])
ax.lines
[<matplotlib.lines.Line2D object at 0x0637A3D0>]
line
<matplotlib.lines.Line2D object at 0x0637A3D0>
```
ax.lines是一个为包含ax的所有曲线的列表，后续的ax.plot调用会往此列表中添加新的曲线。如果想删除某条曲线的话，直接从此列表中删除即可。

#### 2.1 Artist的属性
图表中的每个元素都用一个matplotlib的Artist对象表示，而每个Artist对象都有一大堆属性控制其显示效果。例如Figure对象和Axes对象都有patch属性作为其背景，它的值是一个Rectangle对象。通过设置此它的一些属性可以修改Figrue图表的背景颜色或者透明度等属性，下面的例子将图表的背景颜色设置为绿色：
```python
fig = plt.figure()
fig.show()
fig.patch.set_color("g")
fig.canvas.draw()
```
patch的color属性通过set_color函数进行设置，属性修改之后并不会立即反映到图表的显示上，还需要调用fig.canvas.draw()函数才能够更新显示。

下面是Artist对象都具有的一些属性：
* alpha : 透明度，值在0到1之间，0为完全透明，1为完全不透明
* animated : 布尔值，在绘制动画效果时使用
* axes : 此Artist对象所在的Axes对象，可能为None
* clip_box : 对象的裁剪框
* clip_on : 是否裁剪
* clip_path : 裁剪的路径
* contains : 判断指定点是否在对象上的函数
* figure : 所在的Figure对象，可能为None
* label : 文本标签
* picker : 控制Artist对象选取
* transform : 控制偏移旋转
* visible : 是否可见
* zorder : 控制绘图顺序

```python
# fig.set_alpha(0.5*fig.get_alpha())
fig.set(alpha=0.5, zorder=2)
# matplotlib.pyplot.getp 函数可以方便地输出Artist对象的所有属性名和值
plt.getp(fig.patch)
```

#### 2.2 Figure容器
现在我们知道如何观察和修改已知的某个Artist对象的属性，接下来要解决如何找到指定的Artist对象。前面我们介绍过Artist对象有容器类型和简单类型两种，这一节让我们来详细看看容器类型的内容。

最大的Artist容器是matplotlib.figure.Figure，它包括组成图表的所有元素。图表的背景是一个Rectangle对象，用Figure.patch属性表示。当你通过调用add_subplot或者add_axes方法往图表中添加轴(子图时)，这些子图都将添加到Figure.axes属性中，同时这两个方法也返回添加进axes属性的对象，注意返回值的类型有所不同，实际上AxesSubplot是Axes的子类。
```bash
>>> fig = plt.figure()
>>> ax1 = fig.add_subplot(211)
>>> ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.3])
>>> ax1
<matplotlib.axes.AxesSubplot object at 0x056BCA90>
>>> ax2
<matplotlib.axes.Axes object at 0x056BC910>
>>> fig.axes
[<matplotlib.axes.AxesSubplot object at 0x056BCA90>,
<matplotlib.axes.Axes object at 0x056BC910>]
```
画两条直线
```bash
>>> from matplotlib.lines import Line2D
>>> fig = plt.figure()
>>> line1 = Line2D([0,1],[0,1], transform=fig.transFigure, figure=fig, color="r")
>>> line2 = Line2D([0,1],[1,0], transform=fig.transFigure, figure=fig, color="g")
>>> fig.lines.extend([line1, line2])
>>> fig.show()
```
Figure对象有如下属性包含其它的Artist对象：
* axes : Axes对象列表
* patch : 作为背景的Rectangle对象
* images : FigureImage对象列表，用来显示图片
* legends : Legend对象列表
* lines : Line2D对象列表
* patches : patch对象列表
* texts : Text对象列表，用来显示文字

#### 2.3 Axes容器
Axes容器是整个matplotlib库的核心，它包含了组成图表的众多Artist对象，并且有许多方法函数帮助我们创建、修改这些对象。和Figure一样，它有一个patch属性作为背景，当它是笛卡尔坐标时，patch属性是一个Rectangle对象，而当它是极坐标时，patch属性则是Circle对象。例如下面的语句设置Axes对象的背景颜色为绿色：
```bash
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
>>> ax.patch.set_facecolor("green")
```
当你调用Axes的绘图方法（例如plot），它将创建一组Line2D对象，并将所有的关键字参数传递给这些Line2D对象，并将它们添加进Axes.lines属性中，最后返回所创建的Line2D对象列表：
```bash
>>> x, y = np.random.rand(2, 100)
>>> line, = ax.plot(x, y, "-", color="blue", linewidth=2)
>>> line
<matplotlib.lines.Line2D object at 0x03007030>
>>> ax.lines
[<matplotlib.lines.Line2D object at 0x03007030>]
```
注意plot返回的是一个Line2D对象的列表，因为我们可以传递多组X,Y轴的数据，一次绘制多条曲线。

与plot方法类似，绘制直方图的方法bar和绘制柱状统计图的方法hist将创建一个Patch对象的列表，每个元素实际上都是Patch的子类Rectangle，并且将所创建的Patch对象都添加进Axes.patches属性中：
```bash
>>> ax = fig.add_subplot(111)
>>> n, bins, rects = ax.hist(np.random.randn(1000), 50, facecolor="blue")
>>> rects
<a list of 50 Patch objects>
>>> rects[0]
<matplotlib.patches.Rectangle object at 0x05BC2350>
>>> ax.patches[0]
<matplotlib.patches.Rectangle object at 0x05BC2350>
```
绘制散列图(scatter)
```bash
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
>>> t = ax.scatter(np.random.rand(20), np.random.rand(20))
>>> t # 返回值为CircleCollection对象
<matplotlib.collections.CircleCollection object at 0x06004230>
>>> ax.collections # 返回的对象已经添加进了collections列表中
[<matplotlib.collections.CircleCollection object at 0x06004230>]
>>> fig.show()
>>> t.get_sizes() # 获得Collection的点数
20
```

#### 2.4 Axis容器
Axis容器包括坐标轴上的刻度线、刻度文本、坐标网格以及坐标轴标题等内容。刻度包括主刻度和副刻度，分别通过Axis.get_major_ticks和Axis.get_minor_ticks方法获得。每个刻度线都是一个XTick或者YTick对象，它包括实际的刻度线和刻度文本。为了方便访问刻度线和文本，Axis对象提供了get_ticklabels和get_ticklines方法分别直接获得刻度线和刻度文本：
获得刻度线或者刻度标签之后，可以设置其各种属性，下面设置刻度线为绿色粗线，文本为红色并且旋转45度：
```bash
>>> for label in axis.get_ticklabels():
...     label.set_color("red")
...     label.set_rotation(45)
...     label.set_fontsize(16)
...
>>> for line in axis.get_ticklines():
...     line.set_color("green")
...     line.set_markersize(25)
...     line.set_markeredgewidth(3)
```