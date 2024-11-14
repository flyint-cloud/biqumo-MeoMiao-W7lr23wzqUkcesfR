
教程名称：使用 C\# 入门深度学习


作者：痴者工良


地址：


[https://torch.whuanle.cn](https://github.com)


# 线性代数


目录* [线性代数](https://github.com)
	+ - [推荐书籍](https://github.com)
	+ [基础知识](https://github.com)
		- [标量、向量、矩阵](https://github.com)
		- [Pytorch 的一些数学函数](https://github.com)
	+ [线性代数](https://github.com)
		- [向量](https://github.com)
			* [向量的概念](https://github.com)
			* [向量的加减乘除法](https://github.com)
			* [向量的投影](https://github.com):[FlowerCloud机场](https://hanlianfangzhi.com)
		- [柯西\-施瓦茨不等式](https://github.com)
			* [向量的点积](https://github.com)
			* [向量积](https://github.com)
			* [直线和平面表示法](https://github.com)
		- [矩阵](https://github.com)
			* [矩阵的加减](https://github.com)
			* [矩阵乘法](https://github.com)

### 推荐书籍


大家都知道学习 Pytorch 或 AI 需要一定的数学基础，当然也不需要太高，只需要掌握一些基础知识和求解方法，常见需要的数学基础有线性代数、微积分、概率论等，由于高等数学课程里面同时包含了线性代数和微积分的知识，因此读者只需要学习高等数学、概率论两门课程即可。数学不用看得太深，这样太花时间了，能理解意思就行。


首先推荐以下两本书，无论是否已经忘记了初高中数学知识，对于数学基础薄弱的读者来说，都可以看。


* 《普林斯顿微积分读本》
* 《普林斯顿概率论读本》


国内的书主要是一些教材，学习难度会大一些，不过完整看完可以提升数学水平，例如同济大学出版的《高等数学》上下册、《概率论与数理统计》，不过国内的这些教材主要为了刷题解题、考研考试，可能不太适合读者，而且学习起来的时间也太长了。



接着是推荐《深度学习中的数学》，作者是涌井良幸和涌井贞美，对于入门的读者来说上手难度也大一些，不那么容易看得进去，读者可以在看完本文之后再去阅读这本经典书，相信会更加容易读懂。

另外，千万不要用微信读书这些工具看数学书，排版乱七八糟的，数学公式是各种抠图，数学符号也是用图片拼凑的，再比如公式里面中文英文符号都不分。


建议直接买实体书，容易深度思考，数学要多答题解题才行。就算买来吃灰，放在书架也可以装逼呀。买吧。


本文虽然不要求读者数学基础，但是还是需要知道一些数学符号的，例如求和∑ 、集合交并∩∪等，这些在本文中不会再赘述，读者不理解的时候需要自行搜索资料。


## 基础知识


### 标量、向量、矩阵


笔者只能给出大体的概念，至于数学上的具体定义，这里就不展开了。


标量(scalar)：只有大小没有方向的数值，例如体重、身高。


向量(vector)：既有大小也有方向的数值，可以用行或列来表示。


矩阵(matrix)：由多行多列的向量组成。


张量(Tensor)：在 Pytorch 中，torch.Tensor 类型数据结构就是张量，结构跟数组或矩阵相似。


* Tensor：是PyTorch中的基本数据类型，可以理解为多维数组。 Tensor可以用来表示数据集、模型参数和模型输出等。
* Scalar：是一个特殊类型的Tensor，只有一维。 Scalar用来表示标量值，如学习率、损失值等。
* Vector：是一个特殊类型的Tensor，有一维或两维。 Vector用来表示向量值，如梯度、特征值等。
* Matrix：是一个特殊类型的Tensor，有两维。 Matrix用来表示矩阵值，如权重矩阵、输出矩阵等。


比如说 1\.0、2 这些都是标量，在各种编程语言中都以基础数据类型提供了支持，例如 C\# 的基元类型。


下面将标量转换为 torch.Tensor 类型。



```
var x = torch.tensor(1.0);
var y = torch.tensor(2);

x.print_csharp();
y.print_csharp();

```


```
[], type = Float64, device = cpu, value = 1
[], type = Int32, device = cpu, value = 2

```

将数组转换为 torch.Tensor 类型：



```
var data = new int[ , ]{ {1, 2}, { 3, 4}};
var x_data = torch.tensor(data);

x_data.print_csharp();

```

由于上一章已经讲解了很多数组的创建方式，因此这里不再赘述。


### Pytorch 的一些数学函数


Pytorch 通过 torch.Tensor 表示各种数据类型，torch.Tensor 提供超过 100 多种的张量操作，例如算术运算、线性代数、矩阵操作、采样等。


由于篇幅有限，这里就不单独给出，读者请自行参考以下资料：


[https://pytorch.org/docs/stable/torch.html](https://github.com)


[https://pytorch.ac.cn/docs/stable/torch.html](https://github.com)


## 线性代数


### 向量


#### 向量的概念


在研究力学、物理学等工程应用领域中会碰到两类的量，一类完全由**数值的大小**决定，例如温度、时间、面积、体积、密度、质量等，称为**数量**或**标量**，另一类的量，**只知道数值的大小还不能完全确定所描述量**，例如加速度、速度等，这些量除了大小还有方向，称为向量。


在平面坐标轴上有两点 A(x1,y1)A(x1,y1)、B(x2,y2)B(x2,y2)，以 A 为起点 、B 为终点的线段被称为被称为有向线段，其既有大小也有方向，使用 −−→ABAB→ 表示，使用坐标表示为 (x2−x1,y2−y1)(x2−x1,y2−y1)，如果不强调方向，也可以使用 αα 等符号进行简记。


![image-20241108071154361](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070614824-294474136.png)


A、B 之间的距离称为向量的模，使用 \| −−→ABAB→ \| 或 \| −−→BABA→ \| 或 \| αα \| 表示。


平面中的向量，其距离公式是：


\|−−→AB\|\=√(x2−x1)2\+(y2−y1)2\|AB→\|\=(x2−x1)2\+(y2−y1)2其实原理也很简单，根据勾股定理，AB 的平方等于两个直角边长平方之和，所以：


\|−−→AB\|2\=(x2−x1)2\+(y2−y1)2\|AB→\|2\=(x2−x1)2\+(y2−y1)2
![image-20241108212312023](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070614860-137939569.png)


去平方就是：


\|−−→AB\|\=√(x2−x1)2\+(y2−y1)2\|AB→\|\=(x2−x1)2\+(y2−y1)2
如下图所示，其两点间的距离：


\|−−→AB\|\=√(4−1)2\+(4−1)2\=√18\=3√2\=4\.242640687119285\|AB→\|\=(4−1)2\+(4−1)2\=18\=32\=4\.242640687119285
![image-20241108071828663](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070614879-1515592394.png)


使用 C\# 计算向量的模，结果如下



```
var A = torch.from_array(new[] { 1.0, 1.0 });
var B = torch.from_array(new[] { 4.0, 4.0 });
var a = B - A;

var norm = torch.norm(a);
norm.print_csharp();

```


```
[], type = Float64, device = cpu, value = 4.2426

```


> 注意，计算向量的模只能使用浮点型数据，不能使用 int、long 这些整型。


同理，对于三维空间中的两点 A(x1,y1,z1)A(x1,y1,z1)、B(x2,y2,z2)B(x2,y2,z2) ，距离公式是：


\|−−→AB\|\=√(x2−x1)2\+(y2−y1)2\+(z2−z1)2\|AB→\|\=(x2−x1)2\+(y2−y1)2\+(z2−z1)2
#### 向量的加减乘除法


向量的加法很简单，坐标相加即可。


如图所示，平面中有三点 A(1,1\)、B(3,5\)、C(6,4\)。


![image-20241108205142069](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070615051-1796486240.png)


得到三个向量分别为：$\\overrightarrow{AB} (2,4\) 、、\\overrightarrow{BC} (3,\-1\) 、、\\overrightarrow{AC} (5,3\) $


根据数学上向量的加法可知，−−→ABAB→ \+ −−→BCBC→ \= −−→ACAC→



```
var B = torch.from_array(new[] { 2.0, 4.0 });
var A = torch.from_array(new[] { 3.0, -1.0 });
var a = A + B;

a.print_csharp();

```


```
[2], type = Float64, device = cpu, value = double [] {5, 3}

```

同理，在 Pytorch 中，向量减法也是两个 torch.Tensor 类型相减即可。


推广到三维空间，计算方法也是一样的。



```
var B = torch.from_array(new[] { 2.0, 3.0, 4.0 });
var A = torch.from_array(new[] { 3.0, 4.0, 5.0 });
var a = B - A;

a.print_csharp();

```


```
[3], type = Float64, device = cpu, value = double [] {-1, -1, -1}

```

另外，向量乘以或除以一个标量，直接运算即可，如 −−→AB(2,4)AB→(2,4)，则 3∗−−→AB(2,4)3∗AB→(2,4) \= (6,12\)。


#### 向量的投影


如图所示， −−→AB(2,4)AB→(2,4) 是平面上的向量，如果我们要计算向量在 x、y 上的投影是很简单的，例如在 x 轴上的投影是 2，因为 A 点的 x 坐标是 1，B 点的 x 坐标是 3，所以 3 \- 1 \= 2 为 −−→AB(2,4)AB→(2,4) 在 x 轴上的投影，5 \- 1 \= 4 是在 y 轴上的投影。


![image-20241108211302187](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070614852-1244705322.png)


在数学上使用 Projx(u)Projx(u) 表示向量 u 在 x 上的投影，同理 Projy(u)Projy(u) 是 u 在 y 上的投影。


如果使用三角函数，我们可以这样计算向量在各个轴上的投影。


Projx(u)\=\|−−→AB\|cosα\=\|−−→AC\|Projx(u)\=\|AB→\|cos⁡α\=\|AC→\|Projy(u)\=\|−−→AB\|sinα\=\|−−→BC\|Projy(u)\=\|AB→\|sin⁡α\=\|BC→\|
AC、BC 长度是 4，根据勾股定理得出 AB 长度是 4√242，由于 cosπ2\=√22cosπ2\=22 ，所以 Projx(u)\=4Projx(u)\=4。


![image-20241108212445350](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070614898-342564936.png)


那么在平面中，我们已知向量的坐标，求向量与 x 、y 轴的夹角，可以这样求。


cosα\=x\|v\|cos⁡α\=x\|v\|sinα\=y\|v\|sin⁡α\=y\|v\|
例如上图中 −−→AB(4,4)AB→(4,4)，x 和 y 都是 4，其中 \|v\|\=4√2\|v\|\=42，所以 cosα\=44√2\=√22cos⁡α\=442\=22


从 x、y 轴推广到平面中任意两个向量 αα、ββ，求其夹角 φφ 的公式为：


cosφ\=α⋅β\|α\|⋅\|β\|cos⁡φ\=α⋅β\|α\|⋅\|β\|继续按下图所示，计算 −−→ABAB→、−−→ACAC→ 之间的夹角，很明显，我们按经验直接可以得出夹角 φφ 是 45° 。


![image-20241108221035111](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070615084-2060608368.png)


但是如果我们要通过投影方式计算出来，则根据 α⋅β\|α\|⋅\|β\|α⋅β\|α\|⋅\|β\| ，是 C\# 计算如下。



```
var AB = torch.from_array(new[] { 4.0, 4.0 });
var AC = torch.from_array(new[] { 4.0, 0.0 });

// 点积
var dot = torch.dot(AB, AC);

// 求每个向量的模
var ab = torch.norm(AB);
var ac = torch.norm(AC);

// 求出 cosφ 的值
var cos = dot / (ab * ac);
cos.print_csharp();

// 使用 torch.acos 计算夹角 (以弧度为单位)
var theta = torch.acos(cos);

// 将弧度转换为角度
var theta_degrees = torch.rad2deg(theta);
theta_degrees.print_csharp();

```


```
[], type = Float64, device = cpu, value = 0.70711
[], type = Float64, device = cpu, value = 45

```

![image-20241108221229577](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070614911-1026895378.png)


### 柯西\-施瓦茨不等式


aa、bb 是两个向量，根据前面学到的投影和夹角知识，我们可以将以下公式进行转换。


cosφ\=α⋅β\|α\|⋅\|β\|cos⁡φ\=α⋅β\|α\|⋅\|β\|α⋅β\=\|α\|⋅\|β\|cosφα⋅β\=\|α\|⋅\|β\|cos⁡φ由于 −1≤cosφ≤1−1≤cos⁡φ≤1，所以：


−\|α\|⋅\|β\|≤α⋅β≤\|α\|⋅\|β\|−\|α\|⋅\|β\|≤α⋅β≤\|α\|⋅\|β\|
这个就是 柯西\-施瓦茨不等式。


也就是说，当两个向量的夹角最小时，两个向量的方向相同(角度为0\)，此时两个向量的乘积达到最大值，角度越大，乘积越小。在深度学习中，可以将两个向量的方向表示为相似程度，例如向量数据库检索文档时，可以算法计算出向量，然后根据相似程度查找最优的文档信息。


![image-20241112112037795](./http://20.116.118.174:8081/01.base/images/image-20241112112037795.png)


#### 向量的点积


**点积即向量的数量积，点积、数量积、内积，都是同一个东西。**


两个向量的数量积是标量，即一个数值，而向量积是不同的东西，这里只说明数量积。


数量积称为两个向量的数乘，而向量积才是两个向量的乘法。


向量的数乘公式如下：


a⋅b\=n∑i\=1aibi\=a1b1\+a2b2\+...\+anbna⋅b\=∑i\=1naibi\=a1b1\+a2b2\+...\+anbn
加上前面学习投影时列出的公式，如果可以知道向量的模和夹角，我们也可以这样求向量的点积：


α⋅β\=\|α\|⋅\|β\|cosφα⋅β\=\|α\|⋅\|β\|cos⁡φ
例如 $\\overrightarrow{AB} (2,4\) 、、\\overrightarrow{BC} (3,\-1\) $ 两个向量，如下图所示。


![image-20241108205142069](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070615051-1796486240.png)


计算其点积如下：



```
var B = torch.from_array(new[] { 2.0, 4.0 });
var A = torch.from_array(new[] { 3.0, -1.0 });

var dot = torch.dot(A, B);

dot.print_csharp();

```


```
[], type = Float64, device = cpu, value = 2

```

读者可以试试根据点积结果计算出 ∠ABC∠ABC 的角度。


#### 向量积


在画坐标轴时，我们默认轴上每个点间距都是 1，此时 x、y、z 上的单位向量都是 1，如果一个向量的模是 1，那么这个向量就是单位向量，所以单位向量可以有无数个。


![image-20241113004516264](./http://20.116.118.174:8081/01.base/images/image-20241113004516264.png)


在数学中，我们往往会有很多未知数，此时我们使用 ii、jj、kk 分别表示与 x、y、z 轴上正向一致的三个单位向量，**在数学和物理中，单位向量通常用于表示方向而不关心其大小**。不理解这句话也没关系，忽略。


在不关心向量大小的情况下，我们使用单位向量可以这样表示两个向量：


a\=x1i\+y1j\+z1k\=(x1,y1,z1)a\=x1i\+y1j\+z1k\=(x1,y1,z1)b\=x2i\+y2j\+z2k\=(x2,y2,z2)b\=x2i\+y2j\+z2k\=(x2,y2,z2)
在三维空间中，ii、jj、kk 分别表示三个轴方向的单位向量。


* ii 表示沿 x 轴方向的单位向量。
* jj 表示沿 y 轴方向的单位向量。
* kk 表示沿 z 轴方向的单位向量。


这种方式表示 a 在 x 轴上有 x1x1 个单位，在 y 轴上有 y1y1 个单位，在 z 轴上有 z1z1 个单位。


一般来说，提供这种向量表示法，我们并不关心向量的大小，我们只关心其方向，如下图所示。


![image-20241108223336564](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070615107-118161212.png)


现在我们来求解一个问题，在空间中找到跟 $\\overrightarrow{AB} 、、\\overrightarrow{BC} 同时垂直的向量，例如下图的同时垂直的向量，例如下图的\\overrightarrow{AD} $，很明显，这样的向量不止一个，有无数个，所以我们这个时候要了解什么是法向量和单位向量。


![image-20241113005446796](./http://20.116.118.174:8081/01.base/images/image-20241113005446796.png)


法向量是一个与平面垂直的向量（这里不涉及曲面、曲线这些），要找出法向量也很简单，有两种方法，一种是坐标表示：


a×b\=∣∣
∣∣ijkx1y1z1x2y2z2∣∣
∣∣\=(y1z2−z1y2)i−(x1z2−z1x2)j\+(x1y2−y1x2)ka×b\=\|ijkx1y1z1x2y2z2\|\=(y1z2−z1y2)i−(x1z2−z1x2)j\+(x1y2−y1x2)k这样记起来有些困难，我们可以这样看，容易记得。


a×b\=∣∣
∣∣ijkx1y1z1x2y2z2∣∣
∣∣\=(y1z2−z1y2)i\+(z1x2−x1z2)j\+(x1y2−y1x2)ka×b\=\|ijkx1y1z1x2y2z2\|\=(y1z2−z1y2)i\+(z1x2−x1z2)j\+(x1y2−y1x2)k那么法向量 nn 的 x\=(y1z2−z1y2)x\=(y1z2−z1y2) ，y、z 轴同理，就不给出了，x、y、z 分别就是 i、j、k 前面的一块符号公式，所以法向量为：


n(y1z2−z1y2,z1x2−x1z2,x1y2−y1x2)n(y1z2−z1y2,z1x2−x1z2,x1y2−y1x2)
任何一条下式满足的向量，都跟 aa、bb 组成的平面垂直。


c\=(y1z2−z1y2)i\+(z1x2−x1z2)j\+(x1y2−y1x2)kc\=(y1z2−z1y2)i\+(z1x2−x1z2)j\+(x1y2−y1x2)k例题如下。


求与 a\=3i−2j\+4ka\=3i−2j\+4k ，b\=i\+j−2kb\=i\+j−2k 都垂直的法向量 。


首先提取 aa 在每个坐标轴上的分量 (3,−2,4)(3,−2,4)，b 的分量为 (1,1,−2)(1,1,−2)。


则：


a×b\=∣∣
∣∣ijk3−2411−2∣∣
∣∣\=(4−4)i\+(4−(−6))j\+(3−(−2))k\=10j\+5ka×b\=\|ijk3−2411−2\|\=(4−4)i\+(4−(−6))j\+(3−(−2))k\=10j\+5k所以法向量 n(0,10,5)n(0,10,5)。


这就是通过向量积求得与两个向量都垂直的法向量的方法。


你甚至可以使用 C\# 手撸这个算法出来：



```
var A = torch.tensor(new double[] { 3.0, -2, 4 });

var B = torch.tensor(new double[] { 1.0, 1.0, -2.0 });
var cross = Cross(A, B);
cross.print();

static Tensor Cross(Tensor A, Tensor B)
{
    if (A.size(0) != 3 || B.size(0) != 3)
    {
        throw new ArgumentException("Both input tensors must be 3-dimensional.");
    }

    var a1 = A[0];
    var a2 = A[1];
    var a3 = A[2];
    var b1 = B[0];
    var b2 = B[1];
    var b3 = B[2];

    var i = a2 * b3 - a3 * b2;
    var j = a3 * b1 - a1 * b3;
    var k = a1 * b2 - a2 * b1;

    return torch.tensor(new double[] { i.ToDouble(), -j.ToDouble(), k.ToDouble() });
}

```


```
[3], type = Float64, device = cpu 0 -10 5

```

由于当前笔者所用的 C\# 版本的 cross 函数不对劲，不能直接使用，所以我们也可以利用内核函数直接扩展一个接口出来。



```
public static class MyTorch
{
    [DllImport("LibTorchSharp")]
    public static extern IntPtr THSLinalg_cross(IntPtr input, IntPtr other, long dim);

    public static Tensor cross(Tensor input, Tensor other, long dim = -1)
    {
        var res = THSLinalg_cross(input.Handle, other.Handle, dim);
        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
        return torch.Tensor.UnsafeCreateTensor(res);
    }
}

```


```
var A = torch.tensor(new double[] { 3.0, -2, 4 });

var B = torch.tensor(new double[] { 1.0, 1.0, -2.0 });

var cross = MyTorch.cross(A, B);
cross.print_csharp();

```


```
[3], type = Float64, device = cpu, value = double [] {0, 10, 5}

```

当前笔者所用版本 other 参数是 Scalar 而不是 Tensor，这里应该是个 bug，最新 main 分支已经修复，但是还没有发布。


![image-20241109024627974](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070615036-1769444845.png)


另外，还有一种通过夹角求得法向量的方法，如果知道两个向量的夹角，也可以求向量积，公式如下：


a×b\=\|a\|⋅\|b\|sinαa×b\=\|a\|⋅\|b\|sin⁡α一般来说，对于空间求解问题，我们往往是可以计算向量积的，然后通过向量积得出 \|a\|⋅\|b\|sinα\|a\|⋅\|b\|sin⁡α 的结果，而不是通过 \|a\|⋅\|b\|sinα\|a\|⋅\|b\|sin⁡α 求出 a×ba×b 。


关于此条公式，这里暂时不深入。


#### 直线和平面表示法


在本小节节中，我们将学习空间中的直线和平面的一些知识。


在空间中的平面，可以使用一般式方程表达：


v\=Ax\+By\+Cz\+Dv\=Ax\+By\+Cz\+D其中 A、B、C 是法向量的坐标，即 n\={A,B,C}n\={A,B,C}。


首先，空间中的直线有三种表示方法，分别是对称式方程、参数式方程、截距式方程。


**直线的对称式方程**


给定空间中的一点 P0(x0,y0,z0)P0(x0,y0,z0) 有一条直线 L 穿过 p0p0 点，以及和非零向量 v\={l,m,n}v\={l,m,n} 平行。


![image-20241109150817967](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070615113-1910651516.png)


直线上任意一点和 p0p0 的向量都和 vv 平行，−−→P0P\={x−x0,y−y0,z−z0}P0P→\={x−x0,y−y0,z−z0}，所以其因为其对应的坐标成比例，所以其截距式方程为：


x−x0l\=y−y0m\=z−z0nx−x0l\=y−y0m\=z−z0n
**直线的参数式方程**


因为：


x−x0l\=y−y0m\=z−z0n\=tx−x0l\=y−y0m\=z−z0n\=t所以：


⎧⎨⎩x\=x0\+lty\=y0\+mtz\=z0\+nt{x\=x0\+lty\=y0\+mtz\=z0\+nt这便是直线的参数式方程。


**直线的一般式方程**


空间中的直线可以看作是两个平面之间的交线，所以直线由两个平面的一般式方程给出：


{v1\=A1x\+B1y\+C1z\+D1v2\=A2x\+B2y\+C2z\+D2{v1\=A1x\+B1y\+C1z\+D1v2\=A2x\+B2y\+C2z\+D2这些公式在计算以下场景问题时很有帮助，不过本文不再赘述。


① 空间中任意一点到平面的距离。


② 直线和平面之间的夹角。


③ 平面之间的夹角。


### 矩阵


矩阵在在线性代数中具有很重要的地位，深度学习大量使用了矩阵的知识，所以读者需要好好掌握。


如下图所示，A 是一个矩阵，具有多行多列，a11、a12、...、a1na11、a12、...、a1n 是一个行，a11、a21、...、am1a11、a21、...、am1 是一个列。


![image-20240910115046782](./http://20.116.118.174:8081/01.base/images/image-20240910115046782.png)


在 C\# 中，矩阵属于二维数组，即 m∗nm∗n ，例如要创建一个 3∗33∗3 的矩阵，可以这样表示：



```
var A = torch.tensor(new double[,]
{
    { 3.0, -2.0, 4.0 },
    { 3.0, -2.0, 4.0 },
    { 3.0, -2.0, 4.0 }
});

A.print_csharp();

```

使用 `.T` 将矩阵的行和列倒过来：



```
var A = torch.tensor(new double[,]
{
    { 3.0, -2.0, 4.0 }
});

A.T.print_csharp();

```

生成的是：



```
{
	{3.0},
	{-2.0},
	{4.0}
}

```

如图所示：


![image-20241109154450656](https://img2024.cnblogs.com/blog/1315495/202411/1315495-20241114070614904-1741231260.png)


#### 矩阵的加减


矩阵的加减法很简单，就是相同位置的数组加减。



```
var A = torch.tensor(new double[,]
{
    { 1.0, 2.0, 4.0 },
    { 1.0, 2.0, 4.0 },
    { 1.0, 2.0, 4.0 }
});

var B = torch.tensor(new double[,]
{
    { 1.0, 1.0, 2.0 },
    { 1.0, 1.0, 2.0 },
    { 1.0, 1.0, 2.0 }
});

(A+B).print_csharp();

```

结果是：



```
{ 
    {2, 3, 6}, 
    {2, 3, 6}, 
    {2, 3, 6}
}

```

如果直接将两个矩阵使用 Pytorch 相乘，则是每个位置的数值相乘，这种乘法称为 Hadamard 乘积：



```
var A = torch.tensor(new double[,]
{
    { 1.0, 2.0 }
});

var B = torch.tensor(new double[,]
{
    { 3.0, 4.0 }
});

// 或者 torch.mul(A, B)
(A * B).print_csharp();

```


```
[2x1], type = Float64, device = cpu, value = double [,] { {3}, {8}}

```

#### 矩阵乘法


我们知道，向量内积可以写成 x2x1\+y2y1\+z2z1x2x1\+y2y1\+z2z1，如果使用矩阵，可以写成：


\[x1y1z1]⋅⎡⎢⎣x2y2z2⎤⎥⎦\=x2x1\+y2y1\+z2z1\[x1y1z1]⋅\[x2y2z2]\=x2x1\+y2y1\+z2z1换成实际案例，则是：


\[123]⋅⎡⎢⎣456⎤⎥⎦\=1∗4\+2∗5\+3∗6\=32\[123]⋅\[456]\=1∗4\+2∗5\+3∗6\=32使用 C\# 计算结果：



```
var a = torch.tensor(new int[] { 1, 2, 3 });
var b = torch.tensor(new int[,] { { 4 }, { 5 }, { 6 } });

var c = torch.matmul(a,b);
c.print_csharp();

```


```
[1], type = Int32, device = cpu, value = int [] {32}

```

上面的矩阵乘法方式使用 \*\*A ⊗ B \*\* 表示，对于两个多行多列的矩阵乘法，则比较复杂，下面单独使用一个小节讲解。


\*\*A ⊗ B \*\*


矩阵的乘法比较麻烦，在前面，我们看到一个只有行的矩阵和一个只有列的矩阵相乘，结果只有一个值，但是对于多行多列的两个矩阵相乘，矩阵每个位置等于 A 矩阵行和 B 矩阵列相乘之和。


比如下面是一个简单的 `2*2` 矩阵。


\[a11a12a21a22]⋅\[b11b12b21b22]\=\[c11c12c21c22]\[a11a12a21a22]⋅\[b11b12b21b22]\=\[c11c12c21c22]因为 c11c11 是第一行第一列，所以 c11c11 是 A 矩阵的第一行乘以 B 第一列的内积。


c11\=\[a11a12]⋅\[b11b21]\=a11∗b11\+a12∗b21c11\=\[a11a12]⋅\[b11b21]\=a11∗b11\+a12∗b21因为 c12c12 是第一行第二列，所以 c12c12 是 A 矩阵的第一行乘以 B 第二列的内积。


c12\=\[a11a12]⋅\[b12b22]\=a11∗b12\+a12∗b22c12\=\[a11a12]⋅\[b12b22]\=a11∗b12\+a12∗b22因为 c21c21 是第二行第一列，所以 c21c21 是 A 矩阵的第二行乘以 B 第一列的内积。


c21\=\[a21a22]⋅\[b22b21]\=a21∗b11\+a22∗b21c21\=\[a21a22]⋅\[b22b21]\=a21∗b11\+a22∗b21因为 c22c22 是第二行第二列，所以 c22c22 是 A 矩阵的第二行乘以 B 第二列的内积。


c22\=\[a21a22]⋅\[b12b22]\=a21∗b12\+a22∗b22c22\=\[a21a22]⋅\[b12b22]\=a21∗b12\+a22∗b22例题如下：


\[1234]⋅\[5678]\=\[(1∗5\+2∗7)(1∗6\+2∗8)(3∗5\+4∗7)(3∗6\+4∗8)]\=\[19224350]\[1234]⋅\[5678]\=\[(1∗5\+2∗7)(1∗6\+2∗8)(3∗5\+4∗7)(3∗6\+4∗8)]\=\[19224350]使用 C\# 计算多行多列的矩阵：



```
var A = torch.tensor(new double[,]
{
    { 1.0, 2.0 },
    { 3.0, 4.0 }
});

var B = torch.tensor(new double[,]
{
     { 5.0 , 6.0 },
     { 7.0 , 8.0 }
});

torch.matmul(A, B).print_csharp();

```


```
{ {19, 22}, {43, 50}}

```

