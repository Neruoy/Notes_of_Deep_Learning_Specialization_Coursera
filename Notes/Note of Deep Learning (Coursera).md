#  Deep Learning (Coursera)

## Neural Networks and Deep Learning

#### Why deep learning take off?

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsqtl505j31fk0u04qp.jpg" style="zoom: 25%;" />

### Course Outline

1. Nerural Networks and Deep Learning
2. Improving Deep Neural networks: Hyperparameter tuning, Regularization and Optimization
3. Structuring your Machine Learning project
4. Convolutional Neural Networks
5. Natural Language Processing: Building sequence models

###  Logistic Regression as a Neural Network

> #### **Notation**
>
> **$m$** denote the **number of training examples**
>
> **$n$** denote the **number of features**
>
> A single training example: $(x, y),x\in\mathbb{R}^{n_x},y\in\{0, 1\}$
>
> m training example: $\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\cdots,(x^{(m)},y^{(m)})\}$
>
> $m_{train}=\#train\ examples$, $m_{test}=\#test\ examples$
>
> $X = \left[\begin{matrix}|&|&&|\\x^{(1)}&x^{(2)}&\cdots &x^{(m)}\\|&|&&|\end{matrix}\right] \in\mathbb{R}^{n_x\times m}$ (more easier to implement)
>
> $Y = \left[\begin{matrix}y^{(1)}&y^{(2)}&\cdots &y^{(m)}\end{matrix}\right]\in \mathbb{R}^{1\times m}$

#### Logistic Regression

Given $x$, want $\hat{y}=P(y=1|x),x\in\mathbb{R}^{n_x}$

Parameters: $w\in\mathbb{R}^{n_x}, b\in\mathbb{R}$

Output: $\hat{y}=\sigma(w^\top x+b)$ , $\sigma(z)=\frac{1}{1+e^{-z}}$

##### Loss function:

$$
L(\hat{y},y)=-(y\log \hat{y}+(1-y)\log(1-\hat{y}))
$$

##### Cost function:

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^m L(\hat{y},y)=-\frac{1}{m}\sum_{i=1}^m[y\log \hat{y}+(1-y)\log(1-\hat{y})]
$$

#### Gradient Descent

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsqq4cf0j313o0medoz.jpg" alt="image-20210505014646172" style="zoom: 50%;" />

#### Computation Graph

##### Forward

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsqjioc8j30rs0ak421.jpg" alt="image-20210505015922894" style="zoom:33%;" />

##### Backward

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsqia641j313m0mctyv.jpg" alt="image-20210505021046546" style="zoom:50%;" />

> $$
> \frac{dL(a,y)}{dz}=\frac{dL}{da}\frac{da}{dz}=(-\frac{y}{a}+\frac{1-y}{1-a})a(1-a)=y(a-1)+a(1-y)=a-y
> $$

#### Logistic regression on m examples

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsqgwbnkj313a0m2nax.jpg" alt="image-20210505022605350" style="zoom:50%;" />

#### Vectorization

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsqfhd7bj312i0jodu3.jpg" alt="image-20210505025320505" style="zoom:50%;" />

#### Verctorizing Logistic Regression

$$
X = \left[\begin{matrix}|&|&&|\\x^{(1)}&x^{(2)}&\cdots &x^{(m)}\\|&|&&|\end{matrix}\right] \in\mathbb{R}^{n_x\times m}\\
\begin{align}
\left[\begin{matrix}z^{(1)}&z^{(2)}&\cdots &z^{(m)}\end{matrix}\right]&=\left[\begin{matrix}w^\top x^{(1)}+b&w^\top x^{(2)}+b&\cdots &w^\top x^{(m)}+b\end{matrix}\right]
\\&=\left[\begin{matrix}w^\top x^{(1)}&w^\top x^{(2)}&\cdots &w^\top x^{(m)}\end{matrix}\right]+\left[\begin{matrix}b&b&\cdots &b\end{matrix}\right]
\\&=\left[\begin{matrix}w^\top&w^\top&\cdots &w^\top\end{matrix}\right]\left[\begin{matrix}|&|&&|\\x^{(1)}&x^{(2)}&\cdots &x^{(m)}\\|&|&&|\end{matrix}\right]+\mathbf{b}
\\&=\mathbf{w}^\top\mathbf{X}+\mathbf{b}
\end{align}
$$

```Python
Z = np.dot(w.T, X) + b # b is real number
```

$$
A=\left[\begin{matrix}a^{(1)}&a^{(2)}&\cdots &a^{(m)}\end{matrix}\right]=\sigma(Z)
$$

##### Gradient computation

$$
dZ=A-Y=\left[\begin{matrix}a^{(1)}-y^{(1)}&a^{(2)}-y^{(2)}&\cdots &a^{(m)}-y^{(m)}\end{matrix}\right]
$$


$$
db=\frac{1}{m}\sum_{i=1}^mdz^{(i)}\\
dw=\frac{1}{m}X(dZ)^\top
$$

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsqbsurmj31360lctry.jpg" alt="image-20210505032616217" style="zoom:50%;" />

```python
Z = np.dot(w.T, X) + b
A = sigmoid(Z)
dZ = A - Y
dw = 1/m * X * dZ.T
db = 1/m * np.sum(dZ)

w -= lr * dw
b -= lr * db
```

$$
\frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\\
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})
$$

##### Broadcasting in Python

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsq9sjd2j310w0kon42.jpg" alt="image-20210505034054474" style="zoom:50%;" />

### Shallow Neural Network

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsq8fc7mj313k0lq7iv.jpg" alt="image-20210505234449814" style="zoom:50%;" />
$$
a^{[l]\leftarrow layter}_{i\leftarrow node}
$$

#### Vetorization

Single example
$$
W^{[l]}=\begin{bmatrix}--(w^{[l]}_1)^\top--\\--(w^{[l]}_2)^\top--\\\vdots
\\--(w^{[l]}_n)^\top--\end{bmatrix}
$$


$$
z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}\\
a^{[l]}=\sigma(z^{[l]})
$$

Across the training set
$$
Z^{[l]}=\begin{bmatrix}|&|&&|\\z^{[l](1)}&z^{[l](2)}&\cdots&z^{[l](m)}\\|&|&&|\end{bmatrix}\\
A^{[l]}=\begin{bmatrix}|&|&&|\\a^{[l](1)}&a^{[l](2)}&\cdots&a^{[l](m)}\\|&|&&|\end{bmatrix}\\
A^{[l]}=\sigma(W^{[l]}A^{[l-1]}+b^{[l]}),\ A^{[0]}=X
$$

#### Activation function

$$
a^{[l]}=g(z^{[l]})
$$

$g$ is the activation function

##### Sigmoid function (Never use this except for output layer)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsq3kzkhj30d006edgs.jpg" alt="image-20210506004732494" style="zoom:50%;" />
$$
a=\frac{1}{1+e^{-z}}
$$

##### Hyperbolic tangent function (tanh)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsq26gmlj30cm07i0t2.jpg" alt="image-20210506004841301" style="zoom:50%;" />
$$
a=\tanh (z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$
A shifted version of sigmoid function, works **better** than sigmoid.

**However**, for **both** sigmoid and tanh fucntion, when the z is either **very large** or **very small**, the **gradient becomes very small**.

##### Rectified linear unit function (ReLU) -- Default choice

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsq0tdqsj30ci0700sx.jpg" alt="image-20210506004909311" style="zoom:50%;" />
$$
a=max(0,z)
$$

##### Leaky ReLU

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnspzdc83j30dc07a0tv.jpg" alt="image-20210506004950963" style="zoom:50%;" />
$$
a=max(0.01z,z)
$$

#### Why do we need non-linear activation function?

If there is no non-linear activation function, there is no hidden layers, because it is just the combination of linear function.

#### Derivatives of activation function

##### Sigmoid

$$
\frac{d}{dz}g(z)=g(z)(1-g(z))
$$

##### tanh

$$
\frac{d}{dz}g(z)=1-(\tanh(z))^2=1-(g(z))^2
$$

##### ReLU

$$
\frac{d}{dz}g(z)=
\begin{cases}
0 & \text{if}\ z<0\\
1 & \text{otherwise}
\end{cases}
$$

##### Leaky ReLU

$$
\frac{d}{dz}g(z)=
\begin{cases}
0.01 & \text{if}\ z<0\\
1 & \text{otherwise}
\end{cases}
$$

The gradient at $z=0$ for ReLU and Leaky ReLU is technically not defined. BUT we just take 1.

#### Gradient decent for neural networks

##### Forward propagation

$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}\\
A^{[l]}=\sigma(Z^{[l]}),\ A^{[0]}=X
$$

##### Backward propagation

$$
dZ^{[L]}=A^{[L]}-Y
$$

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnspv8oyaj60ng0gck1202.jpg" alt="image-20210506022337365" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnspttnvwj311k0kyqm7.jpg" alt="image-20210506024146558" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsps096wj311q0ky7fm.jpg" alt="image-20210506024522855" style="zoom:50%;" />

#### Random initialization

The purpose of random initialization is to **break symmetry**.

$w^{[1]}$ = np.random.randn((2 ,2)) * 0.01

$b^{[1]}$ = np.zeros((2, 1))

$b$ do not have symmetric breaking problem.

#### The general methodology to build a Neural Network is to:

1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
2. Initialize the model's parameters
3. Loop:
    - Implement forward propagation

    - Compute loss
    - Implement backward propagation to get the gradients
    - Update parameters (gradient descent)

### Deep Neural Network

> **Notation**
>
> $L$ is the number of layers
>
> $n^{[l]}$ is the number of units in layer $l$, $n^{[0]}=n_x$
>
> $a^{[l]}$ is the activations in layer $l$
>
> $w^{[l]},\ b^{[l]}$ are parameters for $z^{[l]}$
>
> $x=a^{[0]},\ \hat y=a^{[L]}$

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnspoqi19j311u0l8h2f.jpg" alt="image-20210506182612063" style="zoom:50%;" />

#### Forward propagation for layer l

Input $a^{[l-1]}$

Output $a^{[l]}$, cache($z^{[l]}$)
$$
Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}\\
A^{[l]}=\sigma(Z^{[l]}),\ A^{[0]}=X
$$
The dimension of $W^{[l]}$ and $b^{[l]}$
$$
W^{[l]}\in\mathbb{R}^{n^{[l]}\times n^{[l-1]}}
$$

$$
b^{[l]}\in\mathbb{R}^{n^{[l]}\times 1}
$$

#### Backward propagation for layer l

Input $da^{[l]}$

Output $da^{[l-1]},\ dW^{[l]},\ db^{[l]}$
$$
dZ^{[l]}=dA^{[l]}*g^{[l]\prime}(Z^{[l]})\\
dW^{[l]}=\frac{1}{m}dZ^{[l]}A^{[l-1]\top}\\
db^{[l]}=\frac{1}{m}\text{np.sum}(dZ^{[l]},\text{axis=1, keepdims=True})\\
dA^{[l-1]}=W^{[l]\top}dZ^{[l]}
$$
<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsplfdlxj30j808e0uo.jpg" alt="image-20210506213946520"  />

For final layer
$$
dA^{[L]}=\sum_{i=1}^m (-\frac{y^{(i)}}{a^{(i)}}+\frac{1-y^{(i)}}{1-a^{(i)}})
$$

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsnts5njj31a80s6b29.jpg" alt="image-20210506204504399" style="zoom:50%;" />

#### Hyperparameters

- Learning rate $\alpha$
- $\#$interations

- $\#$hidden layer $L$q
- $\#$Hidden units $n^{[l]}$
- choice of activation function
- ...

## Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

### Train/dev/test sets

- Small dataset -- 70/30, 60/20/20 traditional ratio.

- large dataset -- set the dev and test set smaller than tranditional ratio.

**MAKE SURE** the dev and test sets come from **same distribution**.

### Bias and Variance

#### High Bias

- More complex model (bigger network)
- ...

#### High Variance

- More data

- Regularization
- ...

### Regularization

#### Logistic regression

##### l2 regularization

$$
\frac{\lambda}{2m}\|w\|^2_2=\frac{\lambda}{2m}\sum_{j=1}^{n_x}w_j^2=\frac{\lambda}{2m}w^\top w
$$

##### l1 regularization ($w$ will be sparse -- lots of 0 in $w$)

$$
\frac{\lambda}{m}\sum_{j=1}^{n_x}|w|=\frac{\lambda}{m}\|w\|_1
$$

#### Neural Network

##### Frobenius norm

$$
\frac{\lambda}{2m}\sum^L_{l=1}\|W^{[l]}\|^2_F\\
\|W^{[l]}\|^2_F=\sum^{n^{[l]}}_{i=1}\sum^{n^{[l-1]}}_{j=1}(W^{[l]}_{i,j})^2
$$

$$
dW^{[l]}=(from\ backprap)+\frac{\lambda}{m}W^{[l]}
$$

L2 regularization is also called **weight decay** ($1-\frac{\alpha\lambda}{m}<1$).
$$
\begin{align}
W^{[l]}&:=W^{[l]}-\alpha[(from\ backprap)+\frac{\lambda}{m}W^{[l]}]\\
       &:=W^{[l]}-\frac{\alpha\lambda}{m}W^{[l]}-\alpha(from\ backprap)\\
       &:=(1-\frac{\alpha\lambda}{m})W^{[l]}-\alpha(from\ backprap)
\end{align}
$$

$$
J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} \tag{2}
$$



#### Dropout regularization

##### Implementing dropout ("Inverted dropout") -- with l=3 layer

```python
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3) # a3 *= d3
a3 /= keep_prob # Because a3 is reduced by (1 - keep_prob)
```

**Intuition**: Can't rely on any one feature, so have to spread out weights.

#### Other regularization methods

- ##### Data augmentation

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnskg0otmj310m0g249r.jpg" alt="image-20210512183529854" style="zoom:33%;" />

- ##### Early stopping

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnskemty9j31240kedp3.jpg" alt="image-20210512184438925" style="zoom:33%;" />

### Setting Up Optimization Problem

#### Normalizing inputs

1. Move the training set until it has 0 mean.

$$
\mu=\frac{1}{m}\sum_{i=1}^mx^{(i)}\\
x:=x-\mu
$$

2. Normalize the varience

$$
\sigma=\sqrt{\frac{1}{m}\sum_{i=1}^m(x^{(i)})^2}\\
x:=\frac{x}{\sigma}
$$

**Use same $\mu$ and $\sigma$ to normalize the training set and test set**.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnskav044j31340ls4b7.jpg" alt="image-20210512190236641" style="zoom:50%;" />

#### Vanishing/exploding gradients

For a very deep network,

$w>1\to$ exploding gradient

$w<1\to$ vanishing gradient

#### Weight initialization for deep networks (partial solution for v/e gradients)

##### ReLU

He Initialization
$$
W^{[l]}=\text{np.random.randn(shape)}\ *\ \sqrt{\frac{2}{n^{[l-1]}}}
$$

##### tanh

Xavier initialization
$$
W^{[l]}=\text{np.random.randn(shape)}\ *\ \sqrt{\frac{1}{n^{[l-1]}}}
$$
Other version
$$
W^{[l]}=\text{np.random.randn(shape)}\ *\ \sqrt{\frac{2}{n^{[l-1]}+n^{[l]}}}
$$

#### Numerical approximation of gradients

For a very small $\epsilon$,
$$
f^\prime(\theta)\approx\frac{f(\theta+\epsilon)-f(\theta-\epsilon)}{2\epsilon}
$$
Use this formular for **gradient checking**.

#### Gradient checking for a neural network

Take $W^{[1]},\ b^{[1]},\ \cdots,\ W^{[L]},\ b^{[L]}$ and reshape into a big vector $\theta$.

Take $dW^{[1]},\ db^{[1]},\ \cdots,\ dW^{[L]},\ db^{[L]}$ and reshape into a big vector $d\theta$.
$$
\begin{align}
\text{For each i:}\\
&&&&
d\theta_{approx}[i]&=\frac{J(\theta_1,\theta_2,\cdots,\theta_i+\epsilon,\cdots)-J(\theta_1,\theta_2,\cdots,\theta_i-\epsilon,\cdots)}{2\epsilon}\\
   &&&&
   &\approx d\theta[i]=\frac{\partial J}{\partial \theta_i}
\end{align}
$$
To see whether $d\theta_{approx}=d\theta$
$$
\text{Check}\ \ \frac{\|d\theta_{approx}-d\theta\|_2}{\|d\theta_{approx}\|_2+\|d\theta\|_2}
$$

##### Notes for grad check:

- Don't use in training - only to debug.
- If algorithm fails grad check, look at components ($db^{[l]}$, $dw^{[l]}$) to try to indentify bug.
- Remeber regularization.
- Doesn't work with dropout.
- Run at random initialization; perphaps again after some traning.

### Optimization Algorithms

#### Mini-batch gradient descent

Mini-batch t: $X^{\{t\}}, Y^{\{t\}}$

> $x^{(i)} \to\ \text{the ith example}$
>
> $z^{[l]}\to\ \text{the z value for the L layer of the neural network}$
>
>  $X^{\{t\}}, Y^{\{t\}}\to\ \text{mini-batch}$

![image-20210516000309399](https://tva1.sinaimg.cn/large/008i3skNgy1grnsk5u6ilj31380m4ayj.jpg)

![image-20210516000527781](https://tva1.sinaimg.cn/large/008i3skNgy1grnsk4c34ij312e0fkahv.jpg)

#### Choosing the mini-batch size

If nimi-batch size = m: Batch gradient descent.

If nimi-batch size = 1: Stochasitc gradient descent.



IF small train set (m <= 2000): Use batch GD

Typical mini-batch size: 32, 64, 128, 256, 512

The size is to the power of 2 (the computer will run faster).

Make sure mini-batch **fit in CPU/GPU memory**.

##### Set Up

1. **Shuffle**: Create a shuffled version of the training set (X, Y) as shown below. Each column of X and Y represents a training example. Note that the random shuffling is done synchronously between X and Y. Such that after the shuffling the $i^{th}$ column of X is the example corresponding to the $i^{th}$ label in Y. The shuffling step ensures that examples will be split randomly into different mini-batches. 

![image-20210516030016911](https://tva1.sinaimg.cn/large/008i3skNgy1grnsk0ii58j30uw0gon0o.jpg)

```python
permutation = list(np.random.permutation(m))  #m为样本数
shuffled_X = X[:, permutation]
shuffled_Y = Y[:, permutation].reshape((1,m))
```

2. **Partition**: Partition the shuffled (X, Y) into mini-batches of size `mini_batch_size` (here 64). Note that the number of training examples is not always divisible by `mini_batch_size`. The last mini batch might be smaller, but you don't need to worry about this. When the final mini-batch is smaller than the full `mini_batch_size`, it will look like this: 

![image-20210516030149244](https://tva1.sinaimg.cn/large/008i3skNgy1grnsjymd88j30vk0h0tav.jpg)

#### Exponentially weighted averages

$$
V_t=\beta V_{t-1}+(1-\beta)\theta_t
$$

$V_t$ is approximately averaging over $\frac{1}{1-\beta}$ examples
$$
(1-\epsilon)^{\frac{1}{\epsilon}}=\frac{1}{e}\approx0.35
$$
![image-20210516003048824](https://tva1.sinaimg.cn/large/008i3skNgy1grnsjvgotvj31340m2na3.jpg)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnsjty10uj30kg08ijtm.jpg" alt="image-20210516004914526" style="zoom:50%;" />

The **efficiency** of the exponentially weighted averages is because it the computer can overwrite $V_\theta$ over and over (and only 1 line code), and this requires very small memory. In contrast, the direct computation of average requires very large memory when averaging over many examples.

##### Bias correction

$$
V_t=\frac{\beta V_{t-1}+(1-\beta)\theta_t}{1-\beta^t}
$$

![image-20210516005840923](https://tva1.sinaimg.cn/large/008i3skNgy1grnsjqcmryj314q0mox2n.jpg)

Normally, people just wait for warming up in practice.

#### Gradient descent with momentum

Now, implement the parameters update with momentum. The momentum update rule is, for $l = 1, ..., L$: 
$$
\begin{cases}
v_{dW^{[l]}} = \beta v_{dW^{[l]}} + (1 - \beta) dW^{[l]} \\
W^{[l]} = W^{[l]} - \alpha v_{dW^{[l]}}
\end{cases}
$$

$$
\begin{cases}
v_{db^{[l]}} = \beta v_{db^{[l]}} + (1 - \beta) db^{[l]} \\
b^{[l]} = b^{[l]} - \alpha v_{db^{[l]}} 
\end{cases}
$$

In one sentence, the basic idea is to **compute an exponentially weighter average of gradients, and then use that gradient to update weights instead**.

```pseudocode
VdW, Vdb = zeros(same dimension with W and b)
On iteration t:
	Compute dW, db on current mini-batch
	VdW = beta * VdW + (1 - beta) * dW
	Vdb = beta * Vdb + (1 - beta) * db
  W = W - learning_rate * VdW
  b = b - learning_rate * Vdb
```

The most common value for $\beta$ is 0.9.

In some literature, $(1-\beta)$ is omitted. That is just scaling the VdW or Vdb by $\frac{1}{1-\beta}$ (it will affect the learning rate).

![image-20210517221037367](https://tva1.sinaimg.cn/large/008i3skNgy1grnsjmhbdhj31j20i4q8k.jpg)

#### RMSProp (Root Mean Square Prop)

```pseudocode
VdW, Vdb = zeros(same dimension with W and b)
On iteration t:
	Compute dW, db on current mini-batch
	SdW = beta * VdW + (1 - beta) * dW ** 2
	Sdb = beta * Vdb + (1 - beta) * db ** 2
  W = W - learning_rate * dW / (sqrt(SdW + epsilon)) #make sure not 0
  b = b - learning_rate * db / (sqrt(Sdb + epsilon))
```

#### Adam optimization algorithm (Adaptive Moment Estimation)

```pseudocode
VdW = 0, SdW = 0, Vdb = 0, Sdb = 0
On iteration t:
	Compute dW, db on current mini-batch
	VdW = beta_1 * VdW + (1 - beta_1) * dW #'momentum' beta_1
	Vdb = beta_1 * Vdb + (1 - beta_1) * db #'momentum' beta_1
	SdW = beta_2 * VdW + (1 - beta_2) * dW ** 2 #'RMSProp' beta_2
	Sdb = beta_2 * Vdb + (1 - beta_2) * db ** 2 #'RMSProp' beta_2
	VdW_corrected = VdW / (1 - beta_1 ** t)
	Vdb_corrected = Vdb / (1 - beta_1 ** t)
	SdW_corrected = SdW / (1 - beta_2 ** t)
	Sdb_corrected = Sdb / (1 - beta_2 ** t)
	W = W - learning_rate * VdW_corrected / (sqrt(SdW_corrected + epsilon))
	b = b - learning_rate * Vdb_corrected / (sqrt(Sdb_corrected + epsilon))	
```

![image-20210516015530566](https://tva1.sinaimg.cn/large/008i3skNgy1grnsji6iw2j31260k8qib.jpg)

**How does Adam work?**
1. It calculates an exponentially weighted average of past gradients, and stores it in variables $v$ (before bias correction) and $v^{corrected}$ (with bias correction). 
2. It calculates an exponentially weighted average of the squares of the past gradients, and  stores it in variables $s$ (before bias correction) and $s^{corrected}$ (with bias correction). 
3. It updates parameters in a direction based on combining information from "1" and "2".

The update rule is, for $l = 1, ..., L$: 
$$
\begin{cases}
v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial W^{[l]} } \\
v^{corrected}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - (\beta_1)^t} \\
s_{dW^{[l]}} = \beta_2 s_{dW^{[l]}} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial W^{[l]} })^2 \\
s^{corrected}_{dW^{[l]}} = \frac{s_{dW^{[l]}}}{1 - (\beta_2)^t} \\
W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}{\sqrt{s^{corrected}_{dW^{[l]}}} + \varepsilon}
\end{cases}
$$

##### Hyperparameters choice:

$\alpha$: needs to be tuned

$\beta_1$: 0.9 (recommend) $\to$ $dw$ (first moment)

$\beta_2$: 0.999 (recommend) $\to$ $dw^2$ (second moment)

$\epsilon$: $10^{-8}$ (recommend)

#### Learning rate decay

$$
1 \ epoch=1\ pass\ through\ data\\
\alpha=\frac{1}{1+decay\ rate\times epoch\_num}\alpha_0
$$

![image-20210516021021776](https://tva1.sinaimg.cn/large/008i3skNgy1grnsjeqqoij31260lcdno.jpg)

### Hyperparameter Tuning

 (When there are many hyperparameters)**Use `random sampling` instead of `grid search`**!

####  Appropriate scale for hyperparameters

It seems more reasonable to search for hyperparameters on a log scale.

```python
r = -4 * np.random.rand() # r in [-4, 0]
alpha = 10 ** r
```

When tuning hyperparameters for exponentially weighted averages, instead of sample $\beta$, we sample $(1 - \beta)$ on a log scale ($\beta = 1 - 10^r$).

### Batch Normalization

Give some intermediate values in NN $z^{(1)},\cdots,z^{(m)}$
$$
\mu=\frac{1}{m}\sum_iz^{(i)}\\
\sigma^2=\frac{1}{m}\sum_i (z_i-\mu)^2\\
z^{(i)}_{norm}=\frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}\\
\tilde{z}^{(i)}=\gamma z^{(i)}_{norm} + \beta
$$
where $\gamma$ and $\beta$ here are learnable parameters, $\gamma$ and $\beta$ **control the mean and variance** of $\tilde{z}^{(i)}$. 

![image-20210520001012653](https://tva1.sinaimg.cn/large/008i3skNgy1grnsj9upcbj311s0iqtko.jpg)

![image-20210520004803687](https://tva1.sinaimg.cn/large/008i3skNgy1grnsj6lsl8j313a0lstov.jpg)

### Softmax regression

#### Activation function

$$
\large a^{[l]}=\frac{e^{z^{[l]}}}{\sum_{i=1}^Ce^{z^{[l]}}}
$$

where $C$ is the number of classes.

#### Loss function

$$
L(\hat y, y)=-\sum^C_{j=1}y_j\log \hat y_j
$$

#### Back propagation

$$
dZ^{[L]}=\hat y-y
$$

## Structuring Machine Learning Projects

### Orthogonalization

Tune exactly one knob to tune the system.

### Satisficing and Optimizing Metric

$$
\textbf{maximize}\ optimizing\ metric\ \ \textbf{s.t.}\ satisficing\ metric
$$

N metrics: `1` optimizing, `N - 1` statisficing.

### Guideline for choosing dev/test set

Choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on. And the dev set and test set should come from `same distribution`.

Set the test set to be **big enough to give high confidence** i the overall performance of the system.

### Comparing with human-level error

**Bayes error**: the best possible error any function could achieve. **Human-level error** is an `approximation` of Bayes error.

**Avoidable bias**: the difference between bayes errior and training error.

![image-20210520224238509](https://tva1.sinaimg.cn/large/008i3skNgy1grnsj01zfmj312o0ic46x.jpg)

### Improving Model Performance

#### Two fundamental assumptions of supervised learning

1. You can fit the training set pretty well.  `Avoidable bias`
2. The traning set performance generalizes pretty well to the dev/test set.  `Variance`

##### Avoidable bias

- Train bigger model
- Train longer/better optimization algorithms (momentum, RMSProp, Adam...)
- NN architecture/hyperparameters search

##### Variance

- More data
- Regularization (l2, dropout, data augmentation)
- NN architecture/hyperparameters search

### Error analysis

Find a set of mislabeled examples in dev set, and look at the mislabeled examples for false positives and false negatives. Then count up the nimber of errors that fall into various different categories.

![image-20210522185801281](https://tva1.sinaimg.cn/large/008i3skNgy1grnsiu5n46j315g0ec7j6.jpg)

#### Clearning Up Incorrectly Labeled Data

DL algoritms are quite **robust** to random errors in the **training set**.

##### When correcting incorrect dev/test set examples

- Apply same process to dev and test sets to make sure they continue to come from the same distribution.
- Consider examining examples the algorithm got right as well as ones it got wrong.
- Train and dev/test data may now come from slightly different distributions.

#### Bias/varience on mismatched training and dev/test sets

![image-20210522225549795](https://tva1.sinaimg.cn/large/008i3skNgy1grnsior057j30p80akn25.jpg)

![image-20210522230315447](https://tva1.sinaimg.cn/large/008i3skNgy1grnsiko7a8j312a0le7i6.jpg)

#### Addressing Data Mismatch

- Carry out **manual error analysis** to try to understand difference between training and dev/test sets.
  - e.g. noisy
- Make training data more similar; or colloect more data similar to dev/test sets.
  - Artificial data synthesis

### Transfer Learning

Transfer learning makes sense when you have **a lot of** data for the problem you are `transferring from`, and relatively **less** data for the problem you are `transferring to`. To be more specific,

- Task A and B have the **same input x**.
- You have a lot more data for A than B
- Low level features from A could be helpful for B.

![image-20210522232210402](https://tva1.sinaimg.cn/large/008i3skNgy1grnsiedn70j312c0lqnc8.jpg)

### Multi-task learning

- Training on a set of tasks that could benefit from having shared lower-level features.
- Usually: Amount of data you have for each task is quite similar.

### End-to-end Deep Learning

![image-20210523143652507](https://tva1.sinaimg.cn/large/008i3skNgy1grnsibav5xj60zs0j8dof02.jpg)

#### Pros and cons of end-to-end deep learning

##### Pros

- Let the data speak
- Less hand-desogning of components needed

##### Cons

- May need large amount of data
- Excludes potentially useful hand-designed components

## Convolutional Neural Networks

### Computer Vision

Image Classification, Object Detection, Neural Style Transfer...

The problem is that, when we use a large image, the input size will be extremely large. So, we need to implement **convolution operation**.

### Edge Detection Example

![image-20210617122059773](https://tva1.sinaimg.cn/large/008i3skNgy1grnsi4q50qj312w0lodwe.jpg)

![image-20210617122649741](https://tva1.sinaimg.cn/large/008i3skNgy1grnsi0oufoj313a0luqfy.jpg)

![Convolution_schematic](https://tva1.sinaimg.cn/large/008i3skNgy1grnshx2zpeg30em0aojsv.gif)

### Padding

A $n \times n$ image convolved by a $f\times f$ filter, the dimention of output will be $n-f+1\times n-f+1$.

There are two major darwbacks:

- The image shrinks every time when applying convolutional operator.
- The pixels on the corners or on the edges are use much less in the output.

In order to fix both of these problems: **Pad** the image with $p$ (padding amount) before applying convolutional operation. Then the output size becomes $n+2p-f+1\times n+2p-f+1$.

#### Valid and Same convolutions

"Valid" (**no padding**): $n \times n\ *\ f\times f\to n-f+1\times n-f+1$

"Same": Pad so that output size is the same as the input size.
$$
\begin{align}
n + 2p - f + 1 &= n\\
       			 p &= \frac{f - 1}{2}
\end{align}
$$
$f$ is usually (almost always) `odd`.

### Strided convolution

$n\times n$ image, $f\times f$ filter, padding $p$, stride $s$:
$$
n \times n\ *\ f\times f\to \lfloor\frac{n+2p-f}{s}+1\rfloor\times \lfloor\frac{n+2p-f}{s}+1\rfloor
$$

#### Thenical note on cross-correlation vs. convolution

![image-20210617141936996](https://tva1.sinaimg.cn/large/008i3skNgy1grnshrl3ebj312u0lutl8.jpg)

### Convolutions over volumes

The number of channels of the image is equal to the number of channels of filter. If we use multiple filters, we can stack the outputs together to get the final output.

#### Summary (stride = 1)

$$
n\times n\times n_c\ *\ f\times f\times n_c \to n-f+1 \times n-f+1 \times n_c^\prime (\#filters)
$$

### One Layer of a Convolutional Network

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnshmzf00j312u0lue11.jpg" alt="image-20210617201821522" style="zoom:50%;" />

#### Summary of notation

If layer $l$ is a convolution layer:

$f^{[l]}=\text{filter size}$

$p^{[l]}=\text{padding}$

$s^{[l]}=\text{stride}$

$n_C^{[l]}=\text{number of filters}$

Input: $n_H^{[l-1]}\times n_W^{[l-1]}\times n_C^{[l-1]}$

Output: $n_H^{[l]}\times n_W^{[l]}\times n_C^{[l]}$

$n_H^{[l]} = \large\lfloor\frac{n_H^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1\rfloor$

$n_W^{[l]} = \large\lfloor\frac{n_W^{[l-1]}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1\rfloor$

Each filter is: $f^{[l]}\times f^{[l]} \times n_C^{[l-1]}$

Activations: $a^{[l]}\to n_H^{[l]}\times n_W^{[l]}\times n_C^{[l]}$, $A^{[l]}\to m\times n_H^{[l]}\times n_W^{[l]}\times n_C^{[l]}$

Weights: $f^{[l]}\times f^{[l]} \times n_C^{[l-1]} \times n_C^{[l]}$

Bias: $n_C^{[l]}\ -\ (1,1,1,n_C^{[l]})$

<video src="conv_kiank.mp4"></video>

### Simple ConvNet Example

![image-20210617205338917](https://tva1.sinaimg.cn/large/008i3skNgy1grnsexoaaij31300lqk8x.jpg)

#### Types of layer in a convolutional networks:

- Convolution (CONV)
- Pooling (POOL)
- Fully connected (FC)

### Pooling Layers

#### Max pooling

![image-20210617210319029](https://tva1.sinaimg.cn/large/008i3skNgy1grnser2fjzj312w0lstkl.jpg)

![image-20210617210801177](https://tva1.sinaimg.cn/large/008i3skNgy1grnseltl7bj312k0lk4hx.jpg)

The max pooling computation is done `independently` on each of $n_C$ channels.

#### Average pooling (not often used)

Insted of finding maximum, find the average of the "block".

In **very deep neural networks**, you mighe use average pooling to collapse the representations.

#### Summary of pooling

$$
n_H\times n_W \times n_C \to\lfloor\frac{n_W^{[l-1]}-f^{[l]}}{s^{[l]}}+1\rfloor \times\lfloor\frac{n_W^{[l-1]}-f^{[l]}}{s^{[l]}}+1\rfloor\times n_C
$$

Hyperparameters: normally $f=2, s=2$ (this will half the height and width)

There is `no parameters to learn!`

In fact, pooling layers modify the input by choosing one value out of several values in their input volume. Also, to compute derivatives for the layers that have parameters (Convolutions, Fully-Connected), we **still need to backpropagate the gradient through the Pooling layers**.

### CNN Examples (Similar to LeNet-5)

![image-20210617214442225](https://tva1.sinaimg.cn/large/008i3skNgy1grnsdkkh44j313e0lska3.jpg)

![image-20210617220340826](https://tva1.sinaimg.cn/large/008i3skNgy1grnsdp5ac3j31240k8qfq.jpg)

### Why Convolutions?

The reasons why ConvNet needs much less parameter):

- **Parameter sharing**: A feature detector (such as a vertical edge detector) that's useful in one part of the image is probaby useful in another part of the image. 

- **Sparsity of connections**: In each layer, each output value depends only on a small number of inputs.

  ![image-20210617222331810](https://tva1.sinaimg.cn/large/008i3skNgy1grnsdyitv9j30zi0a8gt7.jpg)

Through these two mechanisms, a neural network has a lot fewer parameters which allows it to be trained with smaller training set and is less prone to be overfitting.

**What you should remember**:

* A convolution extracts features from an input image by taking the dot product between the input data and a 2D array of weights (the filter). 
* The 2D output of the convolution is called the feature map
* A convolution layer is where the filter slides over the image and computes the dot product 
    * This transforms the input volume into an output volume of different size 
* Zero padding helps keep more information at the image borders, and is helpful for building deeper networks, because you can build a CONV layer without shrinking the height and width of the volumes
* Pooling layers gradually reduce the height and width of the input by sliding a 2D window over each specified region, then summarizing the features in that region

#### Terminology

##### Window, kernel, filter, pool

The words `kernel` and `filter` are used to refer to the same thing. The word `filter` accounts for the amount of `kernels` that will be used in a single convolution layer. `Pool`is the name of the operation that takes the max or average value of the kernels.

### Convolutional Layer Backward Pass 

#### Computing dA:
This is the formula for computing $dA$ with respect to the cost for a certain filter $W_c$ and a given training example:

$$
dA \mathrel{+}= \sum _{h=0} ^{n_H} \sum_{w=0} ^{n_W} W_c \times dZ_{hw} \tag{1}
$$
Where $W_c$ is a filter and $dZ_{hw}$ is a scalar corresponding to the gradient of the cost with respect to the output of the conv layer Z at the hth row and wth column (corresponding to the dot product taken at the ith stride left and jth stride down). Note that at each time, you multiply the the same filter $W_c$ by a different dZ when updating dA. We do so mainly because when computing the forward propagation, each filter is dotted and summed by a different a_slice. Therefore when computing the backprop for dA, you are just adding the gradients of all the a_slices. 

#### Computing dW:
This is the formula for computing $dW_c$ ($dW_c$ is the derivative of one filter) with respect to the loss:

$$
dW_c  \mathrel{+}= \sum _{h=0} ^{n_H} \sum_{w=0} ^ {n_W} a_{slice} \times dZ_{hw}  \tag{2}
$$
Where $a_{slice}$ corresponds to the slice which was used to generate the activation $Z_{ij}$. Hence, this ends up giving us the gradient for $W$ with respect to that slice. Since it is the same $W$, we will just add up all such gradients to get $dW$. 

#### Computing db:

This is the formula for computing $db$ with respect to the cost for a certain filter $W_c$:
$$
db = \sum_h \sum_w dZ_{hw} \tag{3}
$$
As you have previously seen in basic neural networks, db is computed by summing $dZ$. In this case, you are just summing over all the gradients of the conv output (Z) with respect to the cost. 

### Pooling Layer - Backward Pass

Next, let's implement the backward pass for the pooling layer, starting with the MAX-POOL layer. Even though a pooling layer has no parameters for backprop to update, you still need to backpropagate the gradient through the pooling layer in order to compute gradients for layers that came before the pooling layer. 

#### Max Pooling - Backward Pass 

Before jumping into the backpropagation of the pooling layer, you are going to build a helper function called `create_mask_from_window()` which does the following: 

$$
X = \begin{bmatrix}
1 && 3 \\
4 && 2
\end{bmatrix} \quad \rightarrow  \quad M =\begin{bmatrix}
0 && 0 \\
1 && 0
\end{bmatrix}\tag{4}
$$
As you can see, this function creates a "mask" matrix which keeps track of where the maximum of the matrix is. True (1) indicates the position of the maximum in X, the other entries are False (0). You'll see later that the backward pass for average pooling is similar to this, but uses a different mask.  

**Why keep track of the position of the max?** It's because this is the input value that ultimately influenced the output, and therefore the cost. Backprop is computing gradients with respect to the cost, so anything that influences the ultimate cost should have a non-zero gradient. So, backprop will "propagate" the gradient back to this particular input value that had influenced the cost.

#### Average Pooling - Backward Pass 

In max pooling, for each input window, all the "influence" on the output came from a single input value--the max. In average pooling, every element of the input window has equal influence on the output. So to implement backprop, you will now implement a helper function that reflects this.

For example if we did average pooling in the forward pass using a 2x2 filter, then the mask you'll use for the backward pass will look like: 
$$
 dZ = 1 \quad \rightarrow  \quad dZ =\begin{bmatrix}
1/4 && 1/4 \\
1/4 && 1/4
\end{bmatrix}\tag{5}
$$
This implies that each position in the $dZ$ matrix contributes equally to output because in the forward pass, we took an average. 

### Case studies

#### LeNet - 5

![image-20210619191046040](https://tva1.sinaimg.cn/large/008i3skNgy1grnscw3fr1j313c0lwwt0.jpg)

#### AlexNet

![image-20210619193429376](https://tva1.sinaimg.cn/large/008i3skNgy1grnt1ks0gij31380m0qm6.jpg)

#### VGG - 16 (16 refers to there are 16 layers with weights)

![image-20210619194331666](https://tva1.sinaimg.cn/large/008i3skNgy1grntazm1taj31380m2h2d.jpg)

### Residual Networks (ResNets)

Very deep neural networks are difficult to train beacause of **vanishing** and **exploding** gradient types of problems. The **skip connections** allows you to take activation from one layer and suddenly feed it to another layer even much deeper in the neural network. Using that, you will bulid **ResNet** whihc enables you to train very deep networks.

#### Residual block (Identity block)

$$
\large a^{[l+2]}=g(z^{[l+2]}+a^{[l]})
$$

![image-20210619210738878](https://tva1.sinaimg.cn/large/008i3skNgy1grnvqk9rckj313c0lywt3.jpg)

The `identity block` is the standard block used in ResNets, and corresponds to the case where the input activation (say $a^{[l]}$) has the **same dimension** as the output activation (say $a^{[l+2]}$).

![image-20210620024631382](https://tva1.sinaimg.cn/large/008i3skNgy1gro5j3rl88j31080aiabi.jpg)

The ResNet `"convolutional block"` is the second block type. You can use this type of block when the input and output dimensions **don't match up**. The difference with the identity block is that there is a CONV2D layer in the shortcut path:

![image-20210620031553113](https://tva1.sinaimg.cn/large/008i3skNgy1gro6dnocn1j310i09ajvr.jpg)

In theory, having a deeper network should only help. But in practice, having a plain network (no ResNet) is very deep means that your optimization algorithm just has a much harder time training. And so, in reality, your training error gets **worse** if you pick a network that is **too deep.**

![image-20210619211447829](https://tva1.sinaimg.cn/large/008i3skNgy1grnvxy3tbaj31360lqwr2.jpg)

#### Why ResNet works?

The main reason is that it is so easy for these extra layers to learn the **identity function** that you are kind of guranteed that **it doesn't hurt performance** and thena lot the time you may be get lucky and then even **helps performance**.

![image-20210619212917755](https://tva1.sinaimg.cn/large/008i3skNgy1grnwd1c803j312y0lsdxd.jpg)

![image-20210619213131588](https://tva1.sinaimg.cn/large/008i3skNgy1grnwfcvkcoj312c0kyaq5.jpg)

### Network in Network and 1x1 convolutions

![image-20210619214327080](https://tva1.sinaimg.cn/large/008i3skNgy1grnwrrmaehj31300n8e7p.jpg)

The 1 x 1convolution has nonlinearity, it allows you to learn a more complex function of your network by adding another layer.

The 1 x 1 convolution can **shrink the number of channels**.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grnwu8zbkhj30qg0j4gx5.jpg" alt="image-20210619214550624" style="zoom:50%;" />

### Inception Network

#### Motivation for inception network

![image-20210619215605723](https://tva1.sinaimg.cn/large/008i3skNgy1grnx4wuiitj311e0iun9r.jpg)

The basic idea is that instead of you needing to pick one of these filter sizes or pooling you want and committing to that, you can do them all and just concatenate all the outputs, and **let the network learn whatever parameters it wants to use, whatever the combinations of these filter sizes it wants.**

#### The problem of computational cost

![image-20210619220059424](https://tva1.sinaimg.cn/large/008i3skNgy1grnxa0acfcj30yc0jy7cz.jpg)
$$
\#\text{multiplies}=\#\text{multiplies to compute each of output values}\times \#\text{output values}
$$

#### Using 1 x 1 convolution

![image-20210619221253184](https://tva1.sinaimg.cn/large/008i3skNgy1grnxme5nzwj31300logy5.jpg)

#### Inception module

![image-20210619221810169](https://tva1.sinaimg.cn/large/008i3skNgy1grnxrvrx1pj31380lk4ga.jpg)

### MobileNet

#### Motivation for MobileNets

- Low computational cost at deployment
- Useful for mobile and embedded vision applications

#### Key idea: Normal vs. depthwise-separable convolutions

![image-20210619224525528](https://tva1.sinaimg.cn/large/008i3skNgy1grnyk9dkv2j31040jcn92.jpg)

![image-20210619225249465](https://tva1.sinaimg.cn/large/008i3skNgy1grnyry06obj310w0jstix.jpg)

Cost of normal convolution: **2160**

Cose of depth wise separable convolution: 432 + 240 = **672**

The ratio is equal to
$$
ratio=\frac{1}{n_C^\prime}+\frac{1}{f^2}
$$

#### MobileNet architecture

![image-20210619230510053](https://tva1.sinaimg.cn/large/008i3skNgy1grnz4s8a6wj61300lgwul02.jpg)

#### MobileNet v2 Bottleneck

![image-20210619231219001](https://tva1.sinaimg.cn/large/008i3skNgy1grnzc839mzj31260ik7hr.jpg)

**What you should remember**:

* MobileNetV2's unique features are: 
  * Depthwise separable convolutions that provide lightweight feature filtering and creation
  * Input and output bottlenecks that preserve important information on either end of the block
* Depthwise separable convolutions deal with both spatial and depth (number of channels) dimensions

### EfficientNet

Three things you can do to scale things up or down are:

- **r**: resolution (of images)
- **d**: depth (of networks)
- **w**: width (of layers)

The EfficientNet can help you find a way to scale up or down networks.

### Transfer Learning

![image-20210619233726009](https://tva1.sinaimg.cn/large/008i3skNgy1gro02d5pymj31320m4aw1.jpg)

You could try fine-tuning the model by re-running the optimizer in the last layers to improve accuracy. When you use a smaller learning rate, you take smaller steps to adapt it a little more closely to the new data. In transfer learning, the way you achieve this is by unfreezing the layers at the end of the network, and then re-training your model on the final layers with **a very low learning rate**. Adapting your learning rate to go over these layers in smaller steps can yield more fine details - and higher accuracy.

**What you should remember**:

* To adapt the classifier to new data: Delete the top layer, add a new classification layer, and train only on that layer
* When freezing layers, avoid keeping track of statistics (like in the batch normalization layer)
* Fine-tune the final layers of your model to capture high-level details near the end of the network and potentially improve accuracy 

### Data Augmentation

#### Common augmentation method

- Mirroring
- Random cropping
- Rotation
- Shearing
- Local warping
- ...

#### Color shifting

![image-20210619234311374](https://tva1.sinaimg.cn/large/008i3skNgy1gro08czavaj313c0lwngx.jpg)

This make the learning algorithm more robust to changes in the colors of the images.

**"PCA color augmentation" (AlexNet)**: keep the overall color of the tint the same.

![image-20210619234944258](https://tva1.sinaimg.cn/large/008i3skNgy1gro0f60xtnj31300lsdsk.jpg)

![image-20210620000130425](https://tva1.sinaimg.cn/large/008i3skNgy1gro0rexp9rj31320loarr.jpg)

### Object Detection

#### Object localization

![image-20210620122732346](https://tva1.sinaimg.cn/large/008i3skNgy1grombp4q0kj312k0kaazf.jpg)

In this section, the `upper left` of the image is **(0, 0)**, and at the `lower right` is **(1, 1)**.

The bounding box is denoted by:
$$
(b_x, b_y, b_h, b_w)
$$
where $b_x, b_y$ are the coordinates of the midpoint, $b_h, b_w$ are height and weight respectively.

![image-20210620123434171](https://tva1.sinaimg.cn/large/008i3skNgy1gromiyfvchj313g0lm4l9.jpg)

![image-20210620124434634](https://tva1.sinaimg.cn/large/008i3skNgy1gromtdiq78j313g0m2tsb.jpg)

#### Landmark detection

![image-20210620125030756](https://tva1.sinaimg.cn/large/008i3skNgy1gromzjw83yj31380lyu0h.jpg)

#### Object detection

Before the raise of deep learning, people used **sliding windows** detection with simple linear classifer over hand-engineer features in order to perform object detection.

![image-20210620131015174](https://tva1.sinaimg.cn/large/008i3skNgy1gronk3b5qkj31320lutzm.jpg)

#### Convolutional implementation of sliding windows

![image-20210620131642715](https://tva1.sinaimg.cn/large/008i3skNgy1gronqtfnwaj31340lsk7l.jpg)

![image-20210620132303612](https://tva1.sinaimg.cn/large/008i3skNgy1gronxfb5cqj313e0lw1kx.jpg)

#### Bounding box predictions

![image-20210620135417725](https://tva1.sinaimg.cn/large/008i3skNgy1grootx63sxj313m0mcavo.jpg)

The bounding box for YOLO $(b_x, b_y, b_h, b_w)$ is specified relative to the grid cell. And the width and height of the bounding box are specified as `fractions` of the overall width and height of the grid cell.

![image-20210620140151223](https://tva1.sinaimg.cn/large/008i3skNgy1grop1ryn30j30yq0hwn9k.jpg)

#### Intersection over Union (IoU)

$$
IoU=\frac{Area\ of\ intersection}{Area\ of\ union}
$$

"Correct" if $IoU ≥ 0.5$

More generally, IoU is a measure of the overlap between two bounding boxes.

![image-20210620190734216](https://tva1.sinaimg.cn/large/008i3skNgy1groxvv8ug1j30rs07edi2.jpg)

#### Non-max suppression

Non-max suppression is a way to make sure the algorithm detects each object **only once**.

Non-max means that you are going to output your **maximal probabilities classifications**, but suppress the close-by ones that are non-maximal.

![image-20210620141934839](https://tva1.sinaimg.cn/large/008i3skNgy1gropk7xp22j31360lkqmc.jpg)

#### Anchor boxes

![image-20210620142339721](https://tva1.sinaimg.cn/large/008i3skNgy1gropoh6pkuj313a0lkaq1.jpg)

![image-20210620142737956](https://tva1.sinaimg.cn/large/008i3skNgy1gropslya5nj312w0lk7v2.jpg)

Two cases that the algorithm does not handle well:

- If there are **more than two** objects in the same grid cell.
- Two objects are associated with same grid cell but both of  them have the **same anchor box shape**.

**How to choose anchor boxes?**

People used to just choose anchor boxes by hand or choose maybe 5 or 10 anchor box shapes that spans a variety of shapes that seems to cover the types of objects you seem to detect. As a much more advanced version is to use a **K-means algorithm** to group together two types of objects shapes you tend to get.  And then to use that to select a set of anchor boxes that are most stereotypically representative of the multiple, of the dozens of object classes you're trying to detect.

#### YOLO Algorithm

This algorithm "only looks once" at the image in the sense that it requires **only one forward propagation** pass through the network to make predictions. 

![image-20210620144339169](https://tva1.sinaimg.cn/large/008i3skNgy1groq99sknrj31300lu1de.jpg)

![image-20210620144507089](https://tva1.sinaimg.cn/large/008i3skNgy1groqat3w7vj31380jkkau.jpg)

![image-20210620144716478](https://tva1.sinaimg.cn/large/008i3skNgy1groqd1gpq6j312c0iidts.jpg)

#### **Programming Assignment Example**

##### Inputs and outputs
- The **input** is a batch of images, and each image has the shape (m, 608, 608, 3)
- The **output** is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers $(p_c, b_x, b_y, b_h, b_w, c)$ as explained above. If you expand $c$ into an 80-dimensional vector, each bounding box is then represented by 85 numbers. 

##### Anchor Boxes
* Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes.  For this assignment, 5 anchor boxes were chosen for you (to cover the 80 classes), and stored in the file './model_data/yolo_anchors.txt'
* The dimension for anchor boxes is the second to last dimension in the encoding: $(m, n_H,n_W,anchors,classes)$.
* The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85). 

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grow71z6zqj30w80h448d.jpg" alt="image-20210620180907061" style="zoom:50%;" />

##### Encoding
Let's look in greater detail at what this encoding represents. 

![image-20210620175821794](https://tva1.sinaimg.cn/large/008i3skNgy1grovvweh6oj31900n0dtb.jpg)

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Since you're using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, you'll flatten the last two dimensions of the shape (19, 19, 5, 85) encoding, so the output of the Deep CNN is (19, 19, 425).

![image-20210620175951073](https://tva1.sinaimg.cn/large/008i3skNgy1grovxeyfj1j31720nswm5.jpg)

##### Class score

Now, for each box (of each cell) you'll compute the following element-wise product and extract a probability that the box contains a certain class.

The class score is $score_{c,i} = p_{c} \times c_{i}$: the probability that there is an object $p_{c}$ times the probability that the object is a certain class $c_{i}$.

![image-20210620180235933](https://tva1.sinaimg.cn/large/008i3skNgy1grow0a34rkj31660ii4a3.jpg)

##### Visualizing classes
Here's one way to visualize what YOLO is predicting on an image:

- For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across the 80 classes, one maximum for each of the 5 anchor boxes).
- Color that grid cell according to what object that grid cell considers the most likely.

Doing this results in this picture: 

![image-20210620180553837](https://tva1.sinaimg.cn/large/008i3skNgy1grow3pkl4rj319a0dcgty.jpg)

Note that this visualization isn't a core part of the YOLO algorithm itself for making predictions; it's just a nice way of visualizing an intermediate result of the algorithm. 

##### Visualizing bounding boxes
Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this:  

![image-20210620180823790](https://tva1.sinaimg.cn/large/008i3skNgy1grow6au07oj31ic0e416i.jpg)

##### Non-Max suppression
In the figure above, the only boxes plotted are ones for which the model had assigned a high probability, but this is still too many boxes. You'd like to reduce the algorithm's output to a much smaller number of detected objects.  

To do so, you'll use **non-max suppression**. Specifically, you'll carry out these steps: 
- Get rid of boxes with a low score. Meaning, the box is not very confident about detecting a class, either due to the low probability of any object, or low probability of this particular class.
- Select only one box when several boxes overlap with each other and detect the same object.

The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It's convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:  
- `box_confidence`: tensor of shape $(19, 19, 5, 1)$ containing $p_c$ (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
- `boxes`: tensor of shape $(19, 19, 5, 4)$ containing the midpoint and dimensions $(b_x, b_y, b_h, b_w)$ for each of the 5 boxes in each cell.
- `box_class_probs`: tensor of shape $(19, 19, 5, 80)$ containing the "class probabilities" $(c_1, c_2, ... c_{80})$ for each of the 80 classes for each of the 5 boxes per cell.

![image-20210620190704070](https://tva1.sinaimg.cn/large/008i3skNgy1groxvd88inj31ii0f6kef.jpg)









#### Region proposal: R-CNN

![image-20210620145137605](https://tva1.sinaimg.cn/large/008i3skNgy1groqhkckwtj31340joe0l.jpg)

**R-CNN**

Propose regions. Classify proposed regions one at a time. Output label + bounding box.

**Fast R-CNN**

Propose regions. Use convolution implementation of sliding windows to classify all the proposed regions.

**Faster R-CNN**

Use convolutional network to propose regions.

### Semantic Segmentation

![image-20210620150100741](https://tva1.sinaimg.cn/large/008i3skNgy1groqrc6np8j311g0fwqgf.jpg)

**U-Net**

![image-20210620150557342](https://tva1.sinaimg.cn/large/008i3skNgy1groqwgxkkrj312s0fmgyz.jpg)

#### Transpose Convolutions

![image-20210620151546275](https://tva1.sinaimg.cn/large/008i3skNgy1gror6p04o9j30x60j8wpa.jpg)

#### U-Net Architecture

U-Net builds on a previous architecture called the Fully Convolutional Network, or FCN, which replaces the dense layers found in a typical CNN with a transposed convolution layer that upsamples the feature map back to the size of the original input image, while preserving the spatial information. This is necessary because the dense layers destroy spatial information (the "where" of the image), which is an essential part of image segmentation tasks. An added bonus of using transpose convolutions is that the input size no longer needs to be fixed, as it does when dense layers are used.

Unfortunately, the final feature layer of the FCN suffers from information loss due to downsampling too much. It then becomes difficult to upsample after so much information has been lost, causing an output that looks rough.

U-Net improves on the FCN, using a somewhat similar design, but differing in some important ways. Instead of one transposed convolution at the end of the network, it uses a matching number of convolutions for downsampling the input image to a feature map, and **transposed convolutions** for upsampling those maps back up to the original input image size. It also adds **skip connections**, to retain information that would otherwise become lost during encoding. Skip connections send information to every upsampling layer in the decoder from the corresponding downsampling layer in the encoder, capturing finer information while also keeping computation low. These help prevent information loss, as well as model overfitting.

![image-20210620152900576](https://tva1.sinaimg.cn/large/008i3skNgy1grorkgqlllj312i0kg481.jpg)

##### Encoder

The encoder is a stack of various conv_blocks:

Each `conv_block()` is composed of 2 **Conv2D** layers  with ReLU activations.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grp18j6j4dj30py0iotf4.jpg" alt="image-20210620210331805" style="zoom: 50%;" />

##### Decoder

The decoder, or upsampling block, upsamples the features back to the original image size. At each upsampling level, you'll take the output of the corresponding encoder block and concatenate it before feeding to the next decoder block.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grp1ceip4bj30s40fcada.jpg" alt="image-20210620210715479" style="zoom:50%;" />

### Face Recognition

![image-20210621002029510](https://tva1.sinaimg.cn/large/008i3skNgy1grp6xh3o7sj31380luduu.jpg)

#### One-shot learning

Learning from **one example** to recognize the person again. The method is learn a `similarity function`.
$$
d(\text{img1},\text{img2})=\text{degree of difference between images}
$$
If $d(\text{img1},\text{img2})≤\tau\to\text{'same'}$, $d(\text{img1},\text{img2})>\tau\to\text{'diff'}$

#### Siamese Network

![image-20210621192444487](https://tva1.sinaimg.cn/large/008i3skNgy1grq403f47lj310y0k4aj7.jpg)

#### Triplet loss

For an image $x$, its encoding is denoted as $f(x)$, where $f$ is the function computed by the neural network.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grqa5qya8kj30m209a0ud.jpg" alt="image-20210621225748132" style="zoom:50%;" />

![image-20210621193709928](https://tva1.sinaimg.cn/large/008i3skNgy1grq4czbrjuj31360lm1ay.jpg)

Training will use triplets of images $(A, P, N)$:

- A is an "Anchor" image--a picture of a person.
- P is a "Positive" image--a picture of the same person as the Anchor image.
- N is a "Negative" image--a picture of a different person than the Anchor image.

These triplets are picked from the training dataset. $(A^{(i)}, P^{(i)}, N^{(i)})$ is used here to denote the $i$-th training example.

You'd like to make sure that an image $A^{(i)}$ of an individual is closer to the Positive $P^{(i)}$ than to the Negative image $N^{(i)}$ by at least a margin $\alpha$:

$$
|| f\left(A^{(i)}\right)-f\left(P^{(i)}\right)||_{2}^{2}+\alpha<|| f\left(A^{(i)}\right)-f\left(N^{(i)}\right)||_{2}^{2}
$$


You would thus like to minimize the following "triplet cost":

$$
\mathcal{J} = \sum^{m}_{i=1} \large[ \small \underbrace{\mid \mid f(A^{(i)}) - f(P^{(i)}) \mid \mid_2^2}_\text{(1)} - \underbrace{\mid \mid f(A^{(i)}) - f(N^{(i)}) \mid \mid_2^2}_\text{(2)} + \alpha \large ] \small_+
$$
Here, the notation "$[z]_+$" is used to denote $max(z,0)$.

**Notes**:

- The term (1) is the squared distance between the anchor "A" and the positive "P" for a given triplet; you want this to be small.
- The term (2) is the squared distance between the anchor "A" and the negative "N" for a given triplet, you want this to be relatively large. It has a minus sign preceding it because minimizing the negative of the term is the same as maximizing that term.
- $\alpha$ is called the margin. It's a hyperparameter that you pick manually. e.g. $\alpha = 0.2$.

Training set: 10k pictures of 1k persons -- Take 10k pictures and use it to generate the treplets $(A, P, N)$.

#### **Choosing the triplets $A,P,N$**

During training, if $A,P,N$ are chosen **randomly**, $d(A,P)+\alpha≤d(A,N)$ is **easily satisfied**.

Choose triplets that are "hard" to train on.
$$
d(A,P)\approx d(A,N)
$$

#### Face Verification and Binary Classification

![image-20210621200039246](https://tva1.sinaimg.cn/large/008i3skNgy1grq51f5hr4j313e0luatw.jpg)

**What you should remember**:

- Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem.
  
- Triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.
  
- The same encoding can be used for verification and recognition. Measuring distances between two images' encodings allows you to determine whether they are pictures of the same person.

### Neural Style Transfer

![image-20210621200449344](https://tva1.sinaimg.cn/large/008i3skNgy1grq55rdzg2j313e0k6qrn.jpg)

#### Cost function

$$
J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)
$$

Where $J_{content}(C,G)$ measures how similar is the content of the generated image $G$, to the content fo the content image $C$, $J_{style}(C,G)$ measures how similar is the style of the generated image $G$, to the style fo the style image $S$.

**Find the generated image $G$**

1. Initialize $G$ randomly.

$$
G:100\times100\times3
$$

2. Use gradient descent to minimize $J(G)$.

$$
G := G - \frac{\partial}{\partial G}J(G)
$$

#### Content cost function

Recap the cost function:
$$
J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)
$$

- Say you use hidden layer $l$ to compute content cost.
- Use pre-trained ConvNet.
- Let $a^{[l](C)}$ and $a^{[l](G)}$ be the activation of layer $l$ one the images.
- If $a^{[l](C)}$ and $a^{[l](G)}$ are similar, both images have similar content.

$$
J_{content}(C,G)=\frac{1}{2}\|a^{[l](C)} - a^{[l](G)} \|^2
$$

#### Style cost function

Define style as **correlation** between activations across channels.

##### Style matrix (Gram matrix)

* The style matrix is also called a "Gram matrix" .
* In linear algebra, the Gram matrix G of a set of vectors $(v_{1},\dots ,v_{n})$ is the matrix of dot products, whose entries are ${\displaystyle G_{ij} = v_{i}^T v_{j} = np.dot(v_{i}, v_{j})  }$. 
* In other words, $G_{ij}$ compares how similar $v_i$ is to $v_j$: If they are highly similar, you would expect them to have a large dot product, and thus for $G_{ij}$ to be large. 

Let $a^{[l]}_{i,j,k}=\text{activation at }(i,j,k)$. $G^{[l]}$ is $n_c^{[l]}\times n_c^{[l]}$. Particularly, $G^{[l]}_{kk'}$ will measure how correlated are the activations in channel $k$ compared to the activations in channel $k'$.
$$
G^{[l](S)}_{kk'}=\sum_i^{n_H^{[l]}} \sum_j^{n_W^{[l]}} a^{[l](S)}_{i,j,k} a^{[l](S)}_{i,j,k'}\\
G^{[l](G)}_{kk'}=\sum_i^{n_H^{[l]}} \sum_j^{n_W^{[l]}} a^{[l](G)}_{i,j,k} a^{[l](G)}_{i,j,k'}
$$

$$
\begin{align}
J_{style}^{[l]}(S,G) &= \frac{1}{(2n_H^{[l]}n_W^{[l]}n_C^{[l]})^2}\|G^{[l](S)}-G^{[l](G)}\|^2_F\\
                     &= \frac{1}{(2n_H^{[l]}n_W^{[l]}n_C^{[l]})^2}\sum_k \sum_{k'}(G^{[l](S)}_{kk'}-G^{[l](G)}_{kk'})^2
\end{align}
$$


$$
J_{style}(S,G)=\sum_{l}\lambda^{[l]}J_{style}^{[l]}(S,G)
$$

##### Compute Gram matrix $G_{gram}$
You will compute the Style matrix by multiplying the "unrolled" filter matrix with its transpose:

![image-20210622222715919](https://tva1.sinaimg.cn/large/008i3skNgy1grrewbyhn6j31fi0ho4qp.jpg)

###### $G_{(gram)ij}$: correlation
The result is a matrix of dimension $(n_C,n_C)$ where $n_C$ is the number of filters (channels). The value $G_{(gram)i,j}$ measures how similar the activations of filter $i$ are to the activations of filter $j$. 

###### $G_{(gram),ii}$: prevalence of patterns or textures
* The diagonal elements $G_{(gram)ii}$ measure how "active" a filter $i$ is. 
* For example, suppose filter $i$ is detecting vertical textures in the image. Then $G_{(gram)ii}$ measures how common  vertical textures are in the image as a whole.
* If $G_{(gram)ii}$ is large, this means that the image has a lot of vertical texture. 

By capturing the prevalence of different types of features ($G_{(gram)ii}$), as well as how much different features occur together ($G_{(gram)ij}$), the Style matrix $G_{gram}$ measures the style of an image. 

> Neural style transfer is trained as a supervised learning task in which the goal is to input two images (x*x*), and train a network to output a new, synthesized image (y*y*). 
>
> `FALSE:` **Neural style transfer is about training on the pixels of an image to make it look artistic, it is `not learning` any parameters**.

### 1D and 3D Generalizations

![image-20210621214852496](https://tva1.sinaimg.cn/large/008i3skNgy1grq860sx0hj31300loto2.jpg)

![image-20210621215256129](https://tva1.sinaimg.cn/large/008i3skNgy1grq8a8yqcxj312g0j07bx.jpg)

## Sequence Model

![image-20210623011329150](https://tva1.sinaimg.cn/large/008i3skNgy1grrjp84c0rj61380lmh1z02.jpg)

### Notation

$x^{(i)<t>}$: the $t^{th}$ element in the sequence of training example $i$.

$T^{(i)}_x$: the input sequence length for training example $i$.

$T^{(i)}_y$: the output sequence length for training example $i$.

![image-20210623210654210](https://tva1.sinaimg.cn/large/008i3skNgy1grsi71eetqj313i0ls7hi.jpg)

### Recurrent Neural Network Model

![image-20210623212106282](https://tva1.sinaimg.cn/large/008i3skNgy1grsilqtpw1j31220eck3n.jpg)

When making the prediction for $y_3$, it gets the information **not only from** $x_3$ **but also the information from $x_1$ and $x_2$.**

One weakness of RNN is that it only uses the information that is **earlier** in the sequence to male prediction.

#### Forward Propagation

![image-20210623213218643](https://tva1.sinaimg.cn/large/008i3skNgy1grsixeaor2j30xg0digsc.jpg)

Initialize $a^{<0>}=\vec{0}$.

Then,
$$
a^{<1>}=g(W_{aa}a^{<0>}+W_{ax}x^{<1>}+b_a) \leftarrow \tanh(/\text{ReLU})\\
\hat y^{<1>}=g(W_{ya}a^{<1>}+b_y)\leftarrow \text{depends on tasks}
$$
Generally,
$$
a^{<t>}=g(W_{aa}a^{<t-1>}+W_{ax}x^{<t>}+b_a)\\
\hat y^{<t>}=g(W_{ya}a^{<t>}+b_y)
$$

#### Simplified RNN notation

$$
\large a^{<t>}=g(W_{a}[a^{<t-1>},x^{<t>}]+b_a)\\
\large \hat y^{<t>}=g(W_{y}a^{<t>}+b_y)
$$

where
$$
W_a=\begin{bmatrix}
W_{aa}|W_{ax}
\end{bmatrix}\\
[a^{<t-1>},x^{<t>}]=\begin{bmatrix}
a^{<t-1>} \\ x^{<t>}
\end{bmatrix}
$$
So that,
$$
\begin{bmatrix}
W_{aa}|W_{ax}
\end{bmatrix}
\begin{bmatrix}
a^{<t-1>} \\ x^{<t>}
\end{bmatrix}
=W_{aa}a^{<t-1>}+W_{ax}x^{<t>}
$$

### Backpropagation Through Time

$$
L^{<t>}(\hat y^{<t>},y^{<t>})=-y^{<t>}\log {\hat y^{<t>}-(1-y^{<y>})\log (1-\hat y^{<t>})}\\
L(\hat y, y)=\sum_{t=1}^{T_y}L^{<t>}(\hat y^{<t>},y^{<t>})
$$

### Different Types of RNNs

![image-20210623220642100](https://tva1.sinaimg.cn/large/008i3skNgy1grsjx6urd0j31360lkdpl.jpg)

### Language Model and Sequence Generation

![image-20210623221333133](https://tva1.sinaimg.cn/large/008i3skNgy1grsk4bdw71j30v80i0wmb.jpg)

`The probability of sentence:` the chance that the next sentence you read somewhere out there in the world will be the particular sentence.

The language model represent a sentences as `outputs y` rather than `inputs x`.

The **basic job** of a language model is to estimate the probability of the particular input sequences of words. 

![image-20210623221746427](https://tva1.sinaimg.cn/large/008i3skNgy1grsk8plf4aj31320lids5.jpg)

`Corpus:` an NLP terminology that just means a large body or a large set of sentences.

`Tokenize:` form a vocabulary and map each of words in the sentence to **one-hot vectors** or **indicies** in the vocabulary. By tokenization step, you can decide whether or not the **punctuation** should be token as well.

`<EOS>:`"End Of Sentence", an **optional** extra token that can help to figure out when a sentence ends.

`<UNK>:` a unique token stands for **unknown** words.

![image-20210623222532205](https://tva1.sinaimg.cn/large/008i3skNgy1grskgs7u9yj313a0luk8t.jpg)

Each step in the RNN will look at some set of preceding words, such as given the first three words, what is the distribution over the next word? So this learns to predict **one word at a time** going from left to right.

The loss function:
$$
L(\hat y^{<t>},y^{<t>})=-\sum y_i^{<t>}\log \hat y^{<t>}_i \leftarrow \text{softmax loss}\\
L=\sum_t L^{<t>}(\hat y^{<t>},y^{<t>})
$$
Given a new sentence $(y^{<1>},y^{<2>},y^{<3>})$, the chance of this entire sentence would be:
$$
P(y^{<1>},y^{<2>},y^{<3>})=P(y^{<1>})P(y^{<2>}|y^{<1>})P(y^{<3>}|y^{<2>}，y^{<1>})
$$

### Sampling Novel Sequences

![image-20210623223527062](https://tva1.sinaimg.cn/large/008i3skNgy1grskr3qqy1j31360loaoq.jpg)

**what you want to do:**

1. Randomly sample first word $\hat y^{<1>}$ according to the first softmax distribution.

```python
np.random.choice() # To sample according to distribution
```

2. Going to next time step, take the $\hat y^{<1>}$ which is just sampled as the input to the next time step and sample $\hat y^{<2>}$.
3. Repeat 1 $\to$ 2 until get the last time step (such as when `<EOS>` token is sampled or just decide how many words to sample).

![image-20210623223809791](https://tva1.sinaimg.cn/large/008i3skNgy1grsktxb0w7j31300lwgvd.jpg)

**Pros and cons of Character-level language model**

1. Don't have to worry about unknown word tokens.
2. End up with much longer sequences (computational expensive).

### Vanishing Gradients with RNNs

![image-20210623225444375](https://tva1.sinaimg.cn/large/008i3skNgy1grslb6gge1j312y0lwarf.jpg)

Because of vanishing gradients, the basic RNN model has many **local influences**, meaning that the output $y^{<t>}$ is mainly influenced by values close to $y^{<t>}$, so is difficult for it to be strongly influenced by an input that was **very early** in the sequence.

### Gated Recurrent Unit (GRU)

![image-20210623225756573](https://tva1.sinaimg.cn/large/008i3skNgy1grslei1q87j30zq0hejw5.jpg)

![image-20210624023518376](https://tva1.sinaimg.cn/large/008i3skNgy1grsroo5yzgj313m0m2e2h.jpg)

`c:memory cell` will provide a bit of to remember the important information.

The memory cell will have some value $c^{<t>}$ at time $t$. The GRU will out put an activation value $a^{<t>}$ that is equal to $c^{<t>}$.
$$
c^{<t>}=a^{<t>}
$$
At every time step, we are going to **consider** overwritting the memory cell with a `candidate` $\tilde c^{<t>}$ for replacing $c^{<t>}$.
$$
\tilde c^{<t>}=\tanh (W_c[c^{<t-1>},x^{t}]+b_c)
$$
The key idea of GRU is to have a `update gate`
$$
\Gamma_u = \sigma(W_u[c^{<t-1>},x^{t}]+b_u)
$$
The job of gate $\Gamma_u$ is to decide when do you update memory cell.

The specific equation for GRU is
$$
c^{<t>}=\Gamma_u * \tilde c^{<t>} + (1-\Gamma_u)*c^{<t-1>}
$$
Because the gate is easliy to be set to 0, it is cery good at **maintaining** the value for the cell. And because the gate can be so close to 0, it does not suffer from vanishing gradient problems.

#### **Full GRU**

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1grsrvgx8jaj30wm0juq8e.jpg" alt="image-20210624024150843" style="zoom: 50%;" />

For full GRU, there is an additional gate $\Gamma_r$.
$$
\begin{align}
\tilde c^{<t>}&=\tanh (W_c[\Gamma_r * c^{<t-1>},x^{t}]+b_c)\\
\Gamma_r &= \sigma(W_r[c^{<t-1>},x^{t}]+b_r)\\
\Gamma_u &= \sigma(W_u[c^{<t-1>},x^{t}]+b_u)\\
c^{<t>}&=\Gamma_u * \tilde c^{<t>} + (1-\Gamma_u)*c^{<t-1>}
\end{align}
$$
