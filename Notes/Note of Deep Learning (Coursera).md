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

