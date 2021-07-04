$$
\large e_{man}-e_{woman}\approx e_{king}-e_{w}
$$

Find word $w$:
$$
\arg\max_w sim(e_w,e_{king}-e_{man}+e_{woman})
$$

#### Cosine similarity

$$
sim(u,v)=\frac{u\cdot v}{\|u\|_2\|v\|_2}=\cos(\theta)
$$

* $u \cdot v$ is the dot product (or inner product) of two vectors
* $||u||_2$ is the norm (or length) of the vector $u$
* $\theta$ is the angle between $u$ and $v$. 
* The cosine similarity depends on the angle between $u$ and $v$. 
    * If $u$ and $v$ are very similar, their cosine similarity will be close to 1.
    * If they are dissimilar, the cosine similarity will take a smaller value.

![image-20210630150212191](https://tva1.sinaimg.cn/large/008i3skNgy1gs0azp1lvmj30xy0e6adi.jpg)

### Embedding Matrix

$$
E=
\begin{bmatrix}
a & aaron & \cdots & orange & \cdots & zulu & \lang UNK\rang \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots &\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots &
\end{bmatrix} \in \R^{300\times10,000}
$$

The word orange is the 6257th token in the vocabulary:
$$
orange=O_{6257}=
\begin{bmatrix}
0\\\vdots\\\vdots\\1\\\vdots\\0
\end{bmatrix}\in\R^{10,000}
$$
$O_{6257}$ is the one-hot vector of the word orange. Notice that:
$$
E\ \cdot\ O_{6257}=e_{6257}=
\begin{bmatrix}
\vdots\\\vdots\\\vdots\\
\end{bmatrix}\in\R^{300}
$$

### Learning Word Embeddings

![image-20210629213842000](https://tva1.sinaimg.cn/large/008i3skNgy1grzgtxzafqj313a0mear7.jpg)

![image-20210629214149351](https://tva1.sinaimg.cn/large/008i3skNgy1grzgx5f4wej312w0jen8q.jpg)

If you really want to **build a language model**, it is nature to use **last few words** as the context. But if the main goal is to **learn word embedding**, then you can use **all of these other context**, and they will result in meaningful word embedding as well.

### Word2Vec

#### Skip-grams

1. Randomly pick a word to be the context word.
2. Randomly pick another word within some window (Say $\pm5$ words) of the context word.

![image-20210629214951051](https://tva1.sinaimg.cn/large/008i3skNgy1grzh5hzubwj30wy0hgtdv.jpg)

![image-20210629215458810](https://tva1.sinaimg.cn/large/008i3skNgy1grzhatw045j311k0jcgw9.jpg)

$\theta_t$ is the parameter associated with output $t$. It is the chance of the particular word $t$ being the label.

However, there are a couple problems with using skip-grams.

The primary problem is **computational speed**.
$$
\large p(t|c)=\frac{e^{\theta^T_te_c}}{\sum_{j=1}^{10,000}e^{\theta_j^Te_c}}
$$
To evaluate this probability, you need to carry out a sum over all 10,000 words in the vocabulary. One of the solution is to use **Hierachical Softmax Classifier**. In practice, the hierachical softmax classifer can be developed so the **common words** tend to be on **top**, whereas the less common words can be buried much deeper in the tree.

![image-20210629220325915](https://tva1.sinaimg.cn/large/008i3skNgy1grzhjmuqv0j30ls0aqjuu.jpg)

**How to sample the context c?**

In practice the distribution of words pc isn't taken just entirely uniformly at random for the training set purpose, but instead there are different heuristics that you could use in order to balance out something from the common words together with the less common words.

### Negative sampling

To solve the computational speed problems, this algorithm defines a new learning problem.

![image-20210629231821354](https://tva1.sinaimg.cn/large/008i3skNgy1grzjpm7g9zj310y0kgdsj.jpg)

![image-20210629232143947](https://tva1.sinaimg.cn/large/008i3skNgy1grzjt456a3j31300lwk2e.jpg)

Think of this as having 10,000 binary logistic regression classifers, but instead of training all 10,000 of them on every iterarion, **we are going to train `k+1` of them (`1` coresponding to actual target word, and `k` randomly chosen negative examples)**.

#### Selecting negative examples

(Empirically) Sample according to:
$$
p(w_i) = \frac{f(w_i)^{3/4}}{\sum_{j=1}^{10,000}f(w_j)^{3/4}}
$$
where $f(w_i)$ is the observed **frequency** of a particular word in the English language or in the training set corpus.

### GloVe Word Vectors

#### GloVe (global vectors for word representation)

![Wb1_QMhfEemRPArJHNevzg_678507a1c83ea1d6ad8dcf1b481a705a_Glove](https://tva1.sinaimg.cn/large/008i3skNgy1grzky7l5obj31hg0u0h0l.jpg)

For the purposes of the GloVe algorithm, we define context and target as whether or not the two words appear in close proximity.

![image-20210629234927980](https://tva1.sinaimg.cn/large/008i3skNgy1grzklz9l4ij311y0kck04.jpg)
$$
minimize\ \sum^{10,000}_{i=1}\sum^{10,000}_{j=1}f(X_{ij})(\theta^T_ie_j+b_i+b_j'-\log X_{ij})^2
$$
![image-20210629235736440](https://tva1.sinaimg.cn/large/008i3skNgy1grzkufjnwzj31360ls4d6.jpg)

You can't guarantee that the axis used to represent the features will be well-aligned with waht might be easily humanly interpretable axis.

### Sentiment Classification

Sentiment classification is the task of looking at a piece of text and telling if someone likes or dislikes the thing they are talking about.

![image-20210630000437606](https://tva1.sinaimg.cn/large/008i3skNgy1grzl1q8tgaj310s0ik7et.jpg)

One of the challenges of sentiment classification is you might not have a huge label data set.

![image-20210630001459204](https://tva1.sinaimg.cn/large/008i3skNgy1grzlcid1qpj312y0loqfu.jpg)

However, if you use an algorithm like this that ignores word **order** and just sums or averages all of the embeddings for the different words, the result is not convicing for shown example (because there are many "good" in the review, and the alogorithm may think this is a good review after averaging the word embedding). So, instead of just summing all of the word embeddings, you can instead use a **RNN** for sentiment classification.

![image-20210630002314035](https://tva1.sinaimg.cn/large/008i3skNgy1grzll3iddyj31420l0wms.jpg)

### Debiasing Word Embeddings

![image-20210630002720294](https://tva1.sinaimg.cn/large/008i3skNgy1grzlpd03ydj314i0mk17b.jpg)

![image-20210630003528634](https://tva1.sinaimg.cn/large/008i3skNgy1grzlxtv7f8j31400m4tpt.jpg)

#### Neutralization

The figure below should help you visualize what neutralizing does. If you're using a 50-dimensional word embedding, the 50 dimensional space can be split into two parts: The bias-direction $g$, and the remaining 49 dimensions, which is called $g_{\perp}$ here. In linear algebra, we say that the 49-dimensional $g_{\perp}$ is perpendicular (or "orthogonal") to $g$, meaning it is at 90 degrees to $g$. The neutralization step takes a vector such as $e_{receptionist}$ and zeros out the component in the direction of $g$, giving us $e_{receptionist}^{debiased}$. 

Even though $g_{\perp}$ is 49-dimensional, given the limitations of what you can draw on a 2D screen, it's illustrated using a 1-dimensional axis below.

![image-20210630153510740](https://tva1.sinaimg.cn/large/008i3skNgy1gs0bxytn2yj30y60igq6d.jpg)

Given an input embedding $e$, you can use the following formulas to compute $e^{debiased}$: 

$$
e^{bias\_component} = \frac{e \cdot g}{||g||_2} * \frac{g}{||g||_2} =\frac{e \cdot g}{||g||_2^2} * g\\
e^{debiased} = e - e^{bias\_component}
$$
where $e^{bias\_component}$ is the projection of $e$ onto the direction $g$.

#### Equalization

The key idea behind equalization is to make sure that a particular pair of words are equidistant from the 49-dimensional $g_\perp$. The equalization step also ensures that the two equalized steps are now the same distance from $e_{receptionist}^{debiased}$, or from any other work that has been neutralized. Visually, this is how equalization works:

![image-20210630154856143](https://tva1.sinaimg.cn/large/008i3skNgy1gs0cc9vzooj30wa0kygp2.jpg)

The key equations:
$$
\mu = \frac{e_{w1} + e_{w2}}{2}\\
\mu_{B} = \frac {\mu \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}\\
\mu_{\perp} = \mu - \mu_{B}\\
e_{w1B} = \frac {e_{w1} \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}\\
e_{w2B} = \frac {e_{w2} \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}\\
e_{w1B}^{corrected} = \sqrt{ |{1 - ||\mu_{\perp} ||^2_2} |} * \frac{e_{\text{w1B}} - \mu_B} {||(e_{w1} - \mu_{\perp}) - \mu_B||_2}\\
e_{w2B}^{corrected} = \sqrt{ |{1 - ||\mu_{\perp} ||^2_2} |} * \frac{e_{\text{w2B}} - \mu_B} {||(e_{w2} - \mu_{\perp}) - \mu_B||_2}\\
e_1 = e_{w1B}^{corrected} + \mu_{\perp}\\
e_2 = e_{w2B}^{corrected} + \mu_{\perp} 
$$
![IMG_0325](https://tva1.sinaimg.cn/large/008i3skNgy1gs0evnirhjj30u00xg4aw.jpg)

### Sequence to Sequence Model

#### Basic models

![image-20210702144231848](https://tva1.sinaimg.cn/large/008i3skNgy1gs2lnvmnslj30wc0jstk9.jpg)

An **encoding** network and a **decoding** network.

![image-20210702144739769](https://tva1.sinaimg.cn/large/008i3skNgy1gs2lt6jma3j316k0jcqjz.jpg)

Of the key differences is that you do not want to randomly choose (like we did in generating novel text) in translation. You may be want most likely translation.

![image-20210702150206047](https://tva1.sinaimg.cn/large/008i3skNgy1gs2m860avtj317e0oe7na.jpg)

The translation model can be viewed as a **conditional language model**.

![image-20210702150430345](https://tva1.sinaimg.cn/large/008i3skNgy1gs2mao9p1gj316y0medtw.jpg)

Instead of finding the probability distribution $P(y^{\lang1\rang},\cdots,y^{\lang T_y \rang}|x)$, what you would like is to find the English sentence $y$ that **maximizes** that conditional probability.
$$
\mathop{\arg\max}_{y^{\lang1\rang},\cdots,y^{\lang T_y \rang}}\ P(y^{\lang1\rang},\cdots,y^{\lang T_y \rang}|x)
$$
The most common algorithm for doing this is called **beam search**.

#### Beam search

##### **Why not greedy search?**

**Greedy search** is to pick the most likely words every time.

![image-20210702151827019](https://tva1.sinaimg.cn/large/008i3skNgy1gs2mp6qdftj314o0msgy7.jpg)

The greedy search may ends up resulting in a **less optimal**, or more **verbose** sentence. Moreover, it is not always optimal to just pick one word at a time. If our vocabulary contains 10,000 words, there will be $10,000^{10}$ possible sentences that are 10 words long in total, it is impossible to rate them all. So, the most common thing to do is use **an approximate search algorithm**. The approximate search algorithm will try to pick the sentence $y$ that maximizes the conditional probability.

##### Beam search algorithm

The beam search algorithm has a parameter called **beam width** deboted by $B$. Instead of picking only the one most likely words like gready search dose, beam search can consider **multiple alternatives** accoding to beam width.

![image-20210702153310304](https://tva1.sinaimg.cn/large/008i3skNgy1gs2n4i44y1j313c0jo7dl.jpg)

![image-20210702154416193](https://tva1.sinaimg.cn/large/008i3skNgy1gs2ng63emvj315e0n44qp.jpg)
$$
P(A,B|C)=\frac{P(A,B,C)}{P(C)}=\frac{P(A,B,C)}{P(B,C)}\frac{P(B,C)}{P(C)}=P(A|B,C)P(B|C)
$$
![image-20210702160221187](https://tva1.sinaimg.cn/large/008i3skNgy1gs2nywipsvj314y0me1b2.jpg)

Notice that if the beam width is set to be equal to one, then the beam search becomes the greedy search algorithm.

#### Refinements to beam search

Recall that the objective is to
$$
\mathop{\arg\max}_{y^{\lang1\rang},\cdots,y^{\lang T_y \rang}}\ P(y^{\lang1\rang},\cdots,y^{\lang T_y \rang}|x)
$$
Notice that
$$
P(y^{\lang1\rang},\cdots,y^{\lang T_y \rang}|x)=P(y^{\lang 1\rang}|x)P(y^{\lang 2\rang}|x,y^{\lang 1\rang})\cdots P(y^{\lang T_y\rang}|x,y^{\lang 1\rang},\cdots,y^{\lang T_y-1\rang})
$$
So, the objective function is
$$
\mathop{\arg\max}_y\prod^{T_y}_{t=1}P(y^{\lang t\rang}|x,y^{\lang 1\rang},\cdots,y^{\lang t-1\rang})
$$
Notice that in the objective function, these probabilities are all mumbers less than 1, and multiplying a lot of them will result in a very tiny number, which can result in **numerical underflow**, meaning that it is too small for the floating part represenation in your computer to store accurately.

So, inpractice, we will take logs and the objective funtion becomes
$$
\mathop{\arg\max}_y\sum^{T_y}_{t=1}\log P(y^{\lang t\rang}|x,y^{\lang 1\rang},\cdots,y^{\lang t-1\rang})
$$

##### Length normalization

For both of these objective functions, however, if you have a very **long** sentence, the probability of the sentence will be very **low** (multiply lots of number smaller than 1, or sum over many negative numbers). So these objective function have an undesirable affect that they tend to prefer **short** tanslations.

To make the algorithm works better, we need to add one more term to normalize it:
$$
\mathop{\arg\max}_y \frac{1}{T_y}\sum^{T_y}_{t=1}\log P(y^{\lang t\rang}|x,y^{\lang 1\rang},\cdots,y^{\lang t-1\rang})
$$
This will significantly reduce the penalty for outputting longer translations.

In practice, instead of dividing by $T_y$, sometimes we will use a softer approach
$$
\mathop{\arg\max}_y \frac{1}{T_y^\alpha}\sum^{T_y}_{t=1}\log P(y^{\lang t\rang}|x,y^{\lang 1\rang},\cdots,y^{\lang t-1\rang})
$$
where normally $\alpha=0.7$.

##### Beam search discussion

**How to choose beam width $\mathbf{B}?$**

- Large $B$: better result, but slower
- Small $B$: worse result, but faster

Unlike **exact search algorithms** like BFS or DFS, Beam Search **runs faster** but is **not guaranteed** to find exact maximum for $\mathop{\arg\max}\limits_yP(y|x)$.

#### Error analysis in beam search

![image-20210702192042585](https://tva1.sinaimg.cn/large/008i3skNgy1gs2tp9fao5j313i0kqal8.jpg)

What you can do is to compute $P(y^*|x)$ and $P(\hat y|x)$ and see which one is bigger.

![image-20210702192445258](https://tva1.sinaimg.cn/large/008i3skNgy1gs2tth8psnj314s0ly4fl.jpg)

Only when a lot of fractions of faults are due to beam search, then you may need to increase the beam width.

#### Bleu score

What the Bleu score does is given a machine generated translation, it allows you to automatically compute a score that measures how good is that machine translation.

**Bleu** stands for **bilingual evaluation understudy**.

![image-20210702193818127](https://tva1.sinaimg.cn/large/008i3skNgy1gs2u7k46c9j31520mwaqa.jpg)

![image-20210702194902763](https://tva1.sinaimg.cn/large/008i3skNgy1gs2uir5xvzj315g0mqds7.jpg)

For unigram (one word), the precision $P_1$ (1 stands for unigram) is given by:
$$
P_1=\frac{\sum\limits_{\text{unigram}\in\hat y}\text{Count}_{\text{clip}}(\text{unigram})}{\sum\limits_{\text{unigram}\in\hat y}\text{Count}(\text{unigram})}
$$
For n-gram (n words pair), the precision $P_n$ is given by:
$$
P_n=\frac{\sum\limits_{\text{n-grams}\in\hat y}\text{Count}_{\text{clip}}(\text{n-grams})}{\sum\limits_{\text{n-grams}\in\hat y}\text{Count}(\text{n-grams})}
$$
![img](https://tva1.sinaimg.cn/large/008i3skNgy1gs2v53i8vlj31hh0u0k12.jpg)

BP stands for **brevity penalty**.

Today, BLEU score is used to evaluate many systems that **generate text**, such as machine translation systems, as well as image captioning systems where you would have a neural network generated image caption. It is a useful single number evaluation metric to use when you want your algorithm to generate text, and you want to see whether it has similar meaning as the reference text written by human.

#### Attention model

**The problem of long sequences**

It is difficult for a network to memorize long sentence.

![image-20210702222120783](https://tva1.sinaimg.cn/large/008i3skNgy1gs2yx6swkqj30v808e0v6.jpg)

![image-20210702224957661](https://tva1.sinaimg.cn/large/008i3skNgy1gs2zqznzl0j31580n81gi.jpg)

We use
$$
a^{\lang t' \rang}= \begin{bmatrix}
\overrightarrow{a}^{\lang t' \rang}\\\overleftarrow{a}^{\lang t' \rang}
\end{bmatrix}
$$
to represent feature vector for time step $t'$. For gennerating the translation, we use a single direction RNN with state $s$. This single RNN will generate the translation $y$. It also has a input called **context** $c$ for every time step.
$$
c^{\lang t \rang}=\sum_{t'}\alpha^{\lang t,t'\rang}a^{\lang t' \rang}
$$
where $\alpha$ is the **attention parameter**.
$$
\large \alpha^{\lang t,t'\rang}=\text{amount of attention}\ y^{\lang t \rang}\ \text{should pay to }a^{\lang t' \rang}
$$
The formula to compute $\alpha^{\lang t,t'\rang}$ is
$$
\alpha^{\lang t,t'\rang}=\frac{\exp(e^{\lang t,t'\rang})}{\sum^{T_x}_{t'=1}\exp(e^{\lang t,t'\rang})}
$$
Using the softmax prioritization is just to ensure
$$
\sum_{t'}\alpha^{\lang t,t'\rang}=1
$$
$e$ is called the "**energies**" variable, one way to compute the factors $e$ is to use a small neural networs (usually just one hidden layer) with **softmax** activation as follow:

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gs2zzr9ht6j30ds05sdge.jpg" alt="image-20210702225823394" style="zoom:50%;" />

The intuition is, if you want to decide how much attention to pay to the $\large a^{\lang t'\rang}$, it should depend the most on is what is your own hidden state activation from the previous time step, which is $\large s^{\lang t-1\rang}$, and also the features of original words, which is $\large a^{\lang t'\rang}$. The following picture shows how the context is generated.

![attn_mechanism](https://tva1.sinaimg.cn/large/008i3skNgy1gs3uo7s4a9j30wi0u0aeg.jpg)

The whole attention model is as follow:

![attn_model](https://tva1.sinaimg.cn/large/008i3skNgy1gs3uso2za7j30xx0u0juj.jpg)

Here are some properties of the model that you may notice: 

**Pre-attention and Post-attention LSTMs on both sides of the attention mechanism**

- There are two separate LSTMs in this model (see diagram on the left): pre-attention and post-attention LSTMs.
- *Pre-attention* Bi-LSTM
    - The pre-attention Bi-LSTM goes through $T_x$ time steps
- *Post-attention* LSTM 
    - The post-attention LSTM goes through $T_y$ time steps. 
- The post-attention LSTM passes the hidden state $s^{\langle t \rangle}$ and cell state $c^{\langle t \rangle}$ from one time step to the next. 

One **downside** to this algorithm is that it does take **quadratic time** or **quadratic cost** to run this algorithm. Because the total number of attention parameters is $T_x\times T_y$.

### Speech Recognition

![image-20210703132524324](https://tva1.sinaimg.cn/large/008i3skNgy1gs3p4ug9k6j31540moqdr.jpg)

The first figure plots the audio clip, the horizontal axis is time, and the vertical axis is actually air pressure. The common prepocessing step is to convert the audio clip in to a **spectrogram**. The horizontal axis of the spectrogram is time, and the vertical axis is the frequency, and the intensity of different color shows the amount of energy. It shows "how loud is the sound at different frequncy at different time".

**What really is an audio recording?** 

* A microphone records little variations in air pressure over time, and it is these little variations in air pressure that your ear also perceives as sound. 
* You can think of an audio recording as a long list of numbers measuring the little air pressure changes detected by the microphone. 
* We will use audio sampled at 44100 Hz (or 44100 Hertz). 
    * This means the microphone gives us 44,100 numbers per second. 
    * Thus, a 10 second audio clip is represented by 441,000 numbers (= $10 \times 44,100$). 

**Spectrogram**

* It is quite difficult to figure out from this "raw" representation of audio whether the word "activate" was said. 
* In  order to help your sequence model more easily learn to detect trigger words, we will compute a *spectrogram* of the audio. 
* The spectrogram tells us how much different frequencies are present in an audio clip at any moment in time. 
* If you've ever taken an advanced class on signal processing or on Fourier transforms:
    * A spectrogram is computed by sliding a window over the raw audio signal, and calculating the most active frequencies in each window using a Fourier transform. 

#### CTC cost for speech recognition

CTC stands for **Conectionist Temporal Classification**.

**Basic rule**: collapse repeated characters not separated by "blank" (not space).

![image-20210703150357840](https://tva1.sinaimg.cn/large/008i3skNgy1gs3rwga7wmj31560migyz.jpg)

#### Trigger word detection

![image-20210703161951620](https://tva1.sinaimg.cn/large/008i3skNgy1gs3u3eycx7j31560mkwrk.jpg)

### Transformers

#### Transformer network motivation

As we move from RNNs to GRU to LSTM, the models became more and more complex. And all of these models are still **sequential models** in that they ingested the input, maybe one word or one token at a time. And so, each unit was like a **bottleneck** to the flow of information. Because to compute the output of the final unit, for example, you have to compute the outputs of all of the unit that come before.

![image-20210703232516542](https://tva1.sinaimg.cn/large/008i3skNgy1gs46e1av7gj31420f4dlk.jpg)

The transformer architecture allows you to run a lot of these computations for an entire sequence **in parallel**. The major innovation of the transformer architecture is combining the use of **attention based representations** and a **CNN style of processing**. The two key ideas we will go through are:

- Self-attention
- Multi-Head Attention

#### Self-attention

Self-attention enable us to use attention with a style more like CNNs. 

First, we need to compute an **attention-based representation**:
$$
A(q,K,V)=\text{attention-based vector representation of a word}
$$
We will compute this for each word to get $A^{\lang t \rang}$. The intuition of $A^{\lang t \rang}$ is that it will look at the surrounding words to try to figure out the context of this word and find the most appropriate representation for this word.

Recall the RNN attention formula:
$$
\alpha^{\lang t,t' \rang}=\frac{\exp(e^{\lang t,t' \rang})}{\sum_{t'=1}^{T_x}\exp(e^{\lang t,t' \rang})}
$$
The transformers attention formula is similar:
$$
A(q,K,V)=\sum_i\frac{\exp(q\cdot k^{\lang i \rang})}{\sum_j\exp(q\cdot k^{\lang j \rang})}v^{\lang i \rang}
$$
The main difference is that for each word you have three values called
$$
q^{\lang t \rang}:\text{query}\\
k^{\lang t \rang}:\text{key}\\
v^{\lang t \rang}:\text{value}
$$
![image-20210703235755608](https://tva1.sinaimg.cn/large/008i3skNgy1gs47c2j1rqj314m0m8dqy.jpg)

Now, let's go through how to get $A^{\lang t \rang}$

First, we will associate each of words with three values, the **query, key, and value** pairs. If the $x^{\lang t \rang}$ is the word embedding, the way that $q^{\lang t \rang}$ is computed is as a learned matrix:
$$
q^{\lang t \rang}=W^Q\cdot x^{\lang t \rang}
$$
And similarly for the key and value,
$$
k^{\lang t \rang}=W^K\cdot x^{\lang t \rang}
$$

$$
v^{\lang t \rang}=W^V\cdot x^{\lang t \rang}
$$

These matrices, $W^Q,W^K,W^V$, are parameters of this learning algorithm, and they allow you to pull off these query, key, and value vectors for each word.

![image-20210704015839249](https://tva1.sinaimg.cn/large/008i3skNgy1gs4atnisdoj31500mu7o8.jpg)

**One intuition behind the intent of these query, key, value vectors**

$q^{\lang t \rang}$ may represent a question like, what is happening to the word $x^{\lang t \rang} $ when computing $A^{\lang t \rang}$? In the above example, $q^{\lang 3 \rang}$ may represent what is happening to l'Afrique (Africa) when we compute $A^{\lang 3 \rang}$? Dose l'Afrique represent a destination of traveling or the second largest continent in this sentence?

Then, we are goint to computer the inner product between $q^{\lang t \rang}$ and $k^{\lang 1 \rang}$, and this will tell us how good is an answer word $x^{\lang 1 \rang}$ to the question what is happening to the word $x^{\lang t \rang} $? For example,  $q^{\lang 3 \rang}\cdot k^{\lang 2 \rang}$ will tell us how good is "visite" an answer to the question of what is happening to l'Afrique (Africa), and so on for the other words in the sequence. The goal of this operation is to **pull off the most imformation that is needed** to help us compute the most useful representation $A^{\lang t \rang}$.

Next, for the above example, you may find that $q^{\lang 3 \rang}\cdot k^{\lang 2 \rang}$ gives you largest value, this may suggest that "visite" gives you **the most relevant contexts** for the question of what is happening to l'Afrique (Africa) (view Africa as a destination for a visit).

In summary for intuition:

- $Q$: interesting questions about the words in a sentence.
- $K$: qualities of words given a $Q$.
- $V$: specific representations of words given a $Q$.

Then we will take softmax over these inner products, and multiply them with $v^{\lang t \rang}$, which is value for word $x^{\lang t \rang}$. Finally, we sum them up to get $A^{\lang t \rang}$.

**The key advantage** of this representation is that the word of $x^{\lang t \rang}$ is not some fixed word embedding, instead it lets the self-attention mechanism realize that this word is actually associated with another word (Africa is the destination of a visit), and thus computer a richer, more useful representation for this word.

The notation used in literature is like this:
$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

* $Q$ is the matrix of queries 
* $K$ is the matrix of keys
* $V$ is the matrix of values
* ${d_k}$ is the dimension of the keys, which is used to scale everything down so the softmax doesn't explode

It is the vetorized representation of
$$
A(q,K,V)=\sum_i\frac{\exp(q\cdot k^{\lang i \rang})}{\sum_j\exp(q\cdot k^{\lang j \rang})}v^{\lang i \rang}
$$
The term $\sqrt{d_k}$ in the denominator is just to scale the dot product so it doesn't explode. The another name for this type of attention is called **the scaled dot-product attention**.

#### Muli-head attention

Each time we calculate self attention for a sequence is called a **head**.

![image-20210704111946382](https://tva1.sinaimg.cn/large/008i3skNgy1gs4r1j2kzuj31520msx00.jpg)

To calculate the multiple self-attention, we firstly multiply $q^{\lang t \rang},k^{\lang t \rang},v^{\lang t \rang}$ with weight matrices $W_i^Q,W_i^K,W_i^V$, the subscript $i$ represents this is the weight for $head_i$. In literature, we use $h$ to denote the number of heads. Using $h=8$ heads is common in the literature.
$$
h=\#heads
$$
For the sake of intuition, we can think of $W_i^Q,W_i^K,W_i^V$ as being learned to help ask and answer different questions ($i^{th}$ question in particular). For example, $W_1^Q,W_1^K,W_1^V$ are learned to help with asking and answering "what is happenting to the word 'Africa'", and the word "visite" gives the best answer (blue arrow). Similarly, $W_2^Q,W_2^K,W_2^V$ are learned to help with asking and answering the second question like "when is happening", so the inner product between the September key and l'Afrique query will have the highest value (red arrow). And $W_3^Q,W_3^K,W_3^V$ may help with third question "who has something to do with Africa", and this time Jane's value will have the greatest weight in this representation.

After calculating the computation for these heads, the concatenation of these values is used to compute the output of the multi-head attention.
$$
MultiHead(Q,K,V)=concat(head_1,head_2,\cdots,head_n)W_0
$$
where
$$
head_i=Attention(W_1^QQ,W_1^KK,W_1^VV)
$$
Recall that
$$
Attention(Q,K,V)=softmax({QK^T\over\sqrt{d_k}})V
$$
In practice, these multi-head attentions are computed in parallel, because no one head's value depends on the value of any other head.

#### Transformer network

The first step in the transformer is the embeddings of the input sentence get def into an **encoder** block, and the $Q,K,V$ are computed from the embeddings and weight matrices $W$. 

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gs4wtl4oi8j30f80h0ta9.jpg" alt="image-20210704143947481" style="zoom:50%;" />

The multi-head attention layer then produces a matrix that can be passed into a feed forward neural network, which helps determine what interesting features there are in the sentence. In the original paper, this encoding block is repeated $N$ times, typically $N=6$.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gs4wugiudrj30fs0gs76d.jpg" alt="image-20210704144038840" style="zoom:50%;" />

Then the output of the encoder is fed into a **decoder** block. The job of the decoder is to output the English translation. At every step, the decoder block will input the first few words that were generated by decoder and compute $Q,K,V$ for the first multi-head attention block.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gs4yv2gl2kj30f00i6jsi.jpg" alt="image-20210704155025610" style="zoom:50%;" />

Then the first multi-head attention block will <u>generate</u> $Q$ matrix for next multi-head attention block, and the matrix $K,V$ are <u>derived</u> from the output of the encoder. The intuition for this step is that, the input of the first multi-head attention block is what you have translated of the sentence so far. And this will ask the query to say "what is the next word?", and then we will pull context from $K$ and $V$ which are translated from the French to then try to decide the next word.

![image-20210704155224185](https://tva1.sinaimg.cn/large/008i3skNgy1gs4yx8wc0dj312w0hy0x3.jpg)

The outputs of second multi-head attention block will feed to a feed forward neural network, and this decoder block is also going to be repeated $N$ times. And the job of the feed forward neural network is to predict the next word in the sentence.

![image-20210704155610933](https://tva1.sinaimg.cn/large/008i3skNgy1gs4z12usymj31560l4n2w.jpg)

And the generated output will be fed to the input as well and predict the next word.

![image-20210704155826322](https://tva1.sinaimg.cn/large/008i3skNgy1gs4z3gjyuxj315c0moqa4.jpg)

This is the **main idea** of transformer.

Moreover, there is **positional encoding (PE)** of the input. The position within the sentence can be extremely important to translation. However, when you train a Transformer network, you feed your data into the model all at once. While this dramatically reduces training time, there is no information about the order of your data. This is where positional encoding is useful - you can specifically encode the positions of your inputs and pass them into the network using these sine and cosine formulas:
$$
PE_{(pos,2i)}=\sin({pos\over 10000^{2i\over d}})\\
PE_{(pos,2i+1)}=\cos({pos\over 10000^{2i\over d}})
$$
Where 

- $d$ is the dimension of the word embbedings (how many features)
- $pos$ is the numerical position of the word ($pos=1$ for the word Jane). 
- $i$ refers to the different dimensions of the encoding
- the positon encoding for word $x^{\lang t\rang}$ is $p^{\lang t\rang}$.

![image-20210704161714856](https://tva1.sinaimg.cn/large/008i3skNgy1gs4zmz98ytj309005k0u3.jpg)

The $q^{\lang t\rang}$ is unique because the properties of sin and cos function:

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gs4zuc6qe3j30cm076n0d.jpg" alt="image-20210704162417662" style="zoom:67%;" />

The positional encoding $p^{\lang t\rang}$ is added directly to the $x^{\lang t\rang}$, so that each of the word vectors is also influenced by the position of the word. The values of the sine and cosine equations are small enough (between -1 and 1) that when you add the positional encoding to a word embedding, the word embedding is not significantly distorted. 

![image-20210704162725676](https://tva1.sinaimg.cn/large/008i3skNgy1gs4zxm18ycj314y0m8nda.jpg)

In addition to adding this positional encodings to the embeddings, or the attention representations, we would also pass them through the netword with **residual connections**. The add & norm layer is just like the batchnorm layer to speed up learning.

![image-20210704163127609](https://tva1.sinaimg.cn/large/008i3skNgy1gs501t65wpj30og0bywkr.jpg)

And these bachnorm-like layer and residual connections are repeated throughout the architecture. Finaly, for the output of the decoder block, there are a linear and a softmax layer to predict the next word one at a time.

![image-20210704163412917](https://tva1.sinaimg.cn/large/008i3skNgy1gs504opqawj31540matqk.jpg)

In literature on transformer, the first multi-head attention block is actually **masked multi-head attention**. This is important only for training process.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gs506s2gvdj30g20hsgqw.jpg" alt="image-20210704163616428" style="zoom:50%;" />

When training the network, you have the access to the entire correct outputs. So we do not need to generate the words one at a time during training. Instead, what masking does is it **blocks out** the last part of sentence to mimic what the netword will need to do at test time while doing prediction. In other words, all the masked multi-head attention do is to repeatedly pretend that the network had perfectly translated first few words, and hide the remaining words to see if the network can predict the next word accurately when giving perfect inputs.

Finaly, **THIS IS THE TRANSFORMER NETWORK**.

![image-20210704164828355](https://tva1.sinaimg.cn/large/008i3skNgy1gs50jjvq3rj315a0n04hd.jpg)