# CS224N Winter 2019
## Lecture 1: Introduction and Word Vectors

### Word representation
#### One-hot Vectors
In traditional NLP, words is represented by one-hot vectors:
```python
motel = np.array([0, 0, 0, 1, 0])
hotel = np.array([0, 0, 1, 0, 0])
```
But these two vectors are orthogonal, there's no way to measure similarity.

#### Word2Vec
Distributed representation.

For each position $t =1, ..., T$, predict context words within a window of fixed size m, given center word $w_j$.
Objective function
$$J(\theta) = -\frac{1}{T}log L(\theta) = -\frac{1}{T}\sum_{t=1}^T\sum_{-m \le j \le m, \ j\ne 0} log P(w_{t+j} | w_{t};\theta)
$$
To calculate $P(w_{t+j} | w_{t};\theta)$, we propose 2 vectors per word $w$:
	- $v_{w}$ when $w$ is a center word
	- $u_{w}$ when $w$ is a context word
	- Average both at the end
We have prediction function (softmax function):
$$P(o|c) = \frac{exp(u_{o}^Tv_{c})}{\sum_{w \in v}exp(u_{w}^Tv_{c})}$$
Compute gradients:
$$ TBD \\ \frac{\partial}{\partial V_c} = log p(o|c) = u_o - \sum_{x=1}^V p(x|c) u_x \\ TBD$$
Questions so far:
	1. What's the logic of using softmax representation to calculate $P(w_{t+j} | w_{t};\theta)$?
	2. Prof. Manning mentioned that the second part of $\frac{\partial}{\partial V_c}$, $\sum_{x=1}^V p(x|c) u_x$, is deemed as an expectation, it's weighted average of the representation of each word multiply by the probability of it in current model, I'm not clear of this yet.

### Word2Vec Implementation (WIP)
[Jupyter Notebook](https://github.com/jerrytim/LearningNLP/blob/master/word2vec.ipynb)