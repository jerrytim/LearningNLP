# CS224N Winter 2019
## Lecture 2: Word Vectors and Word Senses

### Negative Sampling
Main idea: train binary logistic regressions for a true pair (center word and word in its context window) versus several noise pairs (the center word paired with a random word).

### Skip-gram model with Negative Sampling
Maximize probability that real outside word appears, minimize probability that random words appear
$$J(\theta) = \frac{1}{T}\sum_{t=1}^TJ_t(\theta)
\\ \max J_t(\theta) = log\sigma(u_{o}^Tv_{c}) + \sum_{i=1}^k \mathbb E_{j \sim P(w)} [log\sigma(-u_{j}^Tv_{c})]
\\ \min J_{neg-sample}(o, v_c, U) = -log\sigma(u_{o}^Tv_{c}) - \sum_{k=1}^klog\sigma(-u_{k}^Tv_{c})
\\ \sigma(x) \ \text{is a sigmoid function}, P(w) \ \text{represents a unigram distribution}
$$
question: how to exactly convert maximize cost function to minimize cost function, especially the second part? $E_{j \sim P(w)} [log\sigma(-u_{j}^Tv_{c})]$ is an expectation, in minimize cost function formula, we can change the sigmoid function to $1-\sigma(x)$, but still not sure how the expectation goes away.

### GloVe Model
Ratios of co-occurrence probabilities can encode meaning components
To capture ratios of co-occurrence probabilities as linear meaning components in a word vector space, let
$$ \text{log-bilinear model} \ w_i \cdot w_j = log P(i|j)
\\ \text{with vector differences} \ w_x \cdot (w_a - w_b)  = log \frac{P(x|a)}{P(x|b)}
\\ J= \sum_{i,j=1}^V f(X_{ij}) (w_i^T\tilde w_j + b_j + \tilde b_j - log X_{ij})^2
$$
make dot product of $w_i \cdot w_j$ as close as co-occurrence probabilities $log P(i|j)$