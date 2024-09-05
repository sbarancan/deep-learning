<aside>
💡 *Author: Barancan Sonmez*

</aside>

The notes I wrote down to my notebook for the first two courses of Coursera. I wrote the bullet points when reviewing my notebook. So it's more like a summary and terminology here.

- a machine learning algorithm is the process that uncovers the underlying relationship within the data
- ImageNET → visual recognition challange
- Supervised → data has y which is target attribute, also called labels or ground truth
- unsupervised → no ground truth, learns the underlying pattarns or rules from the data
- Paper → Fisher, “the use of multiple measurements in taxonomic problems
- ML Model
    - Classification → discrete output
    - Regression → continuous output
- *Data determines the upper bound of performance*

---

# Deep Learning

- Why now? → Big data, hardware, software
    - 1952 - Stochoastic Gradient Descent
    - 1958 - Perceptron, learnable weights → building block of deep learning
    - 1986 - **Backpropagation**
    - 1995 - Deep Convolutional NN, digit recognition
- Activation Functions, do know their definitions and derivatives
    - Sigmoid `tf.math.sigmoid(z)`
    - Hyberbolic `tf.math.tanh(z)`
    - ReLU - Rectified Linear Unit `tf.math.relu(z)`
- The purpose of activation functions is to introduce non-linearity into the network.
- **Applying Neural Network**
- Training → Quantifying the Loss; Loss Function, Objective Function, Cost Function
- **Gradient Descent Algorithm**
- Initialize weights randomly $W∼N(0,σ^2)$
- Loop until convergence
    - Compute Gradient $∇J(W)=  \frac{\partial J}{\partial W}$
    - Update Weights $W \leftarrow W - \alpha \nabla J(W)$
- **Return**: Weights W
- The algorithm in practice has differences for efficiency, especially **Mini Batches**
- Setting the Learning Rate$(\alpha)$  → try some see what works, or do smarter, *design an adaptive learning rate that adapts landscape*
- Adaptive Learning Rates
    - SGD `tf.keras.optimizers.SGD`
    - ADAM `rf.keras.optimizers.Adam`
    - Adadelta `rf.keras.optimizers.Adadelta`
    - Adagrad `rf.keras.optimizers.Adagrad`
    - RMSProp `rf.keras.optimizers.RMSProb`
- Backpropagation
- Overfitting
- Regularization
    - Dropout → during training randomly set some activations to zero `tf.keras.layers.Dropout(p=0.5)`
    - Early Stopping → stop the training before overfit
    
    ---
    

# MIT Data Sequence Modelling, RNN, Transformers, and Attention

- One-to-one → Binary Classification
- Many-to-one → Sentiment Classification
- One-to-many → Image Captioning
- RNNs = neurons with recurrence.  $\^y_t = f(x_t, h_{t-1})$    x: input, h: past memory
- RNNs have a state h_t that is updated at each time step as a sequence is processed
- Apply a recurrence relation at each time step

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f99dd4c3-825f-4406-9ff9-a97a1e592af1/3ef1344e-1c6d-4fd4-b9f0-8624ff80626c/Untitled.png)

- RNN State Update and Output
- Update Hidden State → $h_t = tanh(W_{hh}^{T}h_{t-1} + W_{xh}^{T} x_t)$
- Output vector → $\^y = W_{hy}^T h_t$
- `tf.keras.layers.SimpleRNN(rnn_units)`
- Sequence Modeling: Design Criteria
    - variable-length sequences
    - long term dependencies
    - maintain information about order
    - share parameters across the sequence
- Predict Next Word
    - We need to represent language numerically → Word Embeddings (Yoshuo Bengio)
- Backpropagation Through Time
    - Standart RNN Gradient Flow
    - Vanishing Gradients
    - Problem of Long Term Dependencies
- Why vanishing gradients a problem?
    1. Activation functions, using ReLU
    2. Parameter Initialization → initialize weights to idendity matrix, biases to zero
    3. Gated Cells → use gates to selectively add or remove information within each recurrent unit with LSTM, GRU
- Long-Short Term Memory(LSTM)
    - rely on a gated cell to track information through many time steps
    - effective at tracking long time dependencies
        1. Maintain cell state
        2. Use gates to control flow of information
            1. Forhet
            2. Store
            3. Update
            4. Output
        3. Backpropagation in time with partially uninterrupted gradient flow
- `loss = tf.nn.softmax_cross_entropy_with_logits(y, predicted)`
- RNN Applications → music generation, sentiment classification
- Limitations with RNNs
    - Encoding Bottleneck
    - Slow, no parallelization
    - not long memory
- We want;
    - continuous stream of information
    - parallelization
    - long memory
- then → **ATTENTION IS ALL YOU NEED**
    - Attending to the most important parts if an input
    - Identify which parts to attend to → similar to search problem
    - Extract the features with high attention

---

# Deep Learning - Coursera #1

- Real Estate, Online Adds → standart neural networks
- Image Applications → Convolutional Neural Networks
- Sequential Data, Speech, Translation → RNN
- Complicated tasks, autonomous driving → complex and hybrid models
- Logistic Regression
    - Given $x$, we want $\^y = P(y=1|x)$
    - Parameters W bias b
    - $\^y = \sigma(w^Tx + b)$
    - $\sigma(z) = \frac{1}{1+ e^{-z}}$
    - Cost Function = $J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$
    - hypothesis function → $h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$
- Gradient Descent → Want to find weights, biases that minimizes the cost function
    - Initial point → go to minimal direction
    - Repeat → $w:=w - \alpha\frac{dJ(w)}{dw}$   alpha is the learning rate, determining the size of step
    - Computation Graph

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f99dd4c3-825f-4406-9ff9-a97a1e592af1/acd5859e-ad36-4498-8add-bf3b34f9be63/Untitled.png)

- Modify the  parameters w1, w2, b to reduce loss → backpropagation
- For loops, explicit for loops makes the algorithm less efficient.
- To solve this issue → VECTORIZATION → in python called Broadcasting
- Gradient Descent for Neural Network
    - Backpropagation
- Random Initialization → initializing all weights to zero do not work well
- $a^{[2]} =$ activation vector of second layet
- $w_j^{[i]} =$ column vector of parameters of i’th layer and j’th neuron
- use linear activation in regression problems only
- building binary classification, using tanh for hidden layers is always better than using sigmoid
- y = np.sum(x, axis=1) every row is summed, column vector
- initialize weights to small random numbers
- $W^{[k]}$  # of rows = # of neurons in the k’th layer, # of columns = # of inputs of the layer
- General Methodology;
    - Define the neural network structure, num of inputs, hidden units, etc.
    - Initialize model parameters
    - Loop
        - Forward propagation
        - Compute Loss
        - Backpropagation to get the gradients
        - Update parameters
- Parameters →
    - W, b
- Hyperparameters →
    - learning rate $\alpha$ , # iterations, # of hidden layers L, # of hidden units n, choice of activation functions, momentum, minibatches size, …
- Hyperparameters determines parameters
- Applied deep learning is a very emprical process. Loop around idea-code-experiment
- The “cache” is used in the implementation to store values computed during forward propagation to be used in backward propagation.

Why Deep Representations?

> ***I think analogies between the deep learning and the human brain are sometimes a little bit dangerous. But there is a lot of truth too, this being how we thing that humant brain works and that the human brain probably detect simple things like edges first and then put them together to form more and more complex objects, and so that has served as a loose form of inspriation for some deep learning as well.***
> 
> 
> -Andrew Ng
> 

---