## Introduction to Attention

Attention models help machine learning models focus on a subset of its inputs (or features). It is a mechanism that was initially developed to improve the performance of the Encoder-Decoder RNN on machine translation.

The attention mechanism helps memorize long source sentences in neural machine translation.

### Sequence to sequence models:
- Input --> Encoder --> Context --> Decoder --> Output

**Limitation of sequence of sequence models:**
- Encoder is only able to send one context vector (i.e., hidden state dimensions) to the Decoder (irrespective of the length of the input sequence)
- Choosing a reasonable size of the context vector (e.g., 256 or 512) creates issues when dealing with long input sequences
- If you increase the size of the context vector significantly, the model starts overfitting with shorter sequences (also not ideal), and performance reduces

**This is the limitation that Attention solves**

### Attention models:

- In attention models, all the hidden states get sent from the encoder to the decoder (i.e., all hidden states across time steps get sent)
- This gives us the benefit of having the flexibility in the context size --> so longer sequences can have longer context vectors which better capture the context of the longer sentences
- As a reminder, the first hidden state captures the essence of the first input word the most Similarly, the second hidden state with the second word, the third hidden state with the third word, and so on. Even though the last hidden state also captures information from prior time steps
- The size of the context matrix varies depending upon the input sequence

Attention Decoder:
- The decoder uses each of the passed through hidden states to generate output words
- It learns via training which hidden state to use for different words
- The decoder uses a scoring function to score each hidden state in the encoder context matrix
- Each score of the hidden states is then translated via a softmax function to get a probability distribution
- The weighted sum of the hidden state values with the softmax scores results in an Attention Context vector for the decoder
- The Context vector then merges with the decoder's hidden state (i.e., is concatenation to form one long vector)
- This concatenated vector is then passed through a fully connected layer and a tanh activation function to generate the output word

Two types of attention models are most prevalent:
- Additive Attention (Bahdanau Attention)
- Multiplicative Attention (Luong Attention)

Attention scoring functions:
- The function takes in the hidden state of the decoder and the set of hidden states of the Encoder
- Given this scoring function is calculated at each time step in the decoder, we only take the hidden state of the decoder at the specific time step or the one prior time

Different approaches to calculating the scoring functions include:
- Dot product: The dot product is a similarity measure between the vectors
- General product: Introduces a weight matrix between the decoder hidden state and the encoder hidden matrix. This allows you multiple between two different embedding sizes
- 3-Concat: concat the two vectors (i.e., hidden state of decoder at current time step and the encoder hidden matrix) and then pass it through a feed forward neural network

Example of a computer vision application using attention:
- Caption generation with visual attention:
    a. Input image
    b. Convolutional feature extraction using a VGG network as the encoder
    c. RNN with attention over the image as the decoder
    d. Word by word generation of the captions
