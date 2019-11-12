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
