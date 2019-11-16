# This file contains a set of functions that help in implementing Attention in models
# The file is isolated from a larger model to focus just on the attention implementation code
# Attention scoring and attention context vectors are implemented here


#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dec_hidden_state = [5,1,20] #e.g. Decoder hidden state

annotation = [3,12,45] #e.g. Encoder hidden state

# visualize the single annotation
plt.figure(figsize=(1.5, 4.5))
sns.heatmap(np.transpose(np.matrix(annotation)), annot=True, cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)

def single_dot_attention_score(dec_hidden_state, enc_hidden_state):
    # return the dot product of the two vectors
    return np.dot(dec_hidden_state, enc_hidden_state)

print (single_dot_attention_score(dec_hidden_state, annotation))

annotations = np.transpose([[3,12,45], [59,2,5], [1,43,5], [4,3,45.3]])

# visualize our annotation (each column is an annotation)
ax = sns.heatmap(annotations, annot=True, cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)

def dot_attention_score(dec_hidden_state, annotations):
    # TODO: return the product of dec_hidden_state transpose and enc_hidden_states
    return np.matmul(np.transpose(dec_hidden_state), annotations)

attention_weights_raw = dot_attention_score(dec_hidden_state, annotations)
print (attention_weights_raw)

def softmax(x):
    x = np.array(x, dtype=np.float128)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

attention_weights = softmax(attention_weights_raw)
print (attention_weights)

def apply_attention_scores(attention_weights, annotations):
    # Multiple the annotations by their weights
    return attention_weights * annotations

applied_attention = apply_attention_scores(attention_weights, annotations)
print (applied_attention)

# visualize our annotations after applying attention to them
ax = sns.heatmap(applied_attention, annot=True, cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)

def calculate_attention_vector(applied_attention):
    return np.sum(applied_attention, axis=1)

attention_vector = calculate_attention_vector(applied_attention)
print (attention_vector)

# visualize the attention context vector
plt.figure(figsize=(1.5, 4.5))
sns.heatmap(np.transpose(np.matrix(attention_vector)), annot=True, cmap=sns.light_palette("Blue", as_cmap=True), linewidths=1)
