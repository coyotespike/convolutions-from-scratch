![](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

# CNN

A convolutional network at a fairly basic level!

## Usage

`conda activate <yourenv>`
`pip install -r ../requirements.txt`
`python3 cnn.py`

## Implementation notes

### Size

Each layer doubles the channels and roughly halves the input. This is a best practice which balance speed of training with sufficient computation.

### Residual Connections

It includes residual connections in each block. These allow the neuron to consider whether performing the computation was helpful at all! They are essential to building very deep networks. This one has only 4 layers, but could be expanded to 100 as long as the residual connection is included. Otherwise the gradient will fail to get propagated all the way back.

### Dropout

Initially I implemented dropout only for the last fully connected layer (the linear layer), as supposedly the convolutional layers can learn their way around it. However it made a difference everywhere.

### Initialization

I used the default He initialization, but experimented with Xavier initialization.

### Input Normalization

I used a script to calculate the norm and mean over the entire input. This is applied to at the start to each input batch, normalizing to zero mean and unit variance. Ths smooths the the loss landscape into a more uniform shape, speeding up training.

### BatchNorm2d

Norming each batch led to a huge increase in training accuracy. 10/10 recommend. Conceptually, this is similar input normalization, except applied to the activations of a layer rather than to the input, and using learned paramaeters $\gamma$ and $\beta$. See [this brilliant explanation](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739).

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*VsN_9_AN2ji8hCZYSTTV0w.png)

### Max Pooling

This reduces the dimensionality by quite a bit, so I only used it once.
