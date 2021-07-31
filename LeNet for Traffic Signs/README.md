# Implementing Yann LeCun’s LeNet-5 in PyTorch


![image](https://user-images.githubusercontent.com/56880104/127745094-6cc14e5c-d2a4-429c-af18-511c368d6c9c.png)


(W−F+2P)/S+1


Layer 1 (C1): The first convolutional layer with 6 kernels of size 5×5 and the stride of 1. Given the input size (32×32×1), the output of this layer is of size 28×28×6.

Layer 2 (S2): A subsampling/pooling layer with 6 kernels of size 2×2 and the stride of 2. The subsampling layer in the original architecture was a bit more complex than the traditionally used max/average pooling layers. I will quote [1]: “ The four inputs to a unit in S2 are added, then multiplied by a trainable coefficient, and added to a trainable bias. The result is passed through a sigmoidal function.”. As a result of non-overlapping receptive fields, the input to this layer is halved in size (14×14×6).

Layer 3 (C3): The second convolutional layer with the same configuration as the first one, however, this time with 16 filters. The output of this layer is 10×10×16.

Layer 4 (S4): The second pooling layer. The logic is identical to the previous one, but this time the layer has 16 filters. The output of this layer is of size 5×5×16.

Layer 5 (C5): The last convolutional layer with 120 5×5 kernels. Given that the input to this layer is of size 5×5×16 and the kernels are of size 5×5, the output is 1×1×120. As a result, layers S4 and C5 are fully-connected. That is also why in some implementations of LeNet-5 actually use a fully-connected layer instead of the convolutional one as the 5th layer. The reason for keeping this layer as a convolutional one is the fact that if the input to the network is larger than the one used in the initial input, so 32×32 in this case, this layer will not be a fully-connected one, as the output of each kernel will not be 1×1.

Layer 6 (F6): The first fully-connected layer, which takes the input of 120 units and returns 84 units. In the original paper, the authors used a custom activation function — a variant of the tanh activation function. For a thorough explanation, please refer to Appendix A in [1].

Layer 7 (F7): The last dense layer, which outputs 10 units. The authors used Euclidean Radial Basis Function neurons as activation functions for this layer.
