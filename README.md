# Memory-Efficient-Autoencoder
A repo looking at autoencoders that can be applied to extremely large 2D and 3D tensors. There are 3 requirements the network must have,

- Works on any shaped tensor. For example, if the network is kept all convolutional this would work.
- Once the model is trained it can be used on huge tensors in a memory efficient way. Ideally it will be able to encode and decode tensors that are larger then the GPU memory.
- There needs to be as much information flow as possible. For example, if the tensor size is 10,000 by 10,000 I would the pixels in either corner to have at least a little communication.

The real reason I want to do this is to tackle a few problems [here](https://github.com/loliverhennigh/Steady-State-Flow-With-Neural-Nets). Right now I will just focus on this simpler autoencoder problem.

# Networks

I have a few ideas on how to do this. Here is a fig showing one possible idea,

![alt tag](https://github.com/loliverhennigh/Memory-Efficient-Autoencoder/blob/master/figs/low_memory_flow_prediciton.png)

There are a lot of other ideas I am playing with. This [paper](https://arxiv.org/pdf/1705.06820v1.pdf) looks like it has some cool stuff I could try.




