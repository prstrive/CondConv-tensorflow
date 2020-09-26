# CondConv-tensorflow
Conditional convolution (Dynamic convolution) in tensorflow2.2.0. This depository implements the method described in the paper:
 
 >CondConv: Conditionally Parameterized Convolutions for Efficient Inference  
 >Brandon Yang, Gabriel Bender, Quoc V.Le, Jiquan Ngiam  
 >[Source PDF](https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf)
 
Meanwhile, the softmax with a large temperature for kernel attention introduced by [Dynamic Convolution: Attention Over Convolution Kernels](https://arxiv.org/pdf/1912.03458.pdf) is adopted. 
 
Another similar paper: [DyNet: Dynamic Convolution for Accelerating Convolutional Neural Networks](https://arxiv.org/pdf/2004.10694.pdf).

### Start
You can start according to the default arguments by `python main.py`. Or specify the arguments:
```python
python main.py --arch cond_cifar_resnet --num_layers 56 --num_experts 3 --dataset cifar10 --num_classes 10
```
 
