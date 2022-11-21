A simple Pytorch reimplementation.

[![CircleCI](https://circleci.com/gh/Cjkkkk/Pyflow.svg?style=svg)](https://circleci.com/gh/Cjkkkk/Pyflow)

## reference
https://dlsyscourse.org/

## install/develop
```bash
pip install -r requirements.txt
python setup.py develop
```
## example
```bash
python example/mnist.py
```
## test
```bash
python -m unittest
```

PS: `pytorch` is required to check the correctness of implementation.

## todo
### Operator
* mm [done]
* relu [done]
* max_pool2d [done]
* conv2d []
    * efficient im2col
* log_softmax [done]
* view [done]
* nll_loss [done]
* in-place add/sub/mul/div [done]


### Autograd
* gradient accum in backward [done]
* gradient_check_tool [done]
* no_grad [done]
* grad as tensor [done]
* inplace gradient calculation []

### Module
* module load/store [done]

### Test
* test []

### Example
* mnist example [done]

### Memory Optimization
* ref count []
* in-place[]
* normal sharing []

### Other
* dataloader [done]
* hide numpy from user []
* dot graph []


### Bug
* Conv grad with padding