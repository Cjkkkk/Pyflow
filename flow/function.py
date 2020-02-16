from . import autograd
from .tensor import Tensor
import numpy as np


class Add(autograd.Function):  
    @staticmethod      
    def forward(ctx, a, b):
        new_tensor = Tensor(a.data + b.data)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        b_grad = grad_output * 1
        a_grad = grad_output * 1
        return a_grad, b_grad

class Mul(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        new_tensor = Tensor(a.data * b.data)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors()
        b_grad = grad_output * a.data
        a_grad = grad_output * b.data
        return a_grad, b_grad

class Sub(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        new_tensor = Tensor(a.data - b.data)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        b_grad = grad_output * (-1)
        a_grad = grad_output * 1
        return a_grad, b_grad

class Truediv(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        new_tensor = Tensor(a.data / b.data)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors()
        b_grad = grad_output * (-a.data) / (b.data ** 2)
        a_grad = grad_output / b.data
        return a_grad, b_grad

class MM(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        new_tensor = Tensor(np.matmul(a.data, b.data))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors()
        a_grad = np.matmul(grad_output, np.transpose(b.data))
        b_grad = np.matmul(np.transpose(a.data), grad_output)
        return a_grad, b_grad

class ReLU(autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        copy = np.copy(a.data)
        copy[copy < 0] = 0
        return Tensor(copy)
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors()[0]
        a_grad = np.copy(grad_output)
        a_grad[a.data < 0] = 0
        return a_grad

class Sum(autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        new_tensor = Tensor(np.sum(a.data))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors()[0]
        a_grad = np.ones(a.data.shape)
        return a_grad

class SquareLoss(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        new_tensor = Tensor(np.sum(np.square(a.data - b.data)))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors()
        a_grad = 2.0 * (a.data - b.data)
        b_grad = -2.0 * (a.data - b.data)
        return a_grad, b_grad

def im2col(image, kernel_height, kernel_width, stride):
    # image is a 4d tensor([batchsize, channel, height, width])
    image_col = []
    for i in range(0, image.shape[2] - kernel_height + 1, stride):
        for j in range(0, image.shape[3] - kernel_width + 1, stride):
            col = image[:, :, i:i + kernel_height, j:j + kernel_width].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col

class Conv2d(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        batchsize, input_channel, height, width = input.data.shape
        output_channel, input_channel, kernel_height, kernel_width = weight.data.shape
        col_weight = np.transpose(weight.data.reshape([output_channel, -1]))
        input.data = np.pad(input.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
        conv_out = np.zeros(
            (batchsize, 
            output_channel, 
            (height - kernel_height ) // stride + 1,
            (width - kernel_width ) // stride + 1
            ))
        col_image = []
        for i in range(batchsize):
            img_i = input.data[i][np.newaxis, :]
            col_image_i = im2col(img_i, kernel_height, kernel_width, stride)
            col_image.append(col_image_i)
            if bias is not None:
                conv_out[i] = np.reshape(np.transpose(np.dot(col_image_i, col_weight) + bias.data), conv_out[0].shape)
            else:
                conv_out[i] = np.reshape(np.transpose(np.dot(col_image_i, col_weight)), conv_out[0].shape)
        col_image = np.array(col_image)
        ctx.save_for_backward(Tensor(col_image), Tensor(col_weight))
        return Tensor(conv_out)
    
    @staticmethod
    def backward(ctx, grad_output):
        col_image, col_weight = ctx.saved_tensors()
        return None

add = Add.apply
mul = Mul.apply
sub = Sub.apply
true_div = Truediv.apply
mm = MM.apply
sum_ = Sum.apply
square_loss = SquareLoss.apply
relu = ReLU.apply
conv2d = Conv2d.apply