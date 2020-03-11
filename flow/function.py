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
        a_grad = grad_output * np.ones(a.data.shape)
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
        a_grad = grad_output * 2.0 * (a.data - b.data)
        b_grad = grad_output * -2.0 * (a.data - b.data)
        return a_grad, b_grad

class MaxPool2d(autograd.Function):
    @staticmethod
    def forward(ctx, tensor, kernel_size, stride, padding):
        data = tensor.data
        data = np.pad(data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
        batchsize, channel, height, width = data.shape
        output = np.zeros(
            (batchsize, 
            channel, 
            (height - kernel_size[0] + 2 * padding[0]) // stride[0] + 1,
            (width - kernel_size[1] + 2 * padding[1]) // stride[1] + 1
            ))
        batchsize, channel, output_height, output_width = output.shape
        for i in range(batchsize):
            for j in range(channel):
                for h in range(0, height - kernel_size[0] + 1, stride[0]):
                    for w in range(0, width - kernel_size[1] + 1, stride[1]):
                        output[i, j, h // stride[0], w // stride[1]] = np.max(data[
                            i, 
                            j, 
                            h : h + kernel_size[0], 
                            w : w + kernel_size[1]
                            ])
        ctx.save_for_backward(data, kernel_size, stride, padding)
        return Tensor(output)
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        data, kernel_size, stride, padding = ctx.saved_tensors()
        batchsize, channel, height, width = data.shape
        batchsize, channel, output_height, output_width = grad_output.shape
        
        grad = np.zeros(data.shape)
        for i in range(batchsize):
            for j in range(channel):
                for h in range(0, height - kernel_size[0] + 1, stride[0]):
                    for w in range(0, width - kernel_size[1] + 1, stride[1]):
                        mask = (data[i, j, h : h + kernel_size[0], w : w + kernel_size[1]] == np.max(data[i, j, h : h + kernel_size[0], w : w + kernel_size[1]]))
                        grad[i, j, h : h + kernel_size[0], w : w + kernel_size[1]] += mask * grad_output[i, j, h // stride[0], w // stride[1]]
        
        return grad[:, :, padding[0]: height-padding[0], padding[1]: width-padding[1]], None, None, None

def im2col(image, kernel_height, kernel_width, stride):
    # image is a 4d tensor([batchsize, channel, height, width])
    image_col = []
    for i in range(0, image.shape[2] - kernel_height + 1, stride[0]):
        for j in range(0, image.shape[3] - kernel_width + 1, stride[1]):
            col = image[:, :, i:i + kernel_height, j:j + kernel_width].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col

class Conv2d(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        input, weight = input.data, weight.data
        batchsize, input_channel, height, width = input.shape
        output_channel, input_channel, kernel_height, kernel_width = weight.shape
        col_weight = weight.reshape([output_channel, -1])
        input = np.pad(input, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
        conv_out = np.zeros(
            (batchsize, 
            output_channel, 
            (height - kernel_height + 2 * padding[0]) // stride[0] + 1,
            (width - kernel_width + 2 * padding[1]) // stride[1] + 1
            ))
        col_image = []
        for i in range(batchsize):
            img_i = input[i][np.newaxis, :]
            col_image_i = im2col(img_i, kernel_height, kernel_width, stride)
            col_image.append(col_image_i)
            if bias is not None:
                conv_out[i] = np.reshape(np.dot(col_weight, np.transpose(col_image_i)) + bias.data, conv_out[0].shape)
            else:
                conv_out[i] = np.reshape(np.dot(col_weight, np.transpose(col_image_i)), conv_out[0].shape)
        col_image = np.array(col_image)
        ctx.save_for_backward(col_image, col_weight, bias,
            input.shape,
            weight.shape,
            stride,
            padding
            )
        return Tensor(conv_out)
    
    @staticmethod
    def backward(ctx, grad_output):
        col_image, col_weight, bias, input_shape, weight_shape, stride, padding = ctx.saved_tensors()
        batchsize, output_channel, output_height, output_width = grad_output.shape
        batchsize, input_channel, height, width = input_shape
        output_channel, input_channel, kernel_height, kernel_width = weight_shape
        
        # init gradient for img2col
        col_weight_gradient = np.zeros(col_weight.shape)
        conv_out_gradient = grad_output.reshape(batchsize, output_channel, -1)
        
        # init gradient for input tensor
        bias_gradient = np.ones(output_channel) if bias is None else None
        input_gradient = np.zeros(input_shape)
    
        for i in range(batchsize):
            col_image_gradient = np.matmul(np.transpose(conv_out_gradient[i]), col_weight)
            col_weight_gradient += np.matmul(conv_out_gradient[i], col_image[i])
            
            j = 0
            for h in range(0, height - kernel_height + 1, stride[0]):
                for w in range(0, width - kernel_width + 1, stride[1]):
                    input_gradient[i, :, h: h + kernel_height, w: w + kernel_width] += col_image_gradient[j].reshape((input_channel, kernel_height, kernel_width))
                    j += 1
        
        weight_gradient = col_weight_gradient.reshape(output_channel, input_channel, kernel_height, kernel_width)
        # remove padding
        input_gradient = input_gradient[:, :, padding[0]: height-padding[0], padding[1]: width-padding[1]]
        return input_gradient, weight_gradient, bias_gradient, None, None


class View(autograd.Function):
    @staticmethod
    def forward(ctx, tensor, shape):
        ctx.save_for_backward(tensor.data.shape)
        new_tensor = Tensor(tensor.data.reshape(shape))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        original_shape = ctx.saved_tensors()[0]
        grad = grad_output.reshape(grad_output)
        return grad

class LogSoftmax(autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        data = tensor.data
        pro_sum = np.sum(data)
        pro = data / pro_sum
        log_pro = np.log(pro)
        ctx.save_for_backward(pro)
        return Tensor(log_pro)
    
    @staticmethod
    def backward(ctx, grad_output):
        pro = ctx.saved_tensors()[0]
        return grad_output * (1 - pro)

class NllLoss(autograd.Function):
    @staticmethod
    def forward(ctx, input, target, size_average):
        # input is size (N, C), target is size (N, 1), output is size (N, 1)
        input, target = input.data, target.data
        nll = [- log_pro[target[idx]] for idx, log_pro in enumerate(input)]
        if size_average:
            loss = np.average(nll)
        else:
            loss = np.sum(nll)
        ctx.save_for_backward(target, input, size_average)
        return Tensor(loss)
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is size (N, 1), output is size (N, C) 
        target, input, size_average = ctx.saved_tensors()
        output = np.zeros(input.shape)
        batch_size = output.shape[0]
        for idx in range(batch_size):
            output[idx, target[idx]] = - 1
        if size_average:
            output = output * grad_output / batch_size
        else:
            output = output * grad_output
        return output, None, None

add = Add.apply
mul = Mul.apply
sub = Sub.apply
true_div = Truediv.apply
mm = MM.apply
sum_ = Sum.apply
square_loss = SquareLoss.apply
relu = ReLU.apply
conv2d = Conv2d.apply
maxpool2d = MaxPool2d.apply
log_softmax = LogSoftmax.apply
view = View.apply
nll_loss = NllLoss.apply