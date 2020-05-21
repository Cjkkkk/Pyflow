from . import autograd
from .utils import _make_pair
from .tensor import Tensor, ones, zeros, transpose
import numpy as np

class Add(autograd.Function):  
    @staticmethod      
    def forward(ctx, a, b, inplace=False):
        if inplace:
            a.data += b.data
            return a
        else:
            new_tensor = Tensor(a.data + b.data)
            return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        b_grad = grad_output * 1
        a_grad = grad_output * 1
        return a_grad, b_grad

class Mul(autograd.Function):
    @staticmethod
    def forward(ctx, a, b, inplace=False):
        ctx.save_for_backward(a, b)
        if inplace:
            a.data *= b.data
            return a
        else:
            new_tensor = Tensor(a.data * b.data)
            return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        b_grad = grad_output * a.data
        a_grad = grad_output * b.data
        return a_grad, b_grad

class Sub(autograd.Function):
    @staticmethod
    def forward(ctx, a, b, inplace=False):
        if inplace:
            a.data -= b.data
            return a
        else:
            new_tensor = Tensor(a.data - b.data)
            return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        b_grad = grad_output * (-1)
        a_grad = grad_output * 1
        return a_grad, b_grad

class Truediv(autograd.Function):
    @staticmethod
    def forward(ctx, a, b, inplace=False):
        ctx.save_for_backward(a, b)
        if inplace:
            a.data /= b.data
            return a
        else:
            new_tensor = Tensor(a.data / b.data)
            return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
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
        a, b = ctx.saved_tensors
        a_grad = np.matmul(grad_output.data, np.transpose(b.data))
        b_grad = np.matmul(np.transpose(a.data), grad_output.data)
        return Tensor(a_grad), Tensor(b_grad)

class ReLU(autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        copy = a.copy()
        copy[copy < 0] = 0
        return copy
    
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        a_grad = grad_output.copy()
        a_grad[a < 0] = 0
        return a_grad

class Sum(autograd.Function):
    @staticmethod
    def forward(ctx, a, axis=None):
        ctx.save_for_backward(a)
        new_tensor = Tensor(np.sum(a.data, axis=axis))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        a_grad = grad_output * ones(a.shape)
        return a_grad

class Min(autograd.Function):
    @staticmethod
    def forward(ctx, a, axis=None):
        idx = np.argmin(a.data, axis=axis)
        ctx.save_for_backward(a, axis, idx)
        new_tensor = Tensor(np.min(a.data, axis=axis))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, axis, idx = ctx.saved_tensors
        grad = np.zeros(a.shape)
        if axis is None:
            grad.itemset(idx, grad_output.item())
        else:
            expand_idx = np.expand_dims(idx, axis=axis)
            filled_grad = np.expand_dims(grad_output.data, axis=axis)
            np.put_along_axis(grad, expand_idx, filled_grad, axis=axis)
        return Tensor(grad)

class Max(autograd.Function):
    @staticmethod
    def forward(ctx, a, axis=None):
        idx = np.argmax(a.data, axis=axis)
        ctx.save_for_backward(a, axis, idx)
        new_tensor = Tensor(np.max(a.data, axis=axis))
        return new_tensor

    @staticmethod
    def backward(ctx, grad_output):
        a, axis, idx = ctx.saved_tensors
        grad = np.zeros(a.shape)
        if axis is None:
            grad.itemset(idx, grad_output.item())
        else:
            expand_idx = np.expand_dims(idx, axis=axis)
            filled_grad = np.expand_dims(grad_output.data, axis=axis)
            np.put_along_axis(grad, expand_idx, filled_grad, axis=axis)
        return Tensor(grad)
    
class SquareLoss(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        new_tensor = Tensor(np.sum(np.square((a - b).data)))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        a_grad = grad_output * 2.0 * (a - b)
        b_grad = grad_output * -2.0 * (a - b)
        return a_grad, b_grad

class MaxPool2d(autograd.Function):
    @staticmethod
    def forward(ctx, tensor, kernel_size, stride=1, padding=0):
        kernel_size = _make_pair(kernel_size)
        stride = _make_pair(stride)
        padding = _make_pair(padding)

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
        ctx.save_for_backward(tensor, kernel_size, stride, padding)
        return Tensor(output)
    
    @staticmethod
    def backward(ctx, grad_output):
        tensor, kernel_size, stride, padding = ctx.saved_tensors
        batchsize, channel, height, width = tensor.shape
        batchsize, channel, output_height, output_width = grad_output.shape
        
        grad = zeros(tensor.shape)
        for i in range(batchsize):
            for j in range(channel):
                for h in range(0, height - kernel_size[0] + 1, stride[0]):
                    for w in range(0, width - kernel_size[1] + 1, stride[1]):
                        mask = tensor[i, j, h : h + kernel_size[0], w : w + kernel_size[1]] == max(tensor[i, j, h : h + kernel_size[0], w : w + kernel_size[1]])
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
        col_image, col_weight, bias, input_shape, weight_shape, stride, padding = ctx.saved_tensors
        batchsize, output_channel, output_height, output_width = grad_output.shape
        batchsize, input_channel, height, width = input_shape
        output_channel, input_channel, kernel_height, kernel_width = weight_shape
        
        # init gradient for img2col
        col_weight_gradient = zeros(col_weight.shape)
        conv_out_gradient = grad_output.reshape(batchsize, output_channel, -1)
        
        # init gradient for input tensor
        bias_gradient = ones(output_channel) if bias is None else None
        input_gradient = zeros(input_shape)
    
        for i in range(batchsize):
            col_image_gradient = mm(transpose(conv_out_gradient[i]), col_weight)
            col_weight_gradient += mm(conv_out_gradient[i], col_image[i])
            
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
        ctx.save_for_backward(tensor.shape)
        new_tensor = tensor.copy().reshape(shape)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        original_shape, = ctx.saved_tensors
        grad = grad_output.copy().reshape(original_shape)
        return grad

class LogSoftmax(autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim):
        # tensor size is (N, C)
        data = tensor.data
        data_shift = data - np.max(data)
        data_shift_exp = np.exp(data_shift)
        exp_sum = np.sum(data_shift_exp, axis=dim, keepdims=True)
        exp_sum[exp_sum == 0] = 1e-10
        res = data_shift - np.log(exp_sum)
        ctx.save_for_backward(data_shift_exp, exp_sum)
        return Tensor(res)
    
    @staticmethod
    def backward(ctx, grad_output):
        data_shift_exp, exp_sum = ctx.saved_tensors
        e = - data_shift_exp / exp_sum
        N, C = e.shape
        grad = zeros((N, C))
        for i in range(N):
            jac = np.tile(e[i], (C, 1))
            jac[np.diag_indices_from(jac)] += 1
            grad[i] = mm(Tensor(np.transpose(jac)), grad_output[i])
        return grad

class NllLoss(autograd.Function):
    @staticmethod
    def forward(ctx, input, target, reduction="average"):
        # input is size (N, C), target is size (N, 1), output is size (N, 1)
        input, target = input.data, target.data
        nll = [- log_pro[target[idx]] for idx, log_pro in enumerate(input)]
        if reduction == "average":
            loss = np.average(nll)
        elif reduction == "sum":
            loss = np.sum(nll)
        else:
            raise RuntimeError("unsupported reducetion type.")
        ctx.save_for_backward(target, input, reduction)
        return Tensor(loss)
    
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is size (N, 1), output is size (N, C) 
        target, input, reduction = ctx.saved_tensors
        output = zeros(input.shape)
        batch_size = output.shape[0]
        for idx in range(batch_size):
            output[idx, target[idx]] = - 1
        if reduction == "average":
            output = output * grad_output / batch_size
        elif reduction == "sum":
            output = output * grad_output
        else:
            raise RuntimeError("unsupported reducetion type.")
        return output, None, None

add = Add.apply
mul = Mul.apply
sub = Sub.apply
true_div = Truediv.apply

max = Max.apply
min = Min.apply
mm = MM.apply
sum = Sum.apply
square_loss = SquareLoss.apply
relu = ReLU.apply
conv2d = Conv2d.apply
max_pool2d = MaxPool2d.apply
log_softmax = LogSoftmax.apply
view = View.apply
nll_loss = NllLoss.apply