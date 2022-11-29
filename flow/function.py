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
    def forward(ctx, a, b, bias):
        ctx.save_for_backward(a, b, bias)
        if bias is None:
            new_tensor = Tensor(np.matmul(a.data, b.data))
        else:
            new_tensor = Tensor(np.matmul(a.data, b.data) + bias.data)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b, bias = ctx.saved_tensors
        a_grad = np.matmul(grad_output.data, np.transpose(b.data))
        b_grad = np.matmul(np.transpose(a.data), grad_output.data)
        if bias is None:
            bias_grad = None
        else:
            bias_grad = Tensor(np.sum(grad_output.data, axis=0))
        return Tensor(a_grad), Tensor(b_grad), bias_grad

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
    def forward(ctx, tensor, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        kernel_size = _make_pair(kernel_size)
        stride = _make_pair(stride)
        padding = _make_pair(padding)

        KH, KW = kernel_size
        SH, SW = stride
        PH, PW = padding
        
        data = tensor.data
        data = np.pad(data, ((0, 0), (0, 0), (PH, PH), (PW, PW)), 'constant', constant_values=0)
        N, C, H, W = data.shape
        
        output = np.zeros(
            (N, 
            C, 
            (H - KH + 2 * PH) // SH + 1,
            (W - KW + 2 * PW) // SW + 1
            ))
        N, C, H_O, W_O = output.shape
        for i in range(N):
            for j in range(C):
                for h in range(0, H - KH + 1, SH):
                    for w in range(0, W - KW + 1, SW):
                        output[i, j, h // SH, w // SW] = np.max(data[
                            i, 
                            j, 
                            h : h + KH, 
                            w : w + KW
                            ])
        ctx.save_for_backward(tensor, kernel_size, stride, padding)
        return Tensor(output)
    
    @staticmethod
    def backward(ctx, grad_output):
        tensor, kernel_size, stride, padding = ctx.saved_tensors
        N, C, H, W = tensor.shape
        N, C, H_O, W_O = grad_output.shape
        
        KH, KW = kernel_size
        SH, SW = stride
        PH, PW = padding
        
        grad = zeros(tensor.shape)
        for i in range(N):
            for j in range(C):
                for h in range(0, H - KH + 1, SH):
                    for w in range(0, W - KW + 1, SW):
                        mask = tensor[i, j, h : h + KH, w : w + KW] == max(tensor[i, j, h : h + KH, w : w + KW])
                        grad[i, j, h : h + KH, w : w + KW] += mask * grad_output[i, j, h // SH, w // SW]
        
        return grad[:, :, PH: H-PH, PW: W-PW], None, None, None


def im2col(image, KH, KW, stride):
    # image is a 4d tensor([N, channel, H, W])
    image_col = []
    SH, SW = stride
    
    for i in range(0, image.shape[2] - KH + 1, SH):
        for j in range(0, image.shape[3] - KW + 1, SW):
            col = image[:, :, i:i + KH, j:j + KW].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col
    

class Conv2d(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        input, weight = input.data, weight.data
        N, C_i, H, W = input.shape
        C_O, C_i, KH, KW = weight.shape
        
        SH, SW = stride
        PH, PW = padding
        
        col_weight = weight.reshape((C_O, -1))
        input = np.pad(input, ((0, 0), (0, 0), (PH, PH), (PW, PW)), 'constant', constant_values=0)
        conv_out = np.zeros(
            (N, 
            C_O, 
            (H - KH + 2 * PH) // SH + 1,
            (W - KW + 2 * PW) // SW + 1
            ))
        col_image = []
        for i in range(N):
            img_i = input[i][np.newaxis, :]
            col_image_i = im2col(img_i, KH, KW, stride)
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
        N, C_O, H_O, W_O = grad_output.shape
        N, C_i, H, W = input_shape
        C_O, C_i, KH, KW = weight_shape
        SH, SW = stride
        PH, PW = padding
        
        # init gradient for img2col
        col_weight_gradient = zeros(col_weight.shape)
        conv_out_gradient = grad_output.reshape(N, C_O, -1)
        
        # init gradient for input tensor
        bias_gradient = ones(C_O) if bias is None else None
        input_gradient = zeros(input_shape)
    
        for i in range(N):
            col_image_gradient = mm(transpose(conv_out_gradient[i]), col_weight, None)
            col_weight_gradient += mm(conv_out_gradient[i], col_image[i], None)
            
            j = 0
            for h in range(0, H - KH + 1, SH):
                for w in range(0, W - KW + 1, SW):
                    input_gradient[i, :, h: h + KH, w: w + KW] += col_image_gradient[j].reshape((C_i, KH, KW))
                    j += 1
        
        weight_gradient = col_weight_gradient.reshape(C_O, C_i, KH, KW)
        # remove padding
        input_gradient = input_gradient[:, :, PH: H-PH, PW: W-PW]
        return input_gradient, weight_gradient, bias_gradient, None, None


class View(autograd.Function):
    @staticmethod
    def forward(ctx, tensor, shape):
        ctx.save_for_backward(tensor.shape)
        new_tensor = tensor.reshape(shape)
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        original_shape, = ctx.saved_tensors
        grad = grad_output.reshape(original_shape)
        return grad

class Transpose(autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        new_tensor = Tensor(np.transpose(tensor.data))
        return new_tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        grad = Tensor(np.transpose(grad_output.data))
        return grad
    
class LogSoftmax(autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim):
        # tensor size is (N, C)
        data = tensor.data
        data_shift = data - np.amax(data, axis=dim, keepdims=True)
        data_shift_exp = np.exp(data_shift)
        exp_sum = np.sum(data_shift_exp, axis=dim, keepdims=True)
        # exp_sum[exp_sum == 0] = 1e-10
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
            grad[i] = mm(transpose(jac), grad_output[i], None)
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


class Dropout(autograd.Function):
    @staticmethod
    def forward(ctx, input, training, p, inplace):
        mask = None
        if inplace:
            output = input
        else:
            output = input.copy()
        
        if training:
            mask = np.random.binomial(1, p, size=input.shape)
            output[mask == 1] = 0
            output /= (1 - p)
        ctx.save_for_backward(mask, training, p)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        mask, training, p = ctx.saved_tensors
        if training:
            grad_output[mask == 1] = 0
            grad_output /= (1 - p) 
        return grad_output, None, None


class BatchNorm(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, eps, momentum, is_training):
        # input is (N, C, H, W) or (N, C)
        running_mean_ = running_mean
        running_var_ = running_var
        
        input = input.data
        weight = weight.data
        bias = bias.data
        running_mean = running_mean.data
        running_var = running_var.data
        
        num_features = weight.shape[0]
        dim = len(input.shape)
        
        if dim == 2:
            shape_descriptor = (1, num_features)
            axis = 0
        elif dim == 3:
            shape_descriptor = (1, num_features, 1)
            axis = (0, 2)
        else:
            shape_descriptor = (1, num_features, 1, 1)
            axis = (0, 2, 3)

        weight = weight.reshape(shape_descriptor)
        bias = bias.reshape(shape_descriptor)
        running_mean = running_mean.reshape(shape_descriptor)
        running_var = running_var.reshape(shape_descriptor)
        
        if is_training:
            mean = np.mean(input, axis=axis, keepdims=True)
            var = np.mean((input - mean) ** 2, axis=axis, keepdims=True) # biased
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_var = (1 - momentum) * running_var + momentum * var
            
            input_hat = (input - mean) / np.sqrt(var + eps)
            output = input_hat * weight + bias
            ctx.save_for_backward(is_training, input, weight, input_hat, mean, var, eps)
        else:
            input_hat = (input - running_mean) / np.sqrt(running_var + eps)
            output = input_hat * weight + bias
            ctx.save_for_backward(is_training, None, None, None, None, None, None, None)
        
        running_mean_.data = running_mean.reshape(num_features)
        running_var_.data = running_var.reshape(num_features)
        
        return Tensor(output)
    
    @staticmethod
    def backward(ctx, grad_output):
        is_training, input, weight, input_hat, mean, var, eps = ctx.saved_tensors
        grad_output = grad_output.data
        
        dim = len(input.shape)
        if dim == 2:
            axis = 0
            N = input.shape[0]
        elif dim == 3:
            axis = (0, 2)
            N = input.shape[0] * input.shape[2]
        else:
            axis = (0, 2, 3)
            N = input.shape[0] * input.shape[2] * input.shape[3]
        
        if is_training:
            weight_grad = np.sum(input_hat * grad_output, axis=axis)
            bias_grad = np.sum(grad_output, axis=axis)
            input_hat_grad = grad_output * weight
            var_grad = -0.5 * np.sum(input_hat_grad * (input - mean), axis=axis, keepdims=True) * np.power(var + eps, -1.5)
            mean_grad = -np.sum(input_hat_grad / np.sqrt(var + eps), axis=axis, keepdims=True) - 2 * var_grad * np.sum(input - mean, axis=axis, keepdims=True) / N
            input_grad = input_hat_grad / np.sqrt(var + eps) + 2.0 * var_grad * (input - mean) / N + mean_grad / N
            return Tensor(input_grad), Tensor(weight_grad), Tensor(bias_grad), None, None, None, None, None
        else:
            return Tensor(grad_output), None, None, None, None, None, None, None
    
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
transpose = Transpose.apply
nll_loss = NllLoss.apply
dropout = Dropout.apply
batchnorm = BatchNorm.apply