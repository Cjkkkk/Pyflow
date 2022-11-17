import math
import numpy as np

def calculate_fan_in_and_fan_out(shape):
    if len(shape) < 2:
        raise RuntimeError("Need at least 2 dim to calculate fan in and fan out")
    fan_in, fan_out = shape[0], shape[1]
    receptive_field_size = 1
    for s in shape[2:]:
        receptive_field_size *= s
    return fan_in * receptive_field_size, fan_out * receptive_field_size


def rand_(shape, low, high):
    return np.random.rand(*shape) * (high - low) + low

def randn_(shape, mean, std): 
    return np.random.randn(*shape) * std + mean

def xavier_normal(shape, nonlinearity='leaky_relu'):
    fan_in, fan_out = calculate_fan_in_and_fan_out(shape)
    gain = 1
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn_(shape, 0, std)
    
def xavier_uniform(shape, nonlinearity='leaky_relu'):
    fan_in, fan_out = calculate_fan_in_and_fan_out(shape)
    gain = 1
    bound = gain * math.sqrt(6 / (fan_in + fan_out))
    high, low = bound, -bound
    return rand_(shape, low, high)

def kaiming_normal(shape, nonlinearity='leaky_relu'):
    fan_in, fan_out = calculate_fan_in_and_fan_out(shape)
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    return randn_(shape, 0, std)

def kaiming_uniform(shape, nonlinearity='leaky_relu'):
    fan_in, fan_out = calculate_fan_in_and_fan_out(shape)
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    high, low = bound, -bound
    return rand_(shape, low, high)