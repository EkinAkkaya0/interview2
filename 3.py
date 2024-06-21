import numpy as np

# Generate a random 5x5 grayscale image with a single channel
input_image = np.random.rand(5, 5, 1)
print("Input Image:\n", input_image)
def conv2d(image, kernel, stride=2, padding='same'):
    # Add zero padding
    if padding == 'same':
        pad_h = (kernel.shape[0] - 1) // 2
        pad_w = (kernel.shape[1] - 1) // 2
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    # Calculate output dimensions
    out_h = (image.shape[0] - kernel.shape[0]) // stride + 1
    out_w = (image.shape[1] - kernel.shape[1]) // stride + 1
    out_channels = kernel.shape[-1]

    # Initialize output
    output = np.zeros((out_h, out_w, out_channels))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            for k in range(out_channels):
                vert_start = i * stride
                vert_end = vert_start + kernel.shape[0]
                horiz_start = j * stride
                horiz_end = horiz_start + kernel.shape[1]
                output[i, j, k] = np.sum(image[vert_start:vert_end, horiz_start:horiz_end, :] * kernel[:, :, :, k])

    return output

# Define random kernels for convolution
kernel1 = np.random.rand(2, 2, 1, 2)
kernel2 = np.random.rand(2, 2, 2, 2)

# First convolution
conv1_output = conv2d(input_image, kernel1)
print("After First Convolution:\n", conv1_output)
def max_pooling(image, pool_size=(2, 2), stride=2):
    out_h = (image.shape[0] - pool_size[0]) // stride + 1
    out_w = (image.shape[1] - pool_size[1]) // stride + 1
    out_channels = image.shape[-1]

    output = np.zeros((out_h, out_w, out_channels))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(out_channels):
                vert_start = i * stride
                vert_end = vert_start + pool_size[0]
                horiz_start = j * stride
                horiz_end = horiz_start + pool_size[1]
                output[i, j, k] = np.max(image[vert_start:vert_end, horiz_start:horiz_end, k])

    return output

def average_pooling(image, pool_size=(2, 2), stride=2):
    out_h = (image.shape[0] - pool_size[0]) // stride + 1
    out_w = (image.shape[1] - pool_size[1]) // stride + 1
    out_channels = image.shape[-1]

    output = np.zeros((out_h, out_w, out_channels))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(out_channels):
                vert_start = i * stride
                vert_end = vert_start + pool_size[0]
                horiz_start = j * stride
                horiz_end = horiz_start + pool_size[1]
                output[i, j, k] = np.mean(image[vert_start:vert_end, horiz_start:horiz_end, k])

    return output

# Max pooling after first convolution
max_pool_output = max_pooling(conv1_output)
print("After Max Pooling:\n", max_pool_output)

# Second convolution
conv2_output = conv2d(max_pool_output, kernel2)
print("After Second Convolution:\n", conv2_output)

# Average pooling after second convolution
avg_pool_output = average_pooling(conv2_output)
print("After Average Pooling:\n", avg_pool_output)
# The output after average pooling is the final output of the model
final_output = avg_pool_output.flatten()
print("Final Output:\n", final_output)
