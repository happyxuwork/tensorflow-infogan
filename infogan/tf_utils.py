import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorflow.examples.tutorials.mnist import input_data
from infogan.misc_utils import parse_math

def variables_in_current_scope():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)


def scope_variables(name):
    with tf.variable_scope(name):
        return variables_in_current_scope()


def leaky_rectify(x, leakiness=0.01):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    return ret


def identity(x):
    return x

# 公式如下
#y=γ(x-μ)/σ+β
# 其中x是输入，y是输出，μ是均值，σ是方差，γ和β是缩放（scale）、偏移（offset）系数。
# 一般来讲，这些参数都是基于channel来做的，比如输入x是一个16*32*32*128(NWHC格式)的feature map，那么上述参数都是128维的向量。
# 其中γ和β是可有可无的，有的话，就是一个可以学习的参数（参与前向后向），没有的话，就简化成y=(x-μ)/σ。而μ和σ，
# 在训练的时候，使用的是batch内的统计值，测试/预测的时候，采用的是训练时计算出的滑动平均值。
# https://www.jianshu.com/p/0312e04e4e83这里的解释很好，特别是对四维数组进行均值和方差求解过程有了一个具体的说明，
# https://www.cnblogs.com/hrlnw/p/7227447.html给出了具体的归一化公式
# y=γ(x-μ)/σ+β
# 其中x是输入，y是输出，μ是均值，σ是方差，γ和β是缩放（scale）、偏移（offset）系数。
# tf.nn.moments()和tf.nn.batch_norm_with_global_normalization(）结合起来进行卷积层的batch_normal操作
# 但是对于 [128, 32, 32, 64] 这样的4维矩阵，理解就有点困难了。
# 其实很简单，可以这么理解，一个batch里的128个图，经过一个64 kernels卷积层处理，得到了128×64个图，
# 再针对每一个kernel所对应的128个图，求它们所有像素的mean和variance，因为总共有64个kernels，输出的结果就是一个一维长度64的数组啦！
def conv_batch_norm(inputs,
                    name="batch_norm",
                    is_training=True,
                    trainable=True,
                    epsilon=1e-5):
    # 使用的是==滑动平均的方法更新参数
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    # get_shape() of the last dimension
    shp = inputs.get_shape()[-1].value
    print("this is a debug information==输入中的get_shape()[-1]的值为: "+str(shp))
    with tf.variable_scope(name) as scope:
        # 函数原型：tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
        # 返回一个生成具有正态分布的张量的初始化器。
        gamma = tf.get_variable("gamma", [shp], initializer=tf.random_normal_initializer(1., 0.02), trainable=trainable)
        beta = tf.get_variable("beta", [shp], initializer=tf.constant_initializer(0.), trainable=trainable)

       #  tf.nn.moments(inputs, [0, 1, 2])==对四维向量进行均值和方差的计算，其中返回depth维度的tensor
       # * for so-called "global normalization", used with convolutional filters with
       #   shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
       # * for simple batch normalization pass `axes=[0]` (batch only).
        mean, variance = tf.nn.moments(inputs, [0, 1, 2])
        # tensor.set_shape()重新设置tensor的形状
        mean.set_shape((shp,))
        variance.set_shape((shp,))
        ema_apply_op = ema.apply([mean, variance])

        def update():
            with tf.control_dependencies([ema_apply_op]):
                # tf.nn.batch_norm_with_global_normalization===提供了进行批量归一化的操作。
                print("this is a debug information==进行update操作--mean和variance的值为: "+str(mean)+",,"+str(variance))
                return tf.nn.batch_norm_with_global_normalization(
                    inputs, mean, variance, beta, gamma, epsilon,
                    scale_after_normalization=True
                )
        def do_not_update():
            return tf.nn.batch_norm_with_global_normalization(
                inputs, ema.average(mean), ema.average(variance), beta,
                gamma, epsilon,
                scale_after_normalization=True
            )
        # 相当于c语言中的if...else...
        normalized_x = tf.cond(
            is_training,
            update,
            do_not_update
        )
        return normalized_x

NONLINEARITY_NAME_TO_F = {
    'lrelu': leaky_rectify,
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
    'identity': tf.identity,
}


def parse_conv_params(params):
    nonlinearity = 'relu'
    if len(params) == 4:
        params, nonlinearity = params[:-1], params[-1]
    nkernels, stride, num_outputs = [parse_math(p) for p in params]

    return nkernels, stride, num_outputs, nonlinearity

# 网络的结构
# 这里的inpt为噪声的维度(GAN中为[-1,62]，InfoGAN中为[-1,74])，
# string为网络的描述信息--fc:1024,fc:7x7x128,reshape:7:7:128,deconv:4:2:64,deconv:4:2:1:sigmoid
# discriminator_desc==conv:4:2:64:lrelu,conv:4:2:128:lrelu,fc:1024:lrelu
# is_training=True表示需要训练生成器
# strip_batchnorm_from_last_layer=True
def run_network(inpt, string, is_training, use_batch_norm, debug=False, strip_batchnorm_from_last_layer=False):
    # 下面两步没有看懂，？？？？？？？？？
    # 如果use_batch_norm时，下面的两个都有效
    # layers.bacth_norm是对输出的全连接层进行批归一化的
    # 而conv_batch_norm是对卷积层进行批归一化的
    maybe_fc_batch_norm   = layers.batch_norm if use_batch_norm else None
    maybe_conv_batch_norm = conv_batch_norm if use_batch_norm else None

    if debug:
        print ("%s architecture" % (tf.get_variable_scope().name,))

    layer_idx = 0

    out = inpt
    layer_strs = string.split(",")
    for i, layer in enumerate(layer_strs):
        # 最后一层跳过batch_norm
        if i + 1 == len(layer_strs) and strip_batchnorm_from_last_layer:
            maybe_fc_batch_norm   = None
            maybe_conv_batch_norm = None

        # 如果为卷积层，进行卷积操作
        if layer.startswith("conv:"):
            nkernels, stride, num_outputs, nonlinearity_str = parse_conv_params(layer[len("conv:"):].split(":"))
            nonlinearity = NONLINEARITY_NAME_TO_F[nonlinearity_str]

            out = layers.convolution2d(
                out,
                num_outputs=num_outputs,
                kernel_size=nkernels,
                stride=stride,
                normalizer_params={"is_training": is_training},
                normalizer_fn=maybe_conv_batch_norm,
                activation_fn=nonlinearity,
                scope='layer_%d' % (layer_idx,)
            )
            layer_idx += 1

            if debug:
                print ("Convolution with nkernels=%d, stride=%d, num_outputs=%d followed by %s" %
                        (nkernels, stride, num_outputs, nonlinearity_str))
        # 如果为反卷积层，进行反卷积操作
        #这个操作的具体过程可以查看http://blog.csdn.net/fate_fjh/article/details/52882134说得非常好
        #大致的过程就是如果输入为N1*N1，卷积核为N2*N2，步长为K，则输出为（N1-1）*K+N2
        elif layer.startswith("deconv:"):
            nkernels, stride, num_outputs, nonlinearity_str = parse_conv_params(layer[len("deconv:"):].split(":"))
            nonlinearity = NONLINEARITY_NAME_TO_F[nonlinearity_str]

            out = layers.convolution2d_transpose(
                out,
                num_outputs=num_outputs,
                kernel_size=nkernels,
                stride=stride,
                activation_fn=nonlinearity,
                normalizer_fn=maybe_conv_batch_norm,
                normalizer_params={"is_training": is_training},
                scope='layer_%d' % (layer_idx,)
            )
            layer_idx += 1
            if debug:
                print ("Deconvolution with nkernels=%d, stride=%d, num_outputs=%d followed by %s" %
                        (nkernels, stride, num_outputs, nonlinearity_str))
        # 如果为全连接层，进行全连接操作
        # 这里的全连接是tensorflow自己集成的，可以看一下内部的说明：大致的意思是
        # 如果使用了batch_norm的话，就没有bias，并且默认的激活函数是relu
        elif layer.startswith("fc:"):
            params = layer[len("fc:"):].split(":")
            nonlinearity_str = 'relu'
            if len(params) == 2:
                params, nonlinearity_str = params[:-1], params[-1]
            num_outputs = parse_math(params[0])
            nonlinearity = NONLINEARITY_NAME_TO_F[nonlinearity_str]

            out = layers.fully_connected(
                out,
                num_outputs=num_outputs,
                activation_fn=nonlinearity,
                normalizer_fn=maybe_fc_batch_norm,
                normalizer_params={"is_training": is_training, "updates_collections": None},
                scope='layer_%d' % (layer_idx,)
            )
            layer_idx += 1
            if debug:
                print ("Fully connected with num_outputs=%d followed by %s" %
                        (num_outputs, nonlinearity_str))
        # 如果为重构层，进行重构操作
        elif layer.startswith("reshape:"):
            params = layer[len("reshape:"):].split(":")
            dims = [parse_math(dim) for dim in params]
            out = tf.reshape(out, [-1] + dims)
            if debug:
                print("Reshape to %r" % (dims,))
        else:
            raise ValueError("Could not parse layer description: %r" % (layer,))
    if debug:
        print("")
    return out



def load_mnist_dataset():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    pixel_height = 28
    pixel_width = 28
    n_channels = 1
    for dset in [mnist.train, mnist.validation, mnist.test]:
        num_images = len(dset.images)
        dset.images.shape = (num_images, pixel_height, pixel_width, n_channels)
    return mnist.train.images


try:
    NOOP = tf.noop
except:
    # this changed for no reason in latest version. Danke!
    NOOP = tf.no_op()
