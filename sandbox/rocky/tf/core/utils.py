import numpy as np
import tensorflow as tf
import sandbox.rocky.tf.core.layers as L

### Start Helper functions ###
def make_input(shape, input_var=None, name="input", **kwargs):
    if input_var is None:
        if name is not None:
            with tf.variable_scope(name):
                    input_var = tf.placeholder(tf.float32, shape=shape, name="input")
        else:
            input_var = tf.placeholder(tf.float32, shape=shape, name="input")
    return input_var

def _create_param(spec, shape, name, trainable=True, regularizable=True):
    if not hasattr(spec, '__call__'):
        assert isinstance(spec, (tf.Tensor, tf.Variable))
        return spec
    assert hasattr(spec, '__call__')
    if regularizable:
        regularizer = None
    else:
        regularizer = lambda _: tf.constant(0.)
    return tf.get_variable(
        name=name, shape=shape, initializer=spec, trainable=trainable,
        regularizer=regularizer, dtype=tf.float32
    )

def add_param(spec, shape, layer_name, name, weight_norm=None, variable_reuse=None, **tags):
    with tf.variable_scope(layer_name, reuse=variable_reuse):
        tags['trainable'] = tags.get('trainable', True)
        tags['regularizable'] = tags.get('regularizable', True)
        param = _create_param(spec, shape, name, **tags)
    if weight_norm:
        raise NotImplementedError('Not supported.')
    return param

def make_dense_layer(input_shape, num_units, name='fc', W=L.XavierUniformInitializer(), b=tf.zeros_initializer, weight_norm=False, **kwargs):
    # make parameters
    num_inputs = int(np.prod(input_shape[1:]))
    W = add_param(W, (num_inputs, num_units), layer_name=name, name='W', weight_norm=weight_norm)
    if b is not None:
        b = add_param(b, (num_units,), layer_name=name, name='b', regularizable=False, weight_norm=weight_norm)
    output_shape = (input_shape[0], num_units)
    return W,b, output_shape

def forward_dense_layer(input, W, b, nonlinearity=tf.identity, batch_norm=False, scope='', reuse=True, is_training=False):
    # compute output tensor
    if input.get_shape().ndims > 2:
        # if the input has more than two dimensions, flatten it into a
        # batch of feature vectors.
        input = tf.reshape(input, tf.stack([tf.shape(input)[0], -1]))
    activation = tf.matmul(input, W)
    if b is not None:
        activation = activation + tf.expand_dims(b, 0)

    if batch_norm:
        raise NotImplementedError('not supported')
    else:
        return nonlinearity(activation)

def make_param_layer(num_units, name='', param=tf.zeros_initializer(), trainable=True):
    param = add_param(param, (num_units,), layer_name=name, name='param', trainable=trainable)
    return param

def forward_param_layer(input, param):
    ndim = input.get_shape().ndims
    param = tf.convert_to_tensor(param)
    num_units = int(param.get_shape()[0])
    reshaped_param = tf.reshape(param, (1,)*(ndim-1)+(num_units,))
    tile_arg = tf.concat([tf.shape(input)[:ndim-1], [1]], 0)
    tiled = tf.tile(reshaped_param, tile_arg)
    return tiled
### End Helper functions ###


