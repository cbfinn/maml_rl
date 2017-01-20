import numpy as np

from rllab.misc import ext
#from sandbox.rocky.tf.core.parameterized import Parameterized
#from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
#from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.spaces.box import Box

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian # This is just a util class. No params.
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
from sandbox.rocky.tf.misc import tensor_utils
import itertools

import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers

# TODO - what does this mean?
load_params = True




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
        raise NotImplementedError('Chelsea does not support this.')
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
        input = tf.reshape(input, tf.pack([tf.shape(input)[0], -1]))
    activation = tf.matmul(input, W)
    if b is not None:
        activation = activation + tf.expand_dims(b, 0)

    if batch_norm:
        return tf_layers.batch_norm(activation, activation_fn=nonlinearity, reuse=reuse, scope=scope, is_training=is_training)
    else:
        return nonlinearity(activation)

def make_param_layer(num_units, name='', param=tf.zeros_initializer, trainable=True):
    param = add_param(param, (num_units,), layer_name=name, name='param', trainable=trainable)
    return param

def forward_param_layer(input, param):
    ndim = input.get_shape().ndims
    param = tf.convert_to_tensor(param)
    num_units = int(param.get_shape()[0])
    reshaped_param = tf.reshape(param, (1,)*(ndim-1)+(num_units,))
    tile_arg = tf.concat(0, [tf.shape(input)[:ndim-1], [1]])
    tiled = tf.tile(reshaped_param, tile_arg)
    return tiled
### End Helper functions ###


class SensitiveGaussianMLPPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=tf.nn.tanh,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.identity,
            mean_network=None,
            std_network=None,
            std_parametrization='exp',
            grad_step_size=1.0,
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :param grad_step_size: the step size taken in the learner's gradient update
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.input_shape = (None, obs_dim,)
        self.step_size = grad_step_size

        # create network
        if mean_network is None:
            self.all_params = self.create_MLP(
                name="mean_network",
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
            )
            self.input_tensor, _ = self.forward_MLP('mean_network', self.all_params,
                reuse=None # Need to run this for batch norm
            )
            forward_mean = lambda x, params, is_train: self.forward_MLP('mean_network', params,
                input_tensor=x, is_training=is_train)[1]
        else:
            raise NotImplementedError('Chelsea does not support this.')

        if std_network is not None:
            raise NotImplementedError('Contained Gaussian MLP does not support this.')
        else:
            if adaptive_std:
                raise NotImplementedError('Chelsea does not support this.')
                ## NOTE - this isn't tested.
                ## NOTE - there is code that assumes that there is only one MLP (e.g. self.n_hidden)
                #self.std_params = std_params = self.create_MLP(
                    #name="std_network",
                    #input_shape=(None, obs_dim,),
                    #output_dim=action_dim,
                    #hidden_sizes=std_hidden_sizes,
                #)
                #forward_std = lambda x: self.forward_MLP('std_network', std_params,
                                                                  #hidden_nonlinearity=std_hidden_nonlinearity,
                                                                #output_nonlinearity=tf.identity,
                                                                #input_tensor=x)[1]
            else:
                if std_parametrization == 'exp':
                    init_std_param = np.log(init_std)
                elif std_parametrization == 'softplus':
                    init_std_param = np.log(np.exp(init_std) - 1)
                else:
                    raise NotImplementedError
                self.all_params['std_param'] = make_param_layer(
                    num_units=action_dim,
                    param=tf.constant_initializer(init_std_param),
                    name="output_std_param",
                    trainable=learn_std,
                )
                forward_std = lambda x, params: forward_param_layer(x, params['std_param'])

            # unify forward mean and forward std into a single function
            self._forward = lambda obs, params, is_train: (
                    forward_mean(obs, params, is_train), forward_std(obs, params))

            self.std_parametrization = std_parametrization

            if std_parametrization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parametrization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            self._dist = DiagonalGaussian(action_dim)

            self._cached_params = {}

            super(SensitiveGaussianMLPPolicy, self).__init__(env_spec)

            dist_info_sym = self.dist_info_sym(self.input_tensor, dict(), is_training=False)
            mean_var = dist_info_sym["mean"]
            log_std_var = dist_info_sym["log_std"]

            # before sensitive update
            self._init_f_dist = tensor_utils.compile_function(
                inputs=[self.input_tensor],
                outputs=[mean_var, log_std_var],
            )
            self._cur_f_dist = self._init_f_dist



    @property
    def vectorized(self):
        return True

    def set_init_surr_obj(self, input_list, surr_objs_tensor):
        self.input_list_for_grad = input_list
        self.surr_objs = surr_objs_tensor


    def compute_updated_dists(self, samples):
        """ Compute fast gradients once and pull them out of tensorflow for sampling.
        """
        self._updated_f_dists = {}
        num_tasks = len(samples)
        param_keys = self.all_params.keys()
        sess = tf.get_default_session()

        obs_list, action_list, adv_list = [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(samples[i],
                    'observations', 'actions', 'advantages')
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])

        inputs = obs_list + action_list + adv_list

        all_fast_params_tensor = []
        for i in range(num_tasks):
            gradients = dict(zip(param_keys, tf.gradients(self.surr_objs[i], self.all_params.values())))
            fast_params_tensor = dict(zip(param_keys, [self.all_params[key] - self.step_size*gradients[key] for key in param_keys]))
            all_fast_params_tensor.append(fast_params_tensor)
        fast_params_per_task = sess.run(all_fast_params_tensor, feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))

        for i in range(num_tasks):

            info = self.dist_info_sym(self.input_tensor, dict(), all_params=fast_params_per_task[i],
                    is_training=False)

            # before sensitive update
            self._updated_f_dists[i] = tensor_utils.compile_function(
                inputs=[self.input_tensor],
                outputs=[info['mean'], info['log_std']],
            )

        self._cur_f_dist = self._updated_f_dists

    def switch_to_init_dist(self):
        self._cur_f_dist = self._init_f_dist

    def dist_info_sym(self, obs_var, state_info_vars=None, all_params=None, is_training=True):
        # This function constructs the tf graph, only called during beginning of training
        # obs_var - observation tensor
        # mean_var - tensor for policy mean
        # std_param_var - tensor for policy std before output
        #mean_var, std_param_var = L.get_output([self._l_mean, self._l_std_param], obs_var)
        if all_params is None:
            all_params = self.all_params

        mean_var, std_param_var = self._forward(obs_var, all_params, is_training)
        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        if self.std_parametrization == 'exp':
            log_std_var = std_param_var
        elif self.std_parametrization == 'softplus':
            log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var)))
        else:
            raise NotImplementedError
        return dict(mean=mean_var, log_std=log_std_var)

    def updated_dist_info_sym(self, init_obs_var, init_surr_obj, new_obs_var, is_training=True):
        # symbolically create sensitive learning graph, for the meta-optimization
        init_params = self.all_params

        param_keys = init_params.keys()
        gradients = dict(zip(param_keys, tf.gradients(init_surr_obj, init_params.values())))
        fast_params_tensor = dict(zip(param_keys, [init_params[key] - self.step_size*gradients[key] for key in param_keys]))

        return self.dist_info_sym(new_obs_var, all_params=fast_params_tensor, is_training=is_training)


    @overrides
    def get_action(self, observation, idx=None):
        import pdb; pdb.set_trace()
        # this function takes a numpy array observations and outputs randomly sampled actions.
        # idx: index corresponding to the task/updated policy.
        flat_obs = self.observation_space.flatten(observation)
        if type(self._cur_f_dist) == dict:
            assert idx is not None
            f_dist = self._cur_f_dist[idx]
        else:
            f_dist = self._cur_f_dist
        mean, log_std = [x[0] for x in f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        # this function takes a numpy array observations and outputs randomly sampled actions.
        # Assumes that there is one observation per distr, if distr is a dict
        flat_obs = self.observation_space.flatten_n(observations)

        if type(self._cur_f_dist) == dict:
            # It could also be a single number, but I don't think we do that.
            num_tasks = len(self._cur_f_dist)
            assert flat_obs.shape[0] == num_tasks

            means = np.zeros((num_tasks, 2)) # a hack - don't hard code 2
            log_stds = np.zeros((num_tasks, 2))
            for i in range(num_tasks):
                means[i:i+1], log_stds[i:i+1] = self._cur_f_dist[i](flat_obs[i:i+1,:])
        else:
            means, log_stds = self._cur_f_dist(flat_obs) # _f_dist runs the network forward.

        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    @property
    def distribution(self):
        return self._dist

    def get_params_internal(self, **tags):
        if tags.get('trainable', False):
            return tf.trainable_variables()
        else:
            return tf.all_variables()

        if regularizable in tags.keys():
            import pdb; pdb.set_trace()

    # This makes all of the parameters.
    def create_MLP(self, name, output_dim, hidden_sizes,
                   hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                   output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer,
                   weight_normalization=False,
                   ):
        cur_shape = self.input_shape
        with tf.variable_scope(name):
            all_params = {}
            for idx, hidden_size in enumerate(hidden_sizes):
                W, b, cur_shape = make_dense_layer(
                    cur_shape,
                    num_units=hidden_size,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_norm=weight_normalization,
                )
                all_params['W' + str(idx)] = W
                all_params['b' + str(idx)] = b
            W, b, _ = make_dense_layer(
                cur_shape,
                num_units=output_dim,
                name='output',
                W=output_W_init,
                b=output_b_init,
                weight_norm=weight_normalization,
            )
            all_params['W' + str(len(hidden_sizes))] = W
            all_params['b'+str(len(hidden_sizes))] = b

            return all_params

    def forward_MLP(self, name, all_params, input_tensor=None,
                    batch_normalization=False, reuse=True, is_training=False):
        # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
        # set reuse to False if the first time this func is called.
        with tf.variable_scope(name):
            if input_tensor is None:
                l_in = make_input(shape=self.input_shape, input_var=None, name='input')
            else:
                l_in = input_tensor
            l_hid = l_in
            for idx in range(self.n_hidden):
                l_hid = forward_dense_layer(l_hid, all_params['W'+str(idx)], all_params['b'+str(idx)],
                                            batch_norm=batch_normalization,
                                            nonlinearity=self.hidden_nonlinearity,
                                            scope=str(idx), reuse=reuse,
                                            is_training=is_training
                                            )
            output = forward_dense_layer(l_hid, all_params['W'+str(self.n_hidden)], all_params['b'+str(self.n_hidden)],
                                         batch_norm=False, nonlinearity=self.output_nonlinearity,
                                         )
            return l_in, output

    def get_params(self, **tags):
        """
        Get the list of parameters (symbolically), filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(**tags)
        return self._cached_params[tag_tuple]

    def get_param_values(self, **tags):
        params = self.get_params(**tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def log_diagnostics(self, paths, prefix=''):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular(prefix+'AveragePolicyStd', np.mean(np.exp(log_stds)))

    #### CODE NOT USED AFTER HERE ####
    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        # Not used
        import pdb; pdb.set_trace()
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def get_param_dtypes(self, **tags):
        # Not used.
        import pdb; pdb.set_trace()
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, **tags):
        # Not used.
        import pdb; pdb.set_trace()
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(**tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def set_param_values(self, flattened_params, **tags):
        # Not used.
        import pdb; pdb.set_trace()
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(**tags))
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(**tags),
                self.get_param_dtypes(**tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, **tags):
        # Not used.
        import pdb; pdb.set_trace()
        return unflatten_tensors(flattened_params, self.get_param_shapes(**tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.initialize_variables(self.get_params()))
            self.set_param_values(d["params"])


