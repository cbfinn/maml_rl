import numpy as np

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

# TODO - what does this mean?
load_params = True

class GaussianMLPPolicy(StochasticPolicy, Serializable):
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
            output_nonlinearity=None,
            mean_network=None,
            std_network=None,
            std_parametrization='exp'
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
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        with tf.variable_scope(name):

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            # create network
            if mean_network is None:
                # returns layers(), input_layer, output_layer, input_var, output
                mean_network = self.create_MLP(
                    name="mean_network",
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                )
            else:
                raise NotImplementedError('Chelsea does not support this.')
            l_mean = mean_network[2]
            obs_var = mean_network[3]

            if std_network is not None:
                raise NotImplementedError('Contained Gaussian MLP does not support this.')
                l_std_param = std_network.output_layer
            else:
                if adaptive_std:
                    # returns layers(), input_layer, output_layer, input_var, output
                    std_network = self.create_MLP(
                        name="std_network",
                        input_shape=(obs_dim,),
                        input_layer=mean_network[1],
                        output_dim=action_dim,
                        hidden_sizes=std_hidden_sizes,
                        hidden_nonlinearity=std_hidden_nonlinearity,
                        output_nonlinearity=None,
                    )
                    l_std_param = std_network[2]
                else:
                    if std_parametrization == 'exp':
                        init_std_param = np.log(init_std)
                    elif std_parametrization == 'softplus':
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError
                    l_std_param = L.ParamLayer(
                        mean_network[1],
                        num_units=action_dim,
                        param=tf.constant_initializer(init_std_param),
                        name="output_std_param",
                        trainable=learn_std,
                    )

            self.std_parametrization = std_parametrization

            if std_parametrization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parametrization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            self._l_mean = l_mean
            self._l_std_param = l_std_param

            self._dist = DiagonalGaussian(action_dim)

            # LayersPowered.__init__(self, [l_mean, l_std_param])
            self._output_layers = [l_mean, l_std_param]
            #self._input_layers = None
            # Parameterized.__init__(self)
            self._cached_params = {}
            #self._cached_param_dtypes = {} #self._cached_param_shapes = {} #self._cached_assign_ops = {} #self._cached_assign_placeholders = {}

            super(GaussianMLPPolicy, self).__init__(env_spec)

            dist_info_sym = self.dist_info_sym(mean_network[3], dict())
            mean_var = dist_info_sym["mean"]
            log_std_var = dist_info_sym["log_std"]

            def f_dist(*input_vals):
                sess = tf.get_default_session()
                return sess.run([mean_var, log_std_var],
                                feed_dict=dict(list(zip([obs_var], input_vals))))
            self._f_dist = f_dist

            #self._f_dist = tensor_utils.compile_function(
            #    inputs=[obs_var],
            #    outputs=[mean_var, log_std_var],
            #)

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None):
        # This function constructs the tf graph, only called during beginning of training
        # obs_var - observation tensor
        # mean_var - tensor for policy mean
        # std_param_var - tensor for policy std before output
        mean_var, std_param_var = L.get_output([self._l_mean, self._l_std_param], obs_var)
        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        if self.std_parametrization == 'exp':
            log_std_var = std_param_var
        elif self.std_parametrization == 'softplus':
            log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var)))
        else:
            raise NotImplementedError
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        means, log_stds = self._f_dist(flat_obs) # _f_dist runs the network forward.
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    @property
    def distribution(self):
        return self._dist

    def get_params_internal(self, **tags):
        layers = L.get_all_layers(self._output_layers, treat_as_input=None)
        params = itertools.chain.from_iterable(l.get_params(**tags) for l in layers)
        return L.unique(params)

    def create_MLP(self, name, output_dim, hidden_sizes, hidden_nonlinearity,
                   output_nonlinearity, hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                   output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer,
                   input_var=None, input_layer=None, input_shape=None, batch_normalization=False, weight_normalization=False,
                   ):
        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            all_layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                l_hid = L.batch_norm(l_hid)
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_normalization=weight_normalization
                )
                if batch_normalization:
                    l_hid = L.batch_norm(l_hid)
                all_layers.append(l_hid)
            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization
            )
            if batch_normalization:
                l_out = L.batch_norm(l_out)
            all_layers.append(l_out)
            output = L.get_output(l_out)
            # returns layers(), input_layer, output_layer, input_var, output
            return all_layers, l_in, l_out, l_in.input_var, output

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

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

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


