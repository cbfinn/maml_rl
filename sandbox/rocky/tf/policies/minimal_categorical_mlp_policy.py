import sandbox.rocky.tf.core.layers as L
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.policies.base import StochasticPolicy
from rllab.misc import ext
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.spaces.discrete import Discrete
from rllab.misc import logger
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
import itertools
import tensorflow as tf
from sandbox.rocky.tf.core.utils import make_input, _create_param, add_param, make_dense_layer, forward_dense_layer, make_param_layer, forward_param_layer

tf_layers = None
load_params = True


@contextmanager
def suppress_params_loading():
    global load_params
    load_params = False
    yield
    load_params = True



class CategoricalMLPPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            prob_network=None,
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        with tf.variable_scope(name):
            if prob_network is None:
                prob_network = self.create_MLP(
                    input_shape=(obs_dim,),
                    output_dim=env_spec.action_space.n,
                    hidden_sizes=hidden_sizes,
                    name="prob_network",
                )
            self._l_obs, self._l_prob = self.forward_MLP('prob_network', prob_network,
                n_hidden=len(hidden_sizes), input_shape=(obs_dim,),
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=tf.nn.softmax, reuse=None)

            # if you want to input your own tensor.
            self._forward_out = lambda x, is_train: self.forward_MLP('prob_network', prob_network,
                n_hidden=len(hidden_sizes), hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity, input_tensor=x, is_training=is_train)[1]


            self._f_prob = tensor_utils.compile_function(
                [self._l_obs],
                L.get_output(self._l_prob)
            )

            self._dist = Categorical(env_spec.action_space.n)


    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None, is_training=True):
        # sym means symbolic here.
        output = self._forward_out(tf.cast(obs_var,tf.float32), is_training)
        return dict(prob=output)

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self._f_prob(obs))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist


    # This makes all of the parameters.
    def create_MLP(self, name, output_dim, hidden_sizes,
                   hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                   output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer,
                   input_shape=None, weight_normalization=False,
                   ):
        assert input_shape is not None
        cur_shape = input_shape
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

    def forward_MLP(self, name, all_params, input_tensor=None, input_shape=None, n_hidden=-1,
                    hidden_nonlinearity=tf.identity, output_nonlinearity=tf.identity,
                    batch_normalization=False, reuse=True, is_training=False):
        # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
        # set reuse to False if the first time this func is called.
        with tf.variable_scope(name):
            if input_tensor is None:
                assert input_shape is not None
                l_in = make_input(shape=(None,)+input_shape, input_var=None, name='input')
            else:
                l_in = input_tensor
            l_hid = l_in
            for idx in range(n_hidden):
                l_hid = forward_dense_layer(l_hid, all_params['W'+str(idx)], all_params['b'+str(idx)],
                                            batch_norm=batch_normalization,
                                            nonlinearity=hidden_nonlinearity,
                                            scope=str(idx), reuse=reuse,
                                            is_training=is_training
                                            )
            output = forward_dense_layer(l_hid, all_params['W'+str(n_hidden)], all_params['b'+str(n_hidden)],
                                         batch_norm=False, nonlinearity=output_nonlinearity,
                                         )
            return l_in, output


