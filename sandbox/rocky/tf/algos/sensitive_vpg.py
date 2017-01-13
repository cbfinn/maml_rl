

from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_sensitive_polopt import BatchSensitivePolopt
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf


class SensitiveVPG(BatchSensitivePolopt, Serializable):
    """
    Vanilla Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        super(SensitiveVPG, self).__init__(env=env, policy=policy, baseline=baseline, **kwargs)

    @overrides
    def init_opt(self):
        # TODO Commented out all KL stuff for now, since it is only used for logging
        is_recurrent = int(self.policy.recurrent)
        num_tasks = 10 # TODO - don't hardcode

        init_obs_vars, init_action_vars, init_adv_vars = [], [], []
        for i in range(num_tasks):
            init_obs_vars.append(self.env.observation_space.new_tensor_variable(
                'init_obs' + str(i),
                extra_dims=1 + is_recurrent,
            ))
            init_action_vars.append(self.env.action_space.new_tensor_variable(
                'init_action' + str(i),
                extra_dims=1 + is_recurrent,
            ))
            init_adv_vars.append(tensor_utils.new_tensor(
                name='init_advantage' + str(i),
                ndim=1 + is_recurrent,
                dtype=tf.float32,
            ))
        dist = self.policy.distribution

        #old_dist_info_vars = {
        #    k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
        #    for k, shape in dist.dist_info_specs
        #    }
        #old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        # TODO - worry about this later, not used in standard case.
        #state_info_vars = {
        #    k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
        #    for k, shape in self.policy.state_info_specs
        #    }
        #state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]
        state_info_vars, state_info_vars_list = {}, []

        # TODO - commening out some recurrence stuff now for readability
        #if is_recurrent:
        #    valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        #else:
        #    valid_var = None

        # Only need one for computing each of the gradients for sampling.
        init_dist_info_vars = self.policy.dist_info_sym(init_obs_vars[0], state_info_vars)
        logli = dist.log_likelihood_sym(init_action_vars[0], init_dist_info_vars)
        #kl = dist.kl_sym(old_dist_info_vars, init_dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            init_surr_obj = - tf.reduce_sum(logli * init_adv_vars[0] * valid_var) / tf.reduce_sum(valid_var)
            #mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            #max_kl = tf.reduce_max(kl * valid_var)
        else:
            init_surr_obj = - tf.reduce_mean(logli * init_adv_vars[0])
            #mean_kl = tf.reduce_mean(kl)
            #max_kl = tf.reduce_max(kl)

        input_list = [init_obs_vars[0], init_action_vars[0], init_adv_vars[0]] + state_info_vars_list
        #if is_recurrent:
        #    input_list.append(valid_var)
        self.policy.set_init_surr_obj(input_list, init_surr_obj)

        obs_vars, action_vars, adv_vars = [], [], []
        for i in range(num_tasks):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + str(i),
                extra_dims=1 + is_recurrent,
            ))
            action_vars.append(self.env.action_space.new_tensor_variable(
                'action' + str(i),
                extra_dims=1 + is_recurrent,
            ))
            adv_vars.append(tensor_utils.new_tensor(
                name='advantage' + str(i),
                ndim=1 + is_recurrent,
                dtype=tf.float32,
            ))
            #dist_info_vars = self.policy.dist_info_sym(init_obs_vars[0], state_info_vars)
            #logli = dist.log_likelihood_sym(init_action_var, init_dist_info_vars)


        # TODO - this initializes the update opt used.
        #self.optimizer.update_opt(loss=surr_obj, target=self.policy, inputs=input_list)

        #f_kl = tensor_utils.compile_function(
        #    inputs=input_list + old_dist_info_vars_list,
        #    outputs=[mean_kl, max_kl],
        #)
        #self.opt_info = dict(
        #    f_kl=f_kl,
        #)


    @overrides
    def optimize_policy(self, itr, init_samples_data, updated_samples_data):
        # TODO - this function should update self.policy._pre_update_dist (the learner's parameters)
        import pdb; pdb.set_trace()
        logger.log("optimizing policy")
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        if self.policy.recurrent:
            inputs += (samples_data["valids"],)
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        loss_before = self.optimizer.loss(inputs)
        # pass in gradient info as extra inputs? Need to keep track of which policy was used too.
        # TODO - maybe reformulate the surr_obj as the sum of N objs?
        self.optimizer.optimize(inputs)
        loss_after = self.optimizer.loss(inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        # TODO Commenting out for now since it is only used for logging
        #mean_kl, max_kl = self.opt_info['f_kl'](*(list(inputs) + dist_info_list))
        #logger.record_tabular('MeanKL', mean_kl)
        #logger.record_tabular('MaxKL', max_kl)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
