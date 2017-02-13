

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
            use_sensitive=True,
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
        self.use_sensitive = use_sensitive
        super(SensitiveVPG, self).__init__(env=env, policy=policy, baseline=baseline, use_sensitive=use_sensitive, **kwargs)

    @overrides
    def init_opt(self):
        # TODO Commented out all KL stuff for now, since it is only used for logging
        is_recurrent = int(self.policy.recurrent)
        assert not is_recurrent
        dist = self.policy.distribution

        state_info_vars, state_info_vars_list = {}, []

        init_obs_vars, init_action_vars, init_adv_vars = [], [], []
        init_surr_objs = []
        for i in range(self.meta_batch_size):
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

            init_dist_info_vars = self.policy.dist_info_sym(init_obs_vars[i], state_info_vars)
            logli = dist.log_likelihood_sym(init_action_vars[i], init_dist_info_vars)

            # formulate as a minimization problem
            # The gradient of the surrogate objective is the policy gradient
            init_surr_objs.append(- tf.reduce_mean(logli * init_adv_vars[i]))

        # For computing the fast update for sampling
        input_list = init_obs_vars + init_action_vars + init_adv_vars + state_info_vars_list
        self.policy.set_init_surr_obj(input_list, init_surr_objs)

        obs_vars, action_vars, adv_vars = [], [], []
        surr_objs = []
        for i in range(self.meta_batch_size):
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
            dist_info_vars = self.policy.updated_dist_info_sym(i, init_obs_vars[i], init_surr_objs[i], obs_vars[i], state_info_vars)
            logli = dist.log_likelihood_sym(action_vars[i], dist_info_vars)
            surr_objs.append(- tf.reduce_mean(logli * adv_vars[i]))

        surr_obj = tf.reduce_mean(tf.pack(surr_objs, 0))
        new_input_list = input_list + obs_vars + action_vars + adv_vars

        if self.use_sensitive:
            self.optimizer.update_opt(loss=surr_obj, target=self.policy, inputs=new_input_list)
        else:  # baseline method of just training initial policy
            self.optimizer.update_opt(loss=tf.reduce_mean(tf.pack(init_surr_objs,0)), target=self.policy, inputs=input_list)

        #f_kl = tensor_utils.compile_function(
        #    inputs=input_list + old_dist_info_vars_list,
        #    outputs=[mean_kl, max_kl],
        #)
        #self.opt_info = dict(
        #    f_kl=f_kl,
        #)


    @overrides
    def optimize_policy(self, itr, init_samples_data, updated_samples_data):
        logger.log("optimizing policy")

        init_obs_list, init_action_list, init_adv_list = [], [], []
        for i in range(self.meta_batch_size):
            inputs = ext.extract(
                init_samples_data[i],
                "observations", "actions", "advantages"
            )
            init_obs_list.append(inputs[0])
            init_action_list.append(inputs[1])
            init_adv_list.append(inputs[2])

        obs_list, action_list, adv_list = [], [], []
        for i in range(self.meta_batch_size):
            inputs = ext.extract(
                updated_samples_data[i],
                "observations", "actions", "advantages"
            )
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])

        init_inputs = init_obs_list + init_action_list + init_adv_list
        inputs = init_inputs + obs_list + action_list + adv_list

        if not self.use_sensitive:
            # baseline of only training initial policy
            inputs = init_inputs

        loss_before = self.optimizer.loss(inputs)
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
