

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

        init_obs_vars, init_action_vars, init_adv_vars = [], [], []
        init_surr_objs = []
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

            init_dist_info_vars = self.policy.dist_info_sym(init_obs_vars[i], state_info_vars)
            logli = dist.log_likelihood_sym(init_action_vars[i], init_dist_info_vars)
            #kl = dist.kl_sym(old_dist_info_vars, init_dist_info_vars)

            # formulate as a minimization problem
            # The gradient of the surrogate objective is the policy gradient
            if is_recurrent:
                init_surr_objs.append(- tf.reduce_sum(logli * init_adv_vars[i] * valid_var) / tf.reduce_sum(valid_var))
                #mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
                #max_kl = tf.reduce_max(kl * valid_var)
            else:
                init_surr_objs.append(- tf.reduce_mean(logli * init_adv_vars[i]))
                #mean_kl = tf.reduce_mean(kl)
                #max_kl = tf.reduce_max(kl)

        # For computing the fast update for sampling
        #input_list = [init_obs_vars[0], init_action_vars[0], init_adv_vars[0]] + state_info_vars_list
        #self.policy.set_init_surr_obj(input_list, init_surr_objs[0])
        input_list = init_obs_vars + init_action_vars + init_adv_vars + state_info_vars_list
        self.policy.set_init_surr_obj(input_list, init_surr_objs)

        #if is_recurrent:
        #    input_list.append(valid_var)

        obs_vars, action_vars, adv_vars = [], [], []
        surr_objs = []
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
            dist_info_vars = self.policy.updated_dist_info_sym(init_obs_vars[i], init_surr_objs[i], obs_vars[i], state_info_vars)
            logli = dist.log_likelihood_sym(action_vars[i], dist_info_vars)
            surr_objs.append(- tf.reduce_mean(logli * adv_vars[i]))


        surr_obj = tf.reduce_mean(tf.pack(surr_objs, 0))

        new_input_list = input_list + obs_vars + action_vars + adv_vars
        self.optimizer.update_opt(loss=surr_obj, target=self.policy, inputs=new_input_list)

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
        logger.log("optimizing policy")
        num_tasks = len(init_samples_data)

        init_obs_list, init_action_list, init_adv_list = [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(
                init_samples_data[i],
                "observations", "actions", "advantages"
            )
            init_obs_list.append(inputs[0])
            init_action_list.append(inputs[1])
            init_adv_list.append(inputs[2])

        obs_list, action_list, adv_list = [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(
                updated_samples_data[i],
                "observations", "actions", "advantages"
            )
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])

        inputs = init_obs_list + init_action_list + init_adv_list + obs_list + action_list + adv_list

        #agent_infos = init_samples_data["agent_infos"]
        #state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        #inputs += tuple(state_info_list)
        #if self.policy.recurrent:
        #    inputs += (samples_data["valids"],)
        #dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
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
