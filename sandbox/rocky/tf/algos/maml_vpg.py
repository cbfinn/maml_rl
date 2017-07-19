

from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_maml_polopt import BatchMAMLPolopt
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf


class MAMLVPG(BatchMAMLPolopt, Serializable):
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
            use_maml=True,
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
        self.use_maml = use_maml
        super(MAMLVPG, self).__init__(env=env, policy=policy, baseline=baseline, use_maml=use_maml, **kwargs)


    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        obs_vars, action_vars, adv_vars = [], [], []
        for i in range(self.meta_batch_size):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            action_vars.append(self.env.action_space.new_tensor_variable(
                'action' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            adv_vars.append(tensor_utils.new_tensor(
                name='advantage' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.float32,
            ))
        return obs_vars, action_vars, adv_vars


    @overrides
    def init_opt(self):
        # TODO Commented out all KL stuff for now, since it is only used for logging
        # To see how it can be turned on, see maml_npo.py
        is_recurrent = int(self.policy.recurrent)
        assert not is_recurrent # not supported right now.
        dist = self.policy.distribution

        old_dist_info_vars, old_dist_info_vars_list = [], []
        for i in range(self.meta_batch_size):
            old_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            old_dist_info_vars_list += [old_dist_info_vars[i][k] for k in dist.dist_info_keys]

        state_info_vars, state_info_vars_list = {}, []

        all_surr_objs, input_list = [], []
        new_params = None
        for j in range(self.num_grad_updates):
            obs_vars, action_vars, adv_vars = self.make_vars(str(j))
            surr_objs = []

            cur_params = new_params
            new_params = []

            for i in range(self.meta_batch_size):
                if j == 0:
                    dist_info_vars, params = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=self.policy.all_params)
                else:
                    dist_info_vars, params = self.policy.updated_dist_info_sym(i, all_surr_objs[-1][i], obs_vars[i], params_dict=cur_params[i])

                new_params.append(params)
                logli = dist.log_likelihood_sym(action_vars[i], dist_info_vars)

                # formulate as a minimization problem
                # The gradient of the surrogate objective is the policy gradient
                surr_objs.append(- tf.reduce_mean(logli * adv_vars[i]))

            input_list += obs_vars + action_vars + adv_vars + state_info_vars_list
            if j == 0:
                # For computing the fast update for sampling
                self.policy.set_init_surr_obj(input_list, surr_objs)
                init_input_list = input_list

            all_surr_objs.append(surr_objs)


        obs_vars, action_vars, adv_vars = self.make_vars('test')
        surr_objs = []
        kls = []
        for i in range(self.meta_batch_size):
            dist_info_vars, _ = self.policy.updated_dist_info_sym(i, all_surr_objs[-1][i], obs_vars[i], params_dict=new_params[i])
            logli = dist.log_likelihood_sym(action_vars[i], dist_info_vars)
            surr_objs.append(- tf.reduce_mean(logli * adv_vars[i]))

            kls.append(dist.kl_sym(old_dist_info_vars[i], dist_info_vars))

        surr_obj = tf.reduce_mean(tf.stack(surr_objs, 0))
        mean_kl = tf.reduce_mean(tf.concat(kls, 0))
        max_kl = tf.reduce_max(tf.concat(kls, 0))
        input_list += obs_vars + action_vars + adv_vars

        if self.use_maml:
            self.optimizer.update_opt(loss=surr_obj, target=self.policy, inputs=input_list)
        else:  # baseline method of just training initial policy
            self.optimizer.update_opt(loss=tf.reduce_mean(tf.stack(all_surr_objs[0],0)), target=self.policy, inputs=init_input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )

        #f_kl = tensor_utils.compile_function(
        #    inputs=input_list + old_dist_info_vars_list,
        #    outputs=[mean_kl, max_kl],
        #)
        #self.opt_info = dict(
        #    f_kl=f_kl,
        #)


    @overrides
    def optimize_policy(self, itr, all_samples_data):
        logger.log("optimizing policy")
        assert len(all_samples_data) == self.num_grad_updates + 1

        if not self.use_maml:
            all_samples_data = [all_samples_data[0]]

        input_list = []
        for step in range(len(all_samples_data)):
            obs_list, action_list, adv_list = [], [], []
            for i in range(self.meta_batch_size):


                inputs = ext.extract(
                    all_samples_data[step][i],
                    "observations", "actions", "advantages"
                )
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                adv_list.append(inputs[2])
            input_list += obs_list + action_list + adv_list

            if step == 0:
                init_inputs = input_list

        loss_before = self.optimizer.loss(input_list)
        self.optimizer.optimize(input_list)
        loss_after = self.optimizer.loss(input_list)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        dist_info_list = []
        for i in range(self.meta_batch_size):
            agent_infos = all_samples_data[-1][i]['agent_infos']
            dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        if self.use_maml:
            mean_kl, max_kl = self.opt_info['f_kl'](*(list(input_list) + dist_info_list))
            logger.record_tabular('MeanKL', mean_kl)
            logger.record_tabular('MaxKL', max_kl)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
