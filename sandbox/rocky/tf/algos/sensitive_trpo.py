

from sandbox.rocky.tf.algos.sensitive_npo import SensitiveNPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class SensitiveTRPO(SensitiveNPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(SensitiveTRPO, self).__init__(optimizer=optimizer, **kwargs)
