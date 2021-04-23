import matplotlib.pyplot as plt
# from keras.callbacks import Callback
import numpy as np
import tensorflow as tf


class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, mom=0.9, stop_multiplier=None,
                 reload_weights=True, batches_lr_update=5):

        super(LRFinder, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        if stop_multiplier is None:
            self.stop_multiplier = -20 * self.mom / 3 + 10  # 4 if mom=0.9
            # 10 if mom=0
        else:
            self.stop_multiplier = stop_multiplier

    def on_train_begin(self, logs={}):
        p = self.params
        try:
            n_iterations = p['epochs'] * p['samples'] // p['batch_size']
        except:
            n_iterations = p['steps'] * p['epochs']

        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, \
                                           num=n_iterations // self.batches_lr_update + 1)
        self.losses = []
        self.iteration = 0
        self.best_loss = 0
        if self.reload_weights:
            self.model.save_weights('tmp.hdf5')

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')

        if self.iteration != 0:  # Make loss smoother using momentum
            loss = self.losses[-1] * self.mom + loss * (1 - self.mom)

        if self.iteration == 0 or loss < self.best_loss:
            self.best_loss = loss

        if self.iteration % self.batches_lr_update == 0:  # Evaluate each lr over 5 epochs

            if self.reload_weights:
                self.model.load_weights('tmp.hdf5')

            self.model.optimizer.lr = self.learning_rates[self.iteration // self.batches_lr_update]

            self.losses.append(loss)

        if loss > self.best_loss * self.stop_multiplier:  # Stop criteria
            self.model.stop_training = True

        self.iteration += 1

    def on_train_end(self, logs=None):
        if self.reload_weights:
            self.model.load_weights('tmp.hdf5')

        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_rates[:len(self.losses)], self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.show()


class EarlyStoppingByLossVal(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(EarlyStoppingByLossVal, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        # if current is None:
        #     warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True



# Copyright 2019 The TensorFlow Authors, The Hugging Face Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions and classes related to optimization (weight updates)."""


import re
from typing import Callable, List, Optional, Union

import tensorflow as tf



[DOCS]
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.

    Args:
        initial_learning_rate (:obj:`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (:obj:`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (:obj:`int`):
            The number of steps for the warmup part of training.
        power (:obj:`float`, `optional`, defaults to 1):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (:obj:`str`, `optional`):
            Optional name prefix for the returned tensors during the schedule.
    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }




[DOCS]
def create_optimizer(
    init_lr: float,
    num_train_steps: int,
    num_warmup_steps: int,
    min_lr_ratio: float = 0.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    weight_decay_rate: float = 0.0,
    power: float = 1.0,
    include_in_weight_decay: Optional[List[str]] = None,
):
    """
    Creates an optimizer with a learning rate schedule using a warmup phase followed by a linear decay.

    Args:
        init_lr (:obj:`float`):
            The desired learning rate at the end of the warmup phase.
        num_train_steps (:obj:`int`):
            The total number of training steps.
        num_warmup_steps (:obj:`int`):
            The number of warmup steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0):
            The final learning rate at the end of the linear decay will be :obj:`init_lr * min_lr_ratio`.
        adam_beta1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 to use in Adam.
        adam_beta2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 to use in Adam.
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            The epsilon to use in Adam.
        weight_decay_rate (:obj:`float`, `optional`, defaults to 0):
            The weight decay to use.
        power (:obj:`float`, `optional`, defaults to 1.0):
            The power to use for PolynomialDecay.
        include_in_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters except bias and layer norm parameters.
    """
    # Implements linear decay of the learning rate.
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps - num_warmup_steps,
        end_learning_rate=init_lr * min_lr_ratio,
        power=power,
    )
    if num_warmup_steps:
        lr_schedule = WarmUp(
            initial_learning_rate=init_lr,
            decay_schedule_fn=lr_schedule,
            warmup_steps=num_warmup_steps,
        )
    if weight_decay_rate > 0.0:
        optimizer = AdamWeightDecay(
            learning_rate=lr_schedule,
            weight_decay_rate=weight_decay_rate,
            beta_1=adam_beta1,
            beta_2=adam_beta2,
            epsilon=adam_epsilon,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            include_in_weight_decay=include_in_weight_decay,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon
        )
    # We return the optimizer and the LR scheduler in order to better track the
    # evolution of the LR independently of the optimizer.
    return optimizer, lr_schedule



class AdamWeightDecay(tf.keras.optimizers.Adam):
    """
    Adam enables L2 weight decay and clip_by_global_norm on gradients. Just adding the square of the weights to the
    loss function is *not* the correct way of using L2 regularization/weight decay with Adam, since that will interact
    with the m and v parameters in strange ways as shown in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.

    Instead we want ot decay the weights in a manner that doesn't interact with the m/v parameters. This is equivalent
    to adding the square of the weights to the loss with plain (non-momentum) SGD.

    Args:
        learning_rate (:obj:`Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]`, `optional`, defaults to 1e-3):
            The learning rate to use or a schedule.
        beta_1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 parameter in Adam, which is the exponential decay rate for the 1st momentum estimates.
        beta_2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 parameter in Adam, which is the exponential decay rate for the 2nd momentum estimates.
        epsilon (:obj:`float`, `optional`, defaults to 1e-7):
            The epsilon parameter in Adam, which is a small constant for numerical stability.
        amsgrad (:obj:`bool`, `optional`, default to `False`):
            Whether to apply AMSGrad variant of this algorithm or not, see `On the Convergence of Adam and Beyond
            <https://arxiv.org/abs/1904.09237>`__.
        weight_decay_rate (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply.
        include_in_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters by default (unless they are in :obj:`exclude_from_weight_decay`).
        exclude_from_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to exclude from applying weight decay to. If a
            :obj:`include_in_weight_decay` is passed, the names in it will supersede this list.
        name (:obj:`str`, `optional`, defaults to 'AdamWeightDecay'):
            Optional name for the operations created when applying gradients.
        kwargs:
            Keyward arguments. Allowed to be {``clipnorm``, ``clipvalue``, ``lr``, ``decay``}. ``clipnorm`` is clip
            gradients by norm; ``clipvalue`` is clip gradients by value, ``decay`` is included for backward
            compatibility to allow time inverse decay of learning rate. ``lr`` is included for backward compatibility,
            recommended to use ``learning_rate`` instead.
    """

    def __init__(
        self,
        learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        weight_decay_rate: float = 0.0,
        include_in_weight_decay: Optional[List[str]] = None,
        exclude_from_weight_decay: Optional[List[str]] = None,
        name: str = "AdamWeightDecay",
        **kwargs
    ):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {"WarmUp": WarmUp}
        return super(AdamWeightDecay, cls).from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["weight_decay_rate"] = tf.constant(
            self.weight_decay_rate, name="adam_weight_decay_rate"
        )

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var * apply_state[(var.device, var.dtype.base_dtype)]["weight_decay_rate"],
                use_locking=self._use_locking,
            )
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        grads, tvars = list(zip(*grads_and_vars))
        return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars), name=name, **kwargs)

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        # if apply_state is None:
        #     return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients["lr_t"], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"weight_decay_rate": self.weight_decay_rate})
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

