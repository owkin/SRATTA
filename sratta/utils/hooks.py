from torch import clone
from torch.nn import Linear


class DetectLambdasPerGradientStepHook:
    """
    Implementation of a backward hook that return the lambdas.
    """

    def __init__(self, model):
        layers = (m for m in model.modules() if isinstance(m, Linear))
        self.backward_hook = next(layers).register_backward_hook(self.backward_hook_fn)
        self.lambdas_per_samples_per_neurons = None

    def backward_hook_fn(self, module, grad_input, grad_output):
        """Function registred as backward hook.

        The function called by pytorch when backpropagating the gradient through last layer.
        We use it to get the lambdas, ie the gradient of the loss wrt preactivation output of
        the neurons.

        Parameters
        ----------
        module : pytorch.nn.Module
            attached module.

        grad_input : Tuple
            tuple with every gradient of the loss with respect to the input of the function applied by the layer.
            In our case, Linear layer, the input are the bias, the tensor input and the weights. So grad_input = (dL/db, dL/dx, dL/dw).
            For the first layer the input correspond to the data input for which grad is not required, thus grad_inpu[1]=None.

        grad_output : Tuple
            gradient of the loss wrt the output of the layer (preactivation w^T.x+b)

        """
        self.lambdas_per_samples_per_neurons = clone(grad_output[0])
        # must have no output

    def close(self):
        self.backward_hook.remove()


class DetectAbsoluteRiskyNeuronHooks:
    """
    Implementation of a forward hook that detect at inference time
    if one input data has a relative importance
    superior to a given threshold compare to other inputs.
    # Theoretically to detect perfect Exans, the threshold must be 1.
    """

    def __init__(self, model):
        layers = (m for m in model.modules() if isinstance(m, Linear))
        layer = next(layers)  # input layer
        layer = next(layers)  # second layer
        self.backward_hook = layer.register_backward_hook(self.backward_hook_fn)
        self.forward_hook = layer.register_forward_hook(self.forward_hook_fn)
        self.activations = None
        self.dL_dz = None

    def forward_hook_fn(self, module, input, output):
        self.activations = input[0].clone()
        # must have no output

    def backward_hook_fn(self, module, grad_input, grad_output):
        """Function registred as backward hook.

        The function called by pytorch when backpropagating the gradient through last layer.
        We use it to check if the gradient of the loss wrt the output of the layer (z) is null.

        Parameters
        ----------
        module : pytorch.nn.Module
            attached module.

        grad_input : Tuple
            tuple with every gradient of the loss with respect to the input of the function applied by the layer.
            In our case, Linear layer, the input are the bias, the tensor input and the weights. So grad_input = (dL/db, dL/dx, dL/dw).
            For the first layer the input correspond to the data input for which grad is not required, thus grad_inpu[1]=None.
            Here we track the second layer

        grad_output : Tuple
            gradient of the loss wrt the output of the layer (preactivation w^T.x+b)

        """
        self.dL_dz = grad_input[1].clone()  # dL/db
        # must have no output

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
