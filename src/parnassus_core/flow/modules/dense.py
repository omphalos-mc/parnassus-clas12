import torch as T
from torch import Tensor
import torch.nn as nn

from parnassus_core.flow.modules.norms import LayerScale


class MLPBlock(nn.Module):
    """A simple MLP block that makes up a dense network.

    Made up of several layers containing:
    - linear map
    - activation function [Optional]
    - layer normalisation [Optional]
    - dropout [Optional]

    Only the input of the block is concatentated with context information.
    For residual blocks, the input is added to the output of the final layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ctxt_dim: int = 0,
        n_layers: int = 1,
        act: str | None = "ReLU",
        norm: str | None = None,
        dropout: float = 0,
        do_res: bool = False,
        layer_scale: bool = False,
        pre_norm: str | None = None,
        init_zeros: bool = False,
    ) -> None:
        """Init method for MLPBlock.

        Parameters
        ----------
        input_dim : int
            The number of features for the input layer
        output_dim : int
            The number of output features
        ctxt_dim : int, optional
            The number of contextual features to concat to the inputs, by default 0
        n_layers : int, optional
            The number of transform layers in this block, by default 1
        act : str, optional
            A string indicating the name of the activation function, by default "lrlu"
        norm : str, optional
            A string indicating the name of the normalisation, by default None
        dropout : float, optional
            The dropout probability, 0 implies no dropout, by default 0
        do_res : bool, optional
            Add to previous output, only if dim does not change, by default 0
        layer_scale : bool, optional
            If the block uses layer scale, by default False
        init_zeros : bool, optional,
            If the final layer weights and bias values are set to zero
            Does not apply to bayesian layers
        """
        super().__init__()

        # Save the input and output dimensions of the module
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ctxt_dim = ctxt_dim

        # If this layer includes an additive residual connection
        self.do_res = do_res and (input_dim == output_dim)

        # Initialise the block layers as a module list
        self.block = nn.ModuleList()

        if pre_norm is not None:
            # If the block is pre-normalised, add the normalisation first
            self.block.append(
                getattr(nn, pre_norm)(input_dim + ctxt_dim, elementwise_affine=False)
            )
        for n in range(n_layers):
            # Increase the input dimension of the first layer to include context
            layer_input_dim = input_dim + ctxt_dim if n == 0 else output_dim

            # Linear transform, activation, normalisation, dropout
            self.block.append(nn.Linear(layer_input_dim, output_dim))

            # Initialise the final layer with zeros
            if init_zeros and n == n_layers - 1:
                self.block[-1].weight.data.fill_(0)
                self.block[-1].bias.data.fill_(0)

            if act is not None:
                self.block.append(getattr(nn, act)())
            if norm is not None:
                self.block.append(getattr(nn, norm)(output_dim))
            if dropout > 0:
                self.block.append(nn.Dropout(dropout))
        if layer_scale:
            self.block.append(LayerScale(output_dim))

    def forward(self, input: Tensor, ctxt: Tensor | None = None) -> Tensor:
        """
        args:
            tensor: Pytorch tensor to pass through the network
            ctxt: The conditioning tensor, can be ignored
        """

        # Concatenate the context information to the input of the block
        if self.ctxt_dim and ctxt is None:
            raise ValueError(
                "Was expecting contextual information but none has been provided!"
            )
        temp = T.cat([input, ctxt], dim=-1) if self.ctxt_dim else input

        # Pass through each transform in the block
        for layer in self.block:
            temp = layer(temp)

        # Add the original inputs again for the residual connection
        if self.do_res:
            temp = temp + input

        return temp

    def __repr__(self) -> str:
        """Generate a one line string summing up the components of the
        block."""
        string = str(self.input_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += "->"
        string += "->".join([str(b).split("(", 1)[0] for b in self.block])
        string += "->" + str(self.output_dim)
        if self.do_res:
            string += "(add)"
        return string


class DenseNetwork(nn.Module):
    """A dense neural network made from a series of consecutive MLP blocks and
    context injection layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        ctxt_dim: int = 0,
        hidden_dim: int | list[int] = 32,
        num_blocks: int = 1,
        n_lyr_pbk: int = 1,
        act_h: str = "ReLU",
        act_o: str | None = None,
        do_out: bool = True,
        norm: str | None = None,
        dropout: float = 0,
        do_res: bool = False,
        ctxt_in_input: bool = True,
        ctxt_in_hidden: bool = False,
        output_init_zeros: bool = False,
    ) -> None:
        """Initialise the DenseNetwork.

        Parameters
        ----------
        input_dim : int
            The number of input neurons
        output_dim : int, optional
            The number of output neurons. If none it will take from input or hidden,
            by default 0
        ctxt_dim : int, optional
            The number of context features. The context feature use is determined by
            ctxt_type, by default 0
        hidden_dim : int | list[int], optional
            The width of each hidden block. If a list it overides depth, by default 32
        num_blocks : int, optional
            The number of hidden blocks, can be overwritten by hidden_dim, by default 1
        n_lyr_pbk : int, optional
            The number of transform layers per hidden block, by default 1
        act_h : str, optional
            The name of the activation function to apply in the hidden blocks,
            by default "lrlu"
        act_o : str, optional
            The name of the activation function to apply to the outputs,
            by default "none"
        do_out : bool, optional
            If the network has a dedicated output block, by default True
        norm : str, optional
            Type of normalisation (layer or batch) in each hidden block, by default "none"
        dropout : float, optional
            Dropout probability for hidden layers (0 means no dropout), by default 0
        do_res : bool, optional
            Use resisdual-connections between hidden blocks (only if same size),
            by default False
        ctxt_in_input : bool, optional
            Include the ctxt tensor in the input block, by default True
        ctxt_in_hidden : bool, optional
            Include the ctxt tensor in the hidden blocks, by default False
        output_init_zeros : bool, optional
            Initialise the output layer weights as zeros

        Raises
        ------
        ValueError
            If the network was given a context input but both ctxt_in_input and
            ctxt_in_hidden were False
        """
        super().__init__()

        # Check that the context is used somewhere
        if ctxt_dim:
            if not ctxt_in_hidden and not ctxt_in_input:
                raise ValueError("Network has context inputs but nowhere to use them!")

        # We store the input, hidden (list), output, and ctxt dims to query them later
        self.input_dim = input_dim
        if not isinstance(hidden_dim, int):
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = num_blocks * [hidden_dim]
        self.output_dim = output_dim or input_dim if do_out else self.hidden_dim[-1]
        self.num_blocks = len(self.hidden_dim)
        self.ctxt_dim = ctxt_dim
        self.do_out = do_out

        # Necc for this module to work with the nflows package
        self.hidden_features = self.hidden_dim[-1]

        # Input MLP block
        self.input_block = MLPBlock(
            input_dim=self.input_dim,
            output_dim=self.hidden_dim[0],
            ctxt_dim=self.ctxt_dim if ctxt_in_input else 0,
            act=act_h,
            norm=norm,
            dropout=dropout,
        )

        # All hidden blocks as a single module list
        self.hidden_blocks = []
        if self.num_blocks > 1:
            self.hidden_blocks = nn.ModuleList()
            for h_1, h_2 in zip(self.hidden_dim[:-1], self.hidden_dim[1:]):
                self.hidden_blocks.append(
                    MLPBlock(
                        input_dim=h_1,
                        output_dim=h_2,
                        ctxt_dim=self.ctxt_dim if ctxt_in_hidden else 0,
                        n_layers=n_lyr_pbk,
                        act=act_h,
                        norm=norm,
                        dropout=dropout,
                        do_res=do_res,
                    )
                )

        # Output block (optional and there is no normalisation, dropout or context)
        if do_out:
            self.output_block = MLPBlock(
                input_dim=self.hidden_dim[-1],
                output_dim=self.output_dim,
                act=act_o,
                init_zeros=output_init_zeros,
            )

    def forward(self, inputs: Tensor, ctxt: Tensor | None = None) -> Tensor:
        """Pass through all layers of the dense network."""

        # Reshape the context if it is available
        if ctxt is not None:
            dim_diff = inputs.dim() - ctxt.dim()
            if dim_diff > 0:
                ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
                ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        # Pass through the input block
        inputs = self.input_block(inputs, ctxt)

        # Pass through each hidden block
        for h_block in self.hidden_blocks:  # Context tensor will only be used if
            inputs = h_block(inputs, ctxt)  # block was initialised with a ctxt dim

        # Pass through the output block
        if self.do_out:
            inputs = self.output_block(inputs)

        return inputs

    def __repr__(self):
        string = ""
        string += "\n  (inp): " + repr(self.input_block) + "\n"
        for i, h_block in enumerate(self.hidden_blocks):
            string += f"  (h-{i + 1}): " + repr(h_block) + "\n"
        if self.do_out:
            string += "  (out): " + repr(self.output_block)
        return string

    def one_line_string(self):
        """Return a one line string that sums up the network structure."""
        string = str(self.input_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += ">"
        string += str(self.input_block.output_dim) + ">"
        if self.num_blocks > 1:
            string += ">".join(
                [
                    str(layer.out_features)
                    for hidden in self.hidden_blocks
                    for layer in hidden.block
                    if isinstance(layer, nn.Linear)
                ]
            )
            string += ">"
        if self.do_out:
            string += str(self.output_dim)
        return string


def concat_tensors(tensors: list[Tensor]) -> Tensor:
    if len(tensors) == 1:
        return tensors[0]

    x = []

    for tensor in tensors:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        x.append(tensor)

    return T.concatenate(x, dim=-1)


class NamedNetwork(nn.Module):
    def __init__(
        self,
        input_name: str,
        dense_kwargs: dict[str, str],
        fields: list[str],
    ):
        """A wrapper which takes a list of input features, concatenates them, and passes them through a dense
        layer followed by an optional positional encoding module.

        Parameters
        ----------
        input_name : str
            The name of the feature / object that will be embedded, e.g. pix for pixel clusters.
        net : nn.Module
            Module used to perform the feature embedding.
        fields : list[str]
            A list of fields belonging to the feature that will be embedded. E.g. [x, y, z] together with a
            input name of "pix" would result in the fields "pix_x", "pix_y" and "pix_z" being concatenated
            together to make the feature vector.
        posenc : nn.Module
            An optional module used to perform the positional encoding.
        """
        super().__init__()
        self.input_name = input_name
        self.net = DenseNetwork(**dense_kwargs)
        self.fields = fields

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """Embed the set of input features into an embedding.

        Parameters
        ----------
        inputs : dict
            Input data consisting of a dictionary the requested input features.

        Returns
        -------
        x : Tensor
            Tensor containing an embedding of the concatenated input features.
        """
        # Some input fields will be a vector, i.e. have shape (batch, keys, D) where D > 1
        # But must will be scalars, i.e. (batch, keys), so for these we reshape them to (batch, keys, 1)
        # After this we can then concatenate everything together

        x = self.net(
            concat_tensors(
                [inputs[f"{self.input_name}_{field}"] for field in self.fields]
            )
        )

        return x
