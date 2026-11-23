import functools
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair, _single
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from aspp import ASPP1D
from loaddata import AA_ALPHABET, ADAPTED_FEATURE_SIZE, ESM_DIM, MAX_LENGTH, PAD_LABEL


class ESMAdapter(nn.Module):
    def __init__(self, hidden_size, layers, heads, dropout, max_length):
        super().__init__()
        self.input_projection = nn.Linear(ESM_DIM, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, heads, hidden_size * 4, dropout)
            for _ in range(layers)
        ])
        self.output_projection = nn.Linear(hidden_size, len(AA_ALPHABET))

        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_size)
        )
        phase = positions * frequencies
        encoding = torch.zeros(max_length, hidden_size)
        encoding[:, 0::2] = phase.sin()
        encoding[:, 1::2] = phase[:, :hidden_size // 2].cos()
        self.register_buffer('position_encoding', encoding.unsqueeze(1))

    def forward(self, embeddings, valid_mask):
        length = embeddings.size(1)
        if length > self.position_encoding.size(0):
            raise ValueError('The sequence exceeds the adapter maximum length')

        # Attention softmax needs at least one unmasked key per sample.
        padding_mask = ~valid_mask
        padding_mask[:, 0] = padding_mask[:, 0] & valid_mask.any(dim=1)
        hidden = self.input_projection(embeddings).transpose(0, 1)
        hidden = hidden + self.position_encoding[:length].to(hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
        return self.output_projection(hidden.transpose(0, 1))


class GatedCBEASPP(nn.Module):
    def __init__(
        self,
        vocab_size=PAD_LABEL + 1,
        embed_dim=ADAPTED_FEATURE_SIZE,
        kernel_width=3,
        out_channel=64,
        n_layers=64,
        res_block_cnt=2,
        activationFunc='default',
        config=None,
        dropout=0.1,
    ):
        super().__init__()
        self.class_num = getattr(config, 'class_num', vocab_size)
        if self.class_num != PAD_LABEL + 1:
            raise ValueError('The classifier requires eight DSSP states and one pad class')

        ablation = getattr(config, 'ablation', 'none')
        if ablation not in ('none', 'bigru', 'bilstm', 'aspp'):
            raise ValueError('ablation must be one of none, bigru, bilstm or aspp')

        self.adapter = ESMAdapter(
            hidden_size=getattr(config, 'adapter_hidden_size', 512),
            layers=getattr(config, 'adapter_layers', 4),
            heads=getattr(config, 'adapter_heads', 8),
            dropout=getattr(config, 'adapter_dropout', 0.1),
            max_length=getattr(config, 'data_max_text_len', MAX_LENGTH),
        )

        gru_layers = getattr(config, 'bigru_layer_num', 3)
        lstm_layers = getattr(config, 'bilstm_layer_num', getattr(config, 'blstm_layer_num', 2))
        lstm_hidden = getattr(config, 'lstm_hidden_size', 512)
        recurrent_dropout = getattr(config, 'lstm_dropout_rate', 0.0)
        fused_channels = out_channel + 2 * lstm_hidden

        self.bigru_encoder = None
        if ablation != 'bigru':
            self.bigru_encoder = nn.GRU(
                input_size=ADAPTED_FEATURE_SIZE,
                hidden_size=out_channel,
                num_layers=gru_layers,
                batch_first=True,
                dropout=recurrent_dropout if gru_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.bigru_proj = nn.Linear(2 * out_channel, fused_channels)

        self.bilstm_decoder = None
        if ablation != 'bilstm':
            self.bilstm_decoder = nn.LSTM(
                input_size=ADAPTED_FEATURE_SIZE,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=recurrent_dropout if lstm_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.bilstm_proj = nn.Linear(2 * lstm_hidden, fused_channels)

        if self.bigru_encoder is not None and self.bilstm_decoder is not None:
            self.fusion_gate = nn.Linear(2 * fused_channels, fused_channels)

        self.aspp = None
        head_channels = fused_channels
        if ablation != 'aspp':
            self.aspp = ASPP1D(
                fused_channels,
                dilations=getattr(config, 'dilation_list', (1, 2, 4, 8)),
                kernel_sizes=getattr(config, 'window_size_list', (1, 3, 3, 3)),
            )
            head_channels = self.aspp.out_channels

        self.output_dropout = nn.Dropout(getattr(config, 'output_dropout_rate', dropout))
        self.classifier = nn.Conv1d(head_channels, self.class_num, kernel_size=1)

    def _resolve_mask(self, seq, valid_mask):
        if seq.dim() != 3 or seq.size(0) == 0 or seq.size(2) == 0:
            raise ValueError('Expected features shaped (batch, channels, residues)')
        if valid_mask is None:
            return seq[:, -len(AA_ALPHABET):].ne(0).any(dim=1)
        if valid_mask.shape != (seq.size(0), seq.size(2)):
            raise ValueError('The residue mask must have shape (batch, residues)')
        return valid_mask.to(device=seq.device, dtype=torch.bool)

    @staticmethod
    def _recurrent_features(stream, projection, packed, length):
        packed_output, _ = stream(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=length)
        return projection(output)

    def forward(self, seq, valid_mask=None):
        """Return nine logits per residue, with zeros at padded positions."""
        valid_mask = self._resolve_mask(seq, valid_mask)
        input_length = seq.size(2)
        positions = torch.arange(1, input_length + 1, device=seq.device)
        lengths = positions.expand_as(valid_mask).masked_fill(~valid_mask, 0).max(dim=1)[0]
        lengths = lengths.clamp(min=1).cpu()
        length = int(lengths.max().item())
        valid_mask = valid_mask[:, :length]
        pad_mask = ~valid_mask.unsqueeze(2)
        features = seq[:, :, :length].transpose(1, 2).masked_fill(pad_mask, 0)

        if features.size(2) == ESM_DIM + len(AA_ALPHABET):
            embeddings = self.adapter(features[:, :, :ESM_DIM], valid_mask)
            features = torch.cat((embeddings, features[:, :, ESM_DIM:]), dim=2)
            features = features.masked_fill(pad_mask, 0)
        elif features.size(2) != ADAPTED_FEATURE_SIZE:
            raise ValueError('Expected ESM features with appended amino-acid one-hot channels')

        packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        if self.bigru_encoder is None:
            fused = self._recurrent_features(self.bilstm_decoder, self.bilstm_proj, packed, length)
        elif self.bilstm_decoder is None:
            fused = self._recurrent_features(self.bigru_encoder, self.bigru_proj, packed, length)
        else:
            gru = self._recurrent_features(self.bigru_encoder, self.bigru_proj, packed, length)
            lstm = self._recurrent_features(self.bilstm_decoder, self.bilstm_proj, packed, length)
            gate = torch.sigmoid(self.fusion_gate(torch.cat((gru, lstm), dim=2)))
            fused = gate * gru + (1.0 - gate) * lstm

        fused = fused.masked_fill(pad_mask, 0).transpose(1, 2)
        head_features = self.aspp(fused) if self.aspp is not None else fused
        logits = self.classifier(self.output_dropout(head_features))
        logits = logits.masked_fill(~valid_mask.unsqueeze(1), 0)
        return F.pad(logits, (0, input_length - length))

    def predict_proba(self, seq, valid_mask=None):
        """Return Q8 probabilities for valid residues in DSSP class order."""
        valid_mask = self._resolve_mask(seq, valid_mask)
        logits = self(seq, valid_mask)[:, :PAD_LABEL]
        return F.softmax(logits, dim=1).masked_fill(~valid_mask.unsqueeze(1), 0)


class CondConv1D(_ConvNd):
    # class CondConv1D(Conv1d):
    r"""Learn specialized convolutional kernels for each example.
    As described in the paper
    `CondConv: Conditionally Parameterized Convolutions for Efficient Inference`_ ,
    conditionally parameterized convolutions (CondConv), 
    which challenge the paradigm of static convolutional kernels 
    by computing convolutional kernels as a function of the input.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts per layer 
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    .. _CondConv: Conditionally Parameterized Convolutions for Efficient Inference:
       https://arxiv.org/abs/1904.04971
    """

    def __init__(self, in_channels=42, out_channels=9, kernel_size=11, stride=1,
                 padding=int((11-1)/2), dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        # kernel_size = _pair(kernel_size)
        # stride = _pair(stride)
        # padding = _pair(padding)
        # dilation = _pair(dilation)

        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)

        # super(CondConv1D, self).__init__(
        #     in_channels, out_channels, kernel_size, stride, padding, dilation,
        #     False, _pair(0), groups, bias, padding_mode)
        super(CondConv1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(
            F.adaptive_avg_pool1d, output_size=(1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv1d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        # print('input', input.size())
        pooled_inputs = self._avg_pooling(input)
        # print('pooled_inputs', pooled_inputs.size())
        routing_weights = self._routing_fn(pooled_inputs)
        # print('routing_weights', routing_weights.size())
        # print('self.weight', self.weight.size())

        for idx, weight_sub in enumerate(self.weight):
            kernel_sub = weight_sub
            # kernel_sub = torch.squeeze(weight_sub, 0)
            output_0 = self._conv_forward(input, kernel_sub)
            if idx == 0:
                sumout = routing_weights[:, idx, None, None] * output_0
            else:
                # sumout = torch.sum(routing_weights[:, idx, None, None] * output_0 ,1)
                sumout += routing_weights[:, idx, None, None] * output_0
        # print('sumout', sumout.size())
        return sumout

        # first:([100, 42, 11])
        kernels = torch.sum(
            routing_weights[:, None, None, None] * self.weight, 0)
        # print('------kernels', kernels.size())
        # kernels = torch.sum(routing_weights[:,None, None, None, None] * self.weight, 0)

        return self._conv_forward(input, kernels)


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        # x = torch.flatten(x)
        x = torch.squeeze(x, 2)
        # print('flattenX', x.size())
        x = self.dropout(x)
        x = self.fc(x)

        return F.sigmoid(x)
