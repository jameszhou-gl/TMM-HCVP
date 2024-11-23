import math
from functools import reduce
from operator import mul
from torch.nn import Dropout
import copy
from domainbed.lib import wide_resnet
import torchvision.models
import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import partial
from typing import Callable
import os
import sys
import timm
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(src_dir)
from domainbed.prompt_network import HierarchicalPromptNetwork
from domainbed.modulation_network import SimpleModulationNetwork, SharedModulationNetwork, SeparateModulationNetwork


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class Project(nn.Module):
    def __init__(self, n_inputs, n_outputs, mlp_width) -> None:
        super().__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.output = nn.Linear(mlp_width, n_outputs)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:,
                                               i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


class ViT(nn.Module):
    """
    ViT
    """

    def __init__(self, input_shape, hparams):
        super(ViT, self).__init__()
        self.hparams = hparams
        self.input_shape = input_shape
        self.n_outputs = 768

        self.dropout = nn.Dropout(hparams["resnet_dropout"])

        if hparams['im21k']:
            import timm
            self.network = timm.create_model(
                "vit_base_patch16_224", pretrained=True, drop_path_rate=0.1)
            del self.network.head
            self.network.head = Identity()
        else:
            self.network = torchvision.models.vit_b_16(
                pretrained=True, attention_dropout=hparams["attention_dropout"])
            del self.network.heads
            self.network.heads = Identity()

        # # ! choose different backbones
        # if hparams['backbones'] == 'vit_base_patch16_224.orig_in21k':
        #     self.network = timm.create_model(
        #         'vit_base_patch16_224.orig_in21k', pretrained=True)
        #     del self.network.head
        #     self.network.head = Identity()
        # elif hparams['backbones'] == 'vit_base_patch16_224.augreg_in1k':
        #     self.network = timm.create_model(
        #         'vit_base_patch16_224.augreg_in1k', pretrained=True)
        #     del self.network.head
        #     self.network.head = Identity()
        # elif hparams['backbones'] == 'vit_base_patch16_224.mae':
        #     self.network = timm.create_model(
        #         'vit_base_patch16_224.mae', pretrained=True)
        #     del self.network.head
        #     self.network.head = Identity()
        # # elif hparams['backbones'] == 'vit_huge_patch14_224.orig_in21k':
        # #     self.network = timm.create_model(
        # #         'vit_huge_patch14_224.orig_in21k', pretrained=True)
        # #     del self.network.head
        # #     self.network.head = Identity()

        # else:
        #     raise NotImplementedError

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        # print(x.shape)
        # print(self.network.num_classes)
        # print(self.network.default_cfg['hidden_dim'])
        # print(self.network.embed_dim)
        return self.dropout(self.network(x))


class Vit_HCVP(nn.Module):

    def __init__(self, input_shape, hparams):
        super(Vit_HCVP, self).__init__()
        self.hparams = hparams
        self.input_shape = input_shape
        self.n_outputs = 768
        self.dropout = nn.Dropout(hparams["hcvp_dropout"])

        if hparams['im21k']:
            self.network = timm.create_model(
                "vit_base_patch16_224", pretrained=True, drop_path_rate=0.1)
            del self.network.head
            self.network.head = Identity()

        else:
            self.network = torchvision.models.vit_b_16(
                pretrained=True, attention_dropout=hparams["attention_dropout"])
            del self.network.heads
            self.network.heads = Identity()

        self.init_prompts()
        if hparams["hcvp_freeze_backbone"]:
            # Freeze the featurizer's parameters
            for k, p in self.network.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

    def init_prompts(self):
        self.patch_size = (16, 16)
        self.prompt_dim = 768
        self.num_layers = 12
        self.prompt_dropout = Dropout(0.0)
        self.prompt_proj = nn.Identity()
        self.visualization = False
        # initiate prompt:
        # val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.prompt_dim))  # noqa

        self.prompt_generator = HierarchicalPromptNetwork(self.prompt_dim)
        if self.hparams['hcvp_DEEP']:
            total_d_layer = self.num_layers-1
            if self.hparams['hcvp_modulation_select'] == 'simple':
                self.prompt_modulation = SimpleModulationNetwork(
                    self.prompt_dim, total_d_layer)
            elif self.hparams['hcvp_modulation_select'] == 'shared':
                self.prompt_modulation = SharedModulationNetwork(
                    self.prompt_dim, total_d_layer)
            elif self.hparams['hcvp_modulation_select'] == 'separate':
                self.prompt_modulation = SeparateModulationNetwork(
                    self.prompt_dim, total_d_layer)
            else:
                raise NotImplementedError

    def incorporate_prompts(self, x):
        # combine prompt embeddings with image-patch embeddings
        # Assuming x is the input image
        emb_x = self.network.patch_embed(x)  # Apply patch embedding
        cls_token = self.network.cls_token.expand(
            x.size(0), -1, -1)  # Optional, if you have a class token
        emb_x = torch.cat((cls_token, emb_x), dim=1)
        emb_x = emb_x + self.network.pos_embed  # Apply position embedding
        # Generate the prompts
        domain_prompt, task_prompt = self.prompt_generator(x)

        # Incorporate the prompts into the embedding output
        emb_x = torch.cat((
            emb_x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(
                domain_prompt).unsqueeze(1)),
            self.prompt_dropout(self.prompt_proj(
                task_prompt).unsqueeze(1)),
            emb_x[:, 1:, :]
        ), dim=1)
        # (batch_size, cls_token + n_prompt*2 + n_patches, hidden_dim)
        return emb_x, domain_prompt, task_prompt

    def forward_deep_hierarchy_prompt(self, embedding_output, domain_prompt, task_prompt):
        attn_weights = []
        hidden_states = None
        weights = None
        # B = embedding_output.shape[0]
        # num_layers = 12
        modulated_domain_prompts, modulated_task_prompts = self.prompt_modulation(
            domain_prompt, task_prompt)
        self.blocks = self.network.blocks
        for i, block in enumerate(self.blocks):
            if i == 0:
                hidden_states = block(
                    embedding_output)
            else:
                assert i <= len(modulated_domain_prompts)
                deep_domain_prompt_emb = self.prompt_dropout(self.prompt_proj(
                    modulated_domain_prompts[i-1]))
                deep_task_prompt_emb = self.prompt_dropout(self.prompt_proj(
                    modulated_task_prompts[i-1]))
                hidden_states = torch.cat((
                    hidden_states[:, :1, :],
                    deep_domain_prompt_emb.unsqueeze(1),
                    deep_task_prompt_emb.unsqueeze(1),
                    hidden_states[:, 3:, :]
                ), dim=1)

                hidden_states = block(hidden_states)

            if self.visualization:
                attn_weights.append(weights)

        encoded = self.network.norm(hidden_states)
        return encoded[:, 0, :], attn_weights

    def forward_shallow_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        hidden_states = self.network.blocks(embedding_output)
        encoded = self.network.norm(hidden_states)
        return encoded[:, 0, :], attn_weights

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        emb_x, domain_prompt, task_prompt = self.incorporate_prompts(x)
        if self.hparams['hcvp_DEEP']:
            encoded, attn_weights = self.forward_deep_hierarchy_prompt(
                emb_x, domain_prompt, task_prompt)
        else:
            encoded, attn_weights = self.forward_shallow_prompt(emb_x)
        return self.dropout(encoded), domain_prompt, task_prompt


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if hparams['vit_base_16']:
        return ViT(input_shape, hparams)

    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim(
        ) == 3, f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
