import torch
import torch.nn as nn
from torchvision.models import resnet50

# Load a pretrained resnet model
resnet = resnet50(pretrained=True)

# Remove the classification head
features = list(resnet.children())[:-2]
resnet = nn.Sequential(*features)
# Freeze the parameters of the resnet model
for param in resnet.parameters():
    param.requires_grad = False


class HierarchicalPromptNetwork(nn.Module):
    def __init__(self, prompt_dim):
        super().__init__()
        self.features = resnet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # The domain-level branch
        self.fc_domain = nn.Linear(2048, prompt_dim)

        # The task-level branch
        self.conv_task = nn.Conv2d(
            2048 + prompt_dim, prompt_dim, kernel_size=1)

    def forward(self, x):
        x = self.features(x)  # [batch_size, 2048, 7, 7]

        # Domain-level prompt
        domain = self.avgpool(x)  # [batch_size, 2048, 1, 1]
        domain = domain.view(domain.size(0), -1)  # [batch_size, 2048]
        domain_prompt = self.fc_domain(domain)  # [batch_size, prompt_dim]

        # Here, we assume that the spatial dimensions of x are (H, W)
        B, P = domain_prompt.shape
        C, H, W = x.shape[1], x.shape[2], x.shape[3]

        # Reshape domain-level prompt to match spatial dimensions of x
        domain_prompt_reshaped = domain_prompt.view(
            B, P, 1, 1).expand(-1, -1, H, W)

        # Now domain_prompt_reshaped has a shape of [B, P, H, W]

        # Concatenate this with x
        x_concat = torch.cat([x, domain_prompt_reshaped], dim=1)

        # Now x_concat has a shape of [B, C+P, H, W]

        # Task-level prompt
        task_prompt = self.conv_task(x_concat)

        # The task_prompt will have a shape of [B, P', H, W]
        # We reshape this to [B, P'] by taking the mean over the spatial dimensions
        task_prompt = task_prompt.view(B, P, H, W).mean(
            dim=[2, 3])  # [batch_size, prompt_dim]

        return domain_prompt, task_prompt
