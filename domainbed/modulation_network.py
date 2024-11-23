import torch
import torch.nn as nn


class SimpleModulationNetwork(nn.Module):
    def __init__(self, prompt_dim, num_layers):
        super(SimpleModulationNetwork, self).__init__()
        self.layer = nn.Linear(prompt_dim, prompt_dim)
        self.num_layers = num_layers

    def forward(self, domain_prompt, task_prompt):
        modulated_domain_prompts = []
        modulated_task_prompts = []
        for _ in range(self.num_layers):
            modulated_domain_prompt = self.layer(domain_prompt)
            modulated_task_prompt = self.layer(task_prompt)
            modulated_domain_prompts.append(modulated_domain_prompt)
            modulated_task_prompts.append(modulated_task_prompt)
        return modulated_domain_prompts, modulated_task_prompts


class SharedModulationNetwork(nn.Module):
    def __init__(self, prompt_dim, num_layers):
        super(SharedModulationNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(prompt_dim, prompt_dim) for _ in range(num_layers)])

    def forward(self, domain_prompt, task_prompt):
        modulated_domain_prompts = []
        modulated_task_prompts = []
        for layer in self.layers:
            modulated_domain_prompt = layer(domain_prompt)
            modulated_task_prompt = layer(task_prompt)
            modulated_domain_prompts.append(modulated_domain_prompt)
            modulated_task_prompts.append(modulated_task_prompt)
        return modulated_domain_prompts, modulated_task_prompts


class SeparateModulationNetwork(nn.Module):
    def __init__(self, prompt_dim, num_layers):
        super(SeparateModulationNetwork, self).__init__()
        self.domain_layers = nn.ModuleList(
            [nn.Linear(prompt_dim, prompt_dim) for _ in range(num_layers)])
        self.task_layers = nn.ModuleList(
            [nn.Linear(prompt_dim, prompt_dim) for _ in range(num_layers)])

    def forward(self, domain_prompt, task_prompt):
        modulated_domain_prompts = []
        modulated_task_prompts = []
        for domain_layer, task_layer in zip(self.domain_layers, self.task_layers):
            modulated_domain_prompt = domain_layer(domain_prompt)
            modulated_task_prompt = task_layer(task_prompt)
            modulated_domain_prompts.append(modulated_domain_prompt)
            modulated_task_prompts.append(modulated_task_prompt)
        return modulated_domain_prompts, modulated_task_prompts
