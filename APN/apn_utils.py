import torch

def get_middle_graph(kernel_size):    
    # creat middle_graph to mask the L_CPT:
    raw_graph = torch.zeros((2 * kernel_size - 1, 2 * kernel_size - 1))
    for x in range(- kernel_size + 1, kernel_size):
        for y in range(- kernel_size + 1, kernel_size):
            raw_graph[x + (kernel_size - 1), y + (kernel_size - 1)] = x ** 2 + y ** 2
    middle_graph = torch.zeros((kernel_size ** 2, kernel_size, kernel_size))
    for x in range(kernel_size):
        for y in range(kernel_size):
            middle_graph[x * kernel_size + y, :, :] = \
                raw_graph[kernel_size - 1 - x: 2 * kernel_size - 1 - x,
                kernel_size - 1 - y: 2 * kernel_size - 1 - y]
    middle_graph = middle_graph.cuda()

    return middle_graph

def add_glasso(var, group):
    return var[group, :].pow(2).sum(dim=0).add(1e-8).sum().pow(1/2.)