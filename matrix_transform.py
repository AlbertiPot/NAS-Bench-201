import numpy as np

# from xautodl.models.cell_operations import NAS_BENCH_201
NAS_BENCH_201 = [
    "none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"
]
NODE_TYPE_DICT = {
    "none": 0,
    "skip_connect": 1,
    "nor_conv_1x1": 2,
    "nor_conv_3x3": 3,
    "avg_pool_3x3": 4
}
N_NODES = 4


def split_arch_str(arch_str: str, key: str) -> list:
    splited_list = []
    splited_list = arch_str.split(key)
    return splited_list


def build_matrix(arch_str: str):
    matrix = np.zeros(shape=(N_NODES, N_NODES), dtype=np.int32)

    # split origin str to 3 end nodes which are splited by '+'
    nodes_list = split_arch_str(arch_str, '+')
    assert len(nodes_list) == N_NODES - 1, 'Wrong length of nodes'

    for end_node, nodes in enumerate(nodes_list, start=1):
        nodes = nodes.strip('|')
        opts_list = split_arch_str(nodes, '|')
        for opts in opts_list:
            # ['avg_pool_3x3', '0'] [算子类型，前序节点号]
            edge_list = split_arch_str(opts, '~')
            edge_type = NODE_TYPE_DICT[edge_list[0]]
            start_node = int(edge_list[1])
            matrix[start_node][end_node] = edge_type

    return matrix.tolist()


if __name__ == '__main__':
    arch_str = '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
    matrix = build_matrix(arch_str)
    print(matrix)    
