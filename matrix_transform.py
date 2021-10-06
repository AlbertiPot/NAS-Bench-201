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


def split_arch_str(arch_str: str, key: str) -> list:
    splited_list = []
    splited_list = arch_str.split(key)
    return splited_list


# def build_matrix(arch_str: str):
#     node_list = split_arch_str(arch_str, '+')
#     for node in node_list:


#     return


if __name__ == '__main__':
    arch_str = '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|'
    arch_str2 = '|nor_conv_3x3~0|'
    l = split_arch_str(arch_str2, '|')
    print(l)