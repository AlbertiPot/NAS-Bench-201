import os
import json
import time
import torch

from nas_201_api import NASBench201API as API
from xautodl.models import get_cell_based_tiny_net
from fvcore.nn import FlopCountAnalysis, parameter_count
from matrix_transform import build_matrix


NODE_TYPE_DICT = {
    "none": 0,
    "skip_connect": 1,
    "nor_conv_1x1": 2,
    "nor_conv_3x3": 3,
    "avg_pool_3x3": 4
}

def main(api):
    
    dataset = {}
    
    for index, arch_str in enumerate(api):
        arch_dict = {}

        matrix = build_matrix(arch_str)
        arch_dict['cell_adjacency'] = matrix
        
        cifar10_valid_dict = api.get_more_info(index, 'cifar10-valid', 199, hp='200', is_random=False)
        cifar10_dict = api.get_more_info(index, 'cifar10', 199, hp='200', is_random=False)  
        cifar10_val_acc = cifar10_valid_dict['valid-accuracy']
        cifar10_test_acc = cifar10_dict['test-accuracy']
        # print(cifar10_val_acc)
        # print(cifar10_test_acc)
        arch_dict['cifar10_val_acc'] = cifar10_val_acc
        arch_dict['cifar10_test_acc'] = cifar10_test_acc

        cifar100_dict = api.get_more_info(index, 'cifar100', 199, hp='200', is_random=False)
        cifar100_val_acc = cifar100_dict['valid-accuracy']
        cifar100_test_acc = cifar100_dict['test-accuracy']
        # print(cifar100_dict_val_acc)
        # print(cifar100_dict_test_acc)
        arch_dict['cifar100_val_acc'] = cifar100_val_acc
        arch_dict['cifar100_test_acc'] = cifar100_test_acc
        
        imagenat16_dict = api.get_more_info(index, 'ImageNet16-120', 199, hp='200', is_random=False)
        imagenat16_val_acc = imagenat16_dict['valid-accuracy']
        imagenat16_test_acc = imagenat16_dict['test-accuracy']
        # print(imagenat16_val_acc)
        # print(imagenat16_test_acc)
        arch_dict['imagenet16_val_acc'] = imagenat16_val_acc
        arch_dict['imagenet16_test_acc'] = imagenat16_test_acc

        info = api.query_meta_info_by_index(index, '200')

        cifar10_cost_metrics = info.get_compute_costs('cifar10-valid')
        cifar10_flops = cifar10_cost_metrics['flops']
        cifar10_params = cifar10_cost_metrics['params']
        cifar10_latency = cifar10_cost_metrics['latency']
        # print(cifar10_flops, cifar10_params, cifar10_latency)
        # arch_dict['cifar10_flops'] = cifar10_flops
        # arch_dict['cifar10_params'] = cifar10_params
        arch_dict['cifar10_latency'] = cifar10_latency

        cifar100_cost_metrics = info.get_compute_costs('cifar100')
        cifar100_flops = cifar100_cost_metrics['flops']
        cifar100_params = cifar100_cost_metrics['params']
        cifar100_latency = cifar100_cost_metrics['latency']
        # print(cifar100_flops, cifar100_params, cifar100_latency)
        # arch_dict['cifar100_flops'] = cifar100_flops
        # arch_dict['cifar100_params'] = cifar100_params
        arch_dict['cifar100_latency'] = cifar100_latency

        image16_cost_metrics = info.get_compute_costs('ImageNet16-120')
        image16_flops = image16_cost_metrics['flops']
        image16_params = image16_cost_metrics['params']
        image16_latency = image16_cost_metrics['latency']
        # print(image16_flops, image16_params, image16_latency)
        # arch_dict['image16_flops'] = image16_flops
        # arch_dict['image16_params'] = image16_params
        arch_dict['imagenet16_latency'] = image16_latency

        for network_type in ['cifar10-valid', 'cifar100', 'ImageNet16-120']:
            total_flops, total_params, opt_flops, opt_params = calculate_cell_opt_flops_params(api, index, network_type)
            arch_dict['{}_total_flops'.format(network_type)] = total_flops
            arch_dict['{}_total_params'.format(network_type)] = total_params
            arch_dict['{}_opt_flops'.format(network_type)] = opt_flops
            arch_dict['{}_opt_params'.format(network_type)] = opt_params
        
        arch_dict['arch_str'] = arch_str

        dataset[index] = arch_dict

        print('***************************No. {} arch***************************'.format(index))

    assert len(dataset) == len(api), 'Wrong length of dataset'
    
    return dataset


def calculate_cell_opt_flops_params(api, index=0, network_type='cifar10-valid'):
    
    config = api.get_net_config(index, network_type)
    network = get_cell_based_tiny_net(config)

    img_sz = None
    if 'cifar' in network_type:
        img_sz = 32
    if 'ImageNet16-120' in network_type:
        img_sz = 16
    assert img_sz is not None, 'img_sz is None'
    inputs = torch.randn(1,3,img_sz, img_sz)
    
    network.eval()
    #1 cal total flops and params
    flops_obj = FlopCountAnalysis(network, inputs)
    total_flops = flops_obj.total()
    
    params_dict =  parameter_count(network)
    total_params = params_dict['']

    
    #2 extract each opt flops and params in each cell
    extract_op = lambda item:[NODE_TYPE_DICT[item[0]],item[1]]
    
    opts = api.str2lists(config['arch_str'])    # [(('nor_conv_3x3', 0),), (('nor_conv_3x3', 0), ('avg_pool_3x3', 1)), (('skip_connect', 0), ('nor_conv_3x3', 1), ('skip_connect', 2))]
    
    opts_type = []                              # [[3, 0], [3, 0], [4, 1], [1, 0], [3, 1], [1, 2]]
    for node_ops in opts:
        for op in node_ops:
            opts_type.append(extract_op(op))

    N = config['N']
    cells_idx_list = [i for i in range(N)] + [j+1+N for j in range(N)] + [k+2+2*N for k in range(N)]    # [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]
    flops_dict = flops_obj.by_module()

    opt_flops = {}
    opt_params = {}
    for cell_idx in cells_idx_list:
        cell_opt_flops = []
        cell_opt_params = []
        for opt_idx in range(len(opts_type)):
            key = 'cells.{}.layers.{}.op'.format(cell_idx, opt_idx)
            cell_opt_flops.append(int(flops_dict[key]))
            cell_opt_params.append(params_dict[key])
        opt_flops['cells{}'.format(cell_idx)] = cell_opt_flops
        opt_params['cells{}'.format(cell_idx)] = cell_opt_params
    
    return int(total_flops), total_params, opt_flops, opt_params


if __name__ == '__main__':

    start = time.time()

    api = API('./data/NAS-Bench-201-v1_1-096897.pth', verbose=False)
    
    save_path = './data'
    file_name = 'nasbench201_with_edge_flops_and_params.json'
    # file_name = 'target.json'
    save_file = os.path.join(save_path, file_name)
    if os.path.exists(save_file):
        os.remove(save_file)
    
    dataset = main(api)
    with open(save_file, 'w') as r:
            json.dump(dataset, r)
    r.close()

    print('all ok!!!!!!!!!!!!! using {} seconds'.format(time.time()-start))