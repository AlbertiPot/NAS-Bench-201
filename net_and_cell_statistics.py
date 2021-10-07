from nas_201_api import NASBench201API as API

api = API('./data/NAS-Bench-201-v1_1-096897.pth', verbose=False)

for index, arch_str in enumerate(api):
    
    cifar10_valid_dict = api.get_more_info(index, 'cifar10-valid', 199, hp='200', is_random=False)
    cifar10_dict = api.get_more_info(index, 'cifar10', 199, hp='200', is_random=False)  
    cifar10_val_acc = cifar10_valid_dict['valid-accuracy']
    cifar10_test_acc = cifar10_dict['test-accuracy']
    # print(cifar10_val_acc)
    # print(cifar10_test_acc)

    cifar100_dict = api.get_more_info(index, 'cifar100', 199, hp='200', is_random=False)
    cifar100_dict_val_acc = cifar100_dict['valid-accuracy']
    cifar100_dict_test_acc = cifar100_dict['test-accuracy']
    # print(cifar100_dict_val_acc)
    # print(cifar100_dict_test_acc)
    
    imagenat16_dict = api.get_more_info(index, 'ImageNet16-120', 199, hp='200', is_random=False)
    imagenat16_val_acc = imagenat16_dict['valid-accuracy']
    imagenat16_test_acc = imagenat16_dict['test-accuracy']
    # print(imagenat16_val_acc)
    # print(imagenat16_test_acc)

    info = api.query_meta_info_by_index(index, '200')

    cifar10_cost_metrics = info.get_compute_costs('cifar10-valid')
    cifar10_flops = cifar10_cost_metrics['flops']
    cifar10_params = cifar10_cost_metrics['params']
    cifar10_latency = cifar10_cost_metrics['latency']
    # print(cifar10_flops, cifar10_params, cifar10_latency)

    cifar100_cost_metrics = info.get_compute_costs('cifar100')
    cifar100_flops = cifar100_cost_metrics['flops']
    cifar100_params = cifar100_cost_metrics['params']
    cifar100_latency = cifar100_cost_metrics['latency']
    # print(cifar100_flops, cifar100_params, cifar100_latency)

    image16_cost_metrics = info.get_compute_costs('ImageNet16-120')
    image16_flops = image16_cost_metrics['flops']
    image16_params = image16_cost_metrics['params']
    image16_latency = image16_cost_metrics['latency']
    # print(image16_flops, image16_params, image16_latency)