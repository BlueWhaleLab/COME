import numpy as np
import random

seed = 2023

random.seed(seed)
np.random.seed(seed)


def monotone_shift_constructor(q1, q2):
    def monotone_shift(T):
        lamb = 1.0 / (T-1)
        return np.concatenate([np.expand_dims(q1 * (1 - lamb * t) + q2 * lamb * t, axis=0) for t in range(T)], axis=0)
    return monotone_shift


def generate_sample_indices_and_ys(q_all, dataset_name='imagenet1k'):
    np.random.seed(2022)
    if dataset_name == 'imagenet1k':
        num_classes = 1000
        tset_length = 50000
        num_each_class = 50
    elif dataset_name == 'cifar10':
        num_classes = 10
        tset_length = 10000
        num_each_class = 1000
    else:
        assert False, "not supported, now only support imagenet1k"
    ys = np.squeeze(np.asarray([np.random.choice(num_classes, 1, p=q) for q in q_all]))

    print((ys == 3).sum())
    print((ys == 5).sum())
    print((ys == 6).sum())

    print(q_all[:3,:])
    print(ys[:100])


    num_tests = len(ys)
    tset_indices = np.array([i for i in range(tset_length)])
    tset_ys = np.array([i // num_each_class for i in range(tset_length)])
    
    generated_indices = np.zeros([num_tests]) 
    
    for i in range(num_classes):
        num_i = (ys == i).sum()
        if num_i == 0:
            continue
        num_test_i = (tset_ys == i).sum()
        sampled_indices = np.random.randint(0, num_test_i, num_i)
        sampled_indices = tset_indices[tset_ys == i][sampled_indices]
        
        generated_indices[ys == i] = sampled_indices
    return generated_indices

for myir in [10]:
    shift_proccess_name = "per_class_shift" 
    T = 100000 
    dataset_name = 'imagenet1k'

    if dataset_name == 'imagenet1k':
        num_classes = 1000
    elif dataset_name == 'cifar10':
        num_classes = 10



    if shift_proccess_name == "per_class_shift" and dataset_name == "imagenet1k":
        imbalance_ratio = myir
        shuffle_class_order = "yes"
        minor_class_prob = 1 / (imbalance_ratio + num_classes - 1)
        major_class_prob = minor_class_prob * imbalance_ratio
        q_for_all_classes = np.ones([num_classes, num_classes]) * minor_class_prob
        print(q_for_all_classes.shape)
        for i in range(num_classes):
            q_for_all_classes[i, i] = major_class_prob
        if shuffle_class_order == "yes":
            indices = list(range(num_classes))
            random.shuffle(indices)
            q_for_all_classes = q_for_all_classes[indices,:]
        def shift_proccess(T):
            num_for_repeat_each_q = T // num_classes
            assert num_for_repeat_each_q > 0, "T should greater than number of classes"
            return np.concatenate([np.expand_dims(q_for_all_classes[i,:], axis=0) for i in range(num_classes) for _ in range(num_for_repeat_each_q)], axis=0)
    else:
        assert False, NotImplementedError

    q_all = shift_proccess(T)

    print(q_all.shape)

    simulated_indices = generate_sample_indices_and_ys(q_all, dataset_name=dataset_name)

    print(simulated_indices.shape)
    print(simulated_indices[:100])

    print(list(simulated_indices[:10]))

    np.save('seed{}_total_{}_ir_{}_class_order_shuffle_{}'.format(seed, T, imbalance_ratio, shuffle_class_order), simulated_indices)



