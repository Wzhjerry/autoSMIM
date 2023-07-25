from importlib import import_module


def load_dataset(args, fold=0, train=True, aug_k=40, aug_n=1, patch=False):
    print("=> creating dataset [{}], fold {}...".format(args.dataset_name, fold))
    m = import_module("datasets." + args.dataset_name.lower())
    if train:
        # training mode
        train_dataset, val_dataset = m.load_dataset(
            args, fold, train, aug_k=aug_k, aug_n=aug_n
        )
        return train_dataset, val_dataset
    else:
        # testing mode
        test_dataset = m.load_dataset(
            args, fold, train, aug_k=aug_k, aug_n=aug_n
        )
        return test_dataset
