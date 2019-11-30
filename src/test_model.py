import torch

if __name__ == '__main__':
    import re

    checkpoint = torch.load('models/ctdet_pascal_resdcn18_512.pth', map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    # convert data_parallal to model
    for k in state_dict_:
        print(k)
        if re.match(r'.+?num_batches_tracked$', k):
            print(state_dict_[k])
