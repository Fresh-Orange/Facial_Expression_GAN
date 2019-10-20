import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    
    if opt.dataset_mode == 'keypoint':
        from data.keypoint import KeyDataset
        dataset = KeyDataset()
        
    elif opt.dataset_mode == 'blend_keypoint':
        from data.blend_keypoint import KeyDataset
        dataset = KeyDataset()

    elif opt.dataset_mode == 'test_keypoint':
        from data.test_keypoint import KeyDataset
        dataset = KeyDataset()

    elif opt.dataset_mode == 'test_blend_keypoint':
        from data.test_blend_keypoint import KeyDataset
        dataset = KeyDataset()

    elif opt.dataset_mode == 'keypoint_generate_blend':
        from data.keypoint_generate_blend import KeyDataset
        dataset = KeyDataset()

    elif opt.dataset_mode == 'blend_keypoint2':
        from data.blend_keypoint2 import KeyDataset
        dataset = KeyDataset()

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
