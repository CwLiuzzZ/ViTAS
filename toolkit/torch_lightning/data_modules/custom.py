from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

class CustomDataModule(LightningDataModule):

    def __init__(self, hparams,train_dataset,valid_dataset=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print('{} samples found for training'.format(len(self.train_dataset)))
        if not self.valid_dataset is None:
            print('{} samples found for validation'.format(len(self.valid_dataset)))

    def train_dataloader(self):
        # sampler = RandomSampler(self.train_dataset,
        #                         replacement=True,
        #                         num_samples=self.hparams['batch_size'] * self.hparams['epoch_size'])
        # sampler = RandomSampler(self.train_dataset,
        #                         replacement=False)
        # return DataLoader(self.train_dataset,
        #                   sampler=sampler,
        #                   num_workers=self.hparams['num_workers'],
        #                   batch_size=self.hparams['batch_size'],
        #                   pin_memory=True,
        #                   persistent_workers=True)
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams['num_workers'],
                          batch_size=self.hparams['batch_size'],
                          pin_memory=False,
                          persistent_workers=False)

    def val_dataloader(self):
        if self.valid_dataset is None:
            return None
        return DataLoader(self.valid_dataset,
                          shuffle=False,
                          num_workers=self.hparams['num_workers'],
                        #   batch_size=self.hparams['batch_size'],
                          batch_size=1,
                        #   num_workers=1,
                          pin_memory=True,
                          persistent_workers=True)