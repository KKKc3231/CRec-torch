# torch dataset
from torch.utils.data import Dataset, DataLoader

class RentCarDataSet(Dataset):
    def __init__(self, data):
        
        # data: data为dict类型    
        self.num_features = data.get('num_features', None)
        self.cat_features = data.get('cat_features', {})
        self.click_label = data.get('click_label', None)
        self.order_label = data.get('order_label', None)
        self.length = 0

        # get length
        if self.num_features is not None:
            self.length = len(self.num_features)
        elif self.cat_features:
            first_cat_feature = next(iter(self.cat_features.values()))
            self.length = len(first_cat_feature)
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 获取单个样本
        sample = {}                 
        if self.num_features is not None:
            sample['num_features'] = self.num_features[idx]
        
        # cat
        cat_features = {}
        for name, feature in self.cat_features.items():
            cat_features[name] = feature[idx]
        sample['cat_features'] = cat_features
        
        # label
        sample['click_label'] = self.click_label[idx]
        sample['order_label'] = self.order_label[idx]
        
        return sample

class RentCarSeqDataSet(Dataset):
    
    def __init__(self, data):
    
        # data: data为dict类型    
        self.num_features = data.get('num_features', None)
        self.cat_features = data.get('cat_features', {})
        self.items_id = data.get('item_id', None)
        self.seqs_id = data.get('seq_id', None)
        self.click_label = data.get('click_label', None)
        self.order_label = data.get('order_label', None)
        self.length = 0

        # get length
        if self.num_features is not None:
            self.length = len(self.num_features)
        elif self.cat_features:
            first_cat_feature = next(iter(self.cat_features.values()))
            self.length = len(first_cat_feature)
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 获取单个样本
        sample = {}          
        # name       
        if self.num_features is not None:
            sample['num_features'] = self.num_features[idx]
        
        # cat
        cat_features = {}
        for name, feature in self.cat_features.items():
            cat_features[name] = feature[idx]
        sample['cat_features'] = cat_features
        
        # item_id && seq_id
        sample['item_id'] = self.items_id[idx]
        sample['seq_id'] = self.seqs_id[idx]

        # label
        sample['click_label'] = self.click_label[idx]
        sample['order_label'] = self.order_label[idx]
        
        return sample