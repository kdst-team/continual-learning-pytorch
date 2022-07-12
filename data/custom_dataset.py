from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
from PIL import Image
class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transform=None,target_transform=None,no_return_target=False,return_index=False):
        self.X = images
        self.y = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader=default_loader
        self.no_return_target=no_return_target
        self.return_index=return_index
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i]

        if isinstance(data,tuple) or isinstance(data,list):
            path, target = data
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
                
            if self.no_return_target:
                if self.return_index:
                    return sample, i
                return sample
            else:
                if self.return_index:
                    return sample, target, i
                return sample, target
        
        if self.transform:
            data = self.transform(data)
            
        if self.y is not None:
            if self.no_return_target:
                if self.return_index:
                    return data, i
                return data
            else:
                if self.return_index:
                    return data, self.y[i], i
                return (data, self.y[i])
        else:
            if self.return_index:
                return data, i
            return data

class ImageDatasetFromLoc(Dataset):
    def __init__(self, images_loc_list,transform=None,target_transform=None,return_idx=False):
        self.X = images_loc_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader=default_loader
        self.return_idx=return_idx

    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, index):
        data = self.X[index]
        try:
            path, target = data
        except:
            path = data
            target = None
            
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if target is not None:

            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.return_idx:
                return sample, target, index
            return sample, target
        else:
            if self.return_idx:
                return sample, index
            return sample


class ImageDatasetFromData(Dataset):
    def __init__(self, images, labels=None, transform=None,target_transform=None, return_idx=False):
        self.X = images
        self.y = labels
        self.transform = transform
        self.target_transform = target_transform
        self.return_idx=return_idx
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data=Image.fromarray(self.X[i])
        
        if self.transform:
            data = self.transform(data)
        
        if self.target_transform:
            target = self.target_transform(target)

        if self.y is not None:
            if self.return_idx:
                return data, self.y[i], i
            return (data, self.y[i])
        else:
            if self.return_idx:
                return data, i
            return data