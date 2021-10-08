from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import config

def SweepDataset(batch_size, transform):
    train_dataset = datasets.MNIST(root = '../MNIST', # 데이터 저장될장소 
                               train = True, # train인지test인지 
                               download = True, # 인터넷에서 다운로드해 이용할건지 
                               transform = transform) #이미지를 tensor 형태로 변환
                                                                  #0~255 를 0~1로 정규화까지 해줌

    test_dataset = datasets.MNIST(root = '../MNIST',
                                train = False,
                                download = True,
                                transform = transform)

    # Subset을 사용하면 Dataset의 부분 집합만 가져올 수 있음.
    train_sub_dataset = Subset(train_dataset, indices=range(0, len(train_dataset), 5))
    test_sub_dataset = Subset(test_dataset, indices=range(0, len(test_dataset), 5))

    train_loader = DataLoader(dataset = train_sub_dataset,
                            batch_size = batch_size,
                            shuffle = True)

    test_loader = DataLoader(dataset = test_sub_dataset,
                            batch_size = batch_size)

    return train_loader # , test_loader