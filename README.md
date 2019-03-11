# CancerDetection
https://www.kaggle.com/c/histopathologic-cancer-detection

# Dependencies 

For install dependencies. Run

```
pip3 install -r requirements.txt
```

# Testing

| Number of expetiment| Augmentation | Network architecture | Additional params | Accuracy
| ---- | ---- | ---- |---- | ---|
| 1 | A | ResNet18 | Epoch 18| 0.9028 |
| 2 | A | ResNet18 | Epoch 1 | 0.8897 |

## Types of augmentation

### A

```
# pretrained ImageNet network
T.ColorJitter(brightness=0.5, contrast=0.5),
T.RandomRotation((0, 5)),
T.Normalize(mean, std) # ImageNet
```
