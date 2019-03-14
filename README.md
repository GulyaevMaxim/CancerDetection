# CancerDetection
https://www.kaggle.com/c/histopathologic-cancer-detection

# Dependencies 

For install dependencies. Run

```
pip3 install -r requirements.txt
```

# Testing

Models are presented [here](https://drive.google.com/drive/folders/1aVX46pmBQUXB2lOooTg7t-0KaXFGUWhY?usp=sharing)

## Solo models

| Number of expetiment| Augmentation | Network architecture | Additional params | Accuracy
| ---- | ---- | ---- |---- | ---|
| 1 | A | ResNet18 | Epoch 18| 0.9028 |
| 2 | A | ResNet18 | Epoch 1 | 0.8897 |
| 3 | A | MobileNetV2 | Epoch 1 | 0.8892 |
| 4 | A | MobileNetV2 | Epoch 6 | 0.8850 |
| 5 | A | MobileNet   | Epoch 7 | 0.8839 |
| 6 | A | ResNet50   | Epoch 7 | 0.8948 |
| 7 | A | ResNet50   | Epoch 17 | 0.8949 |
| 8 | B | ResNet18   | Epoch 2 | 0.9016 |
| 9 | B | ResNet18   | Epoch 3 | 0.9044 |
| 10 | B | ResNet18   | Epoch 6 | 0.9072 |
| 11 | B | ResNet18   | Epoch 9 | 0.9080 |
| 12 | B | ResNet18   | Epoch 12 | 0.8985 |
| 13 | B | ResNet18   | Epoch 22 | 0.8878 |


## Ensembling

| Number of expetiment| Threshold | Number network from prev. table | Accuracy
| ---- | ---- | ---- |---- |
| 1 | Without thr | 1, 3, 6 - 12 | 0.9515 |
| 1 | > 0.8 = 1 | 1, 3, 6 - 12 | 0.9510 |
| 1 | > 0.5 = 1 | 1, 3, 6 - 12 | 0.9484 |

## Types of augmentation

### A

```
# pretrained ImageNet network
T.Resize((224,224))
T.ColorJitter(brightness=0.5, contrast=0.5),
T.RandomRotation((0, 5)),
T.Normalize(mean, std) # ImageNet
```

### B

```
# pretrained ImageNet network
T.Resize((224,224))
T.ColorJitter(brightness=0.5, contrast=0.5),
T.RandomRotation((-90, 90)),
T.RandomHorizontalFlip(p=0.5),
T.RandomVerticalFlip(p=0.5)
T.Normalize(mean, std) # ImageNet
```
