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
| 14 | B | DenseNet169   | Epoch 3 | 0.8999 |
| 15 | B | DenseNet169   | Epoch 5 | 0.9084 |
| 16 | B | DenseNet169   | Epoch 9 | 0.9042 |
| 17 | B | MeNet456   | Epoch 13 | 0.8975 |
| 18 | B | MeNet456   | Epoch 18 | 0.8947 |
| 19 | B | MeNet456   | Epoch 20 | 0.9047 |
| 20 | B | MeNet456   | Epoch 21 | 0.9023 |
| 21 | B | MeNet456   | Epoch 29 | 0.9103 |
| 22 | B | MeNet456   | Epoch 31 | 0.9033 |
| 23 | C | MeNet456*   | Epoch 3 + Output 1x1 + Dropout | 0.9510 |
| 24 | C | Dense121*   | Epoch 20(Last Layer) + Epoch 20 + Output 1x1 + Dropout | 0.9745 |
| 25 | C | Dense169*   | Epoch 23 + Output 1x1 + Dropout | 0.9710 |
| 26 | C | ResNet18*   | Epoch 17 + Output 1x1 + Dropout | 0.9684 |
| 27 | C | ResNet18*   | Epoch 18 + Output 1x1 + Dropout | 0.9685 |
| 28 | C | DenseNet121*   | Input 96x96 Epoch 96 + Output 1x1 + Dropout | 0.9689 |
| 29 | C | DenseNet121*   | Input 96x96 Epoch 20 + Output 1x1 + Dropout | 0.9665 |
| 30 | C | DenseNet121*   | Input 96x96 Epoch 14 + Output 1x1 + Dropout | 0.9655 |

* We use these layers instead fully-connected layer (layer with output 1000 ImageNet classes) for modeles marked `*`

```
    num_ftrs = net.fc.in_features
    out_ftrs = int(net.fc.out_features / 4)
    net.fc = nn.Sequential(
        nn.Sigmoid(),
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, out_ftrs, bias=True),
        nn.SELU(),
        nn.Dropout(0.7),
        nn.Linear(in_features=out_ftrs, out_features=1, bias=True),
    )
```

## Ensembling

| Number of expetiment| Threshold | Number network from prev. table | Accuracy
| ---- | ---- | ---- |---- |
| 1 | Without threshold | 1, 3, 6 - 12 | 0.9515 |
| 2 | > 0.8 = 1 | 1, 3, 6 - 12 | 0.9510 |
| 3 | > 0.5 = 1 | 1, 3, 6 - 12 | 0.9484 |
| 4 | Without threshold | 1, 9 - 11 | 0.9345 |
| 5 | Without threshold | 1, 3, 6 - 12, 14-16 | 0.9539 |
| 6 | Without threshold | 1, 3, 7 - 11, 14-16 | 0.9521 |
| 7 | Without threshold | 1, 3, 6 - 12, 14-16, 19-22 | 0.9577 |
| 8 | Without threshold | 1, 8-11, 15, 16, 19-22 | 0.9544 |
| 9 | Without threshold | 24, 25 | 0.9746 |
| 10 | Without threshold | 6 TTA of Dense121* | 0.9752 |
| 11 | Without threshold | 23-27 | 0.9771 |
| 11 | Without threshold | 24-30 | 0.9783 |


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

### C

```
# pretrained ImageNet network
albumentations.Resize(224, 224)
albumentations.RandomRotate90(p=0.5),
albumentations.Transpose(p=0.5),
albumentations.Flip(p=0.5),
albumentations.OneOf([
albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(),
albumentations.RandomBrightness(), albumentations.RandomContrast(),
albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5),
albumentations.HueSaturationValue(p=0.5),
albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
albumentations.Normalize(),
```
