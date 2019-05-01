# Configuration file structure

## Train

| Argument| Type | Description | Default value |
| ---- | ---- | ---- |---- |
| `batch_size` | int | Batch_size | No|
| `width` | int | Weight image size for net | No |
| `height` | int | Height image size for net | No |
| `dest_model` | str | Path to save model | No |
| `src_model` | str | Path to model | No |
| `train_csv` | str | Path to train data CSV   | No |
| `data_train` | str | Path to train images for dataset   | No |
| `valid_csv` | str | Path to validate data CSV   | No |
| `data_validate` | str | Path to validate images for dataset   | No |
| `cuda` | bool | Use cuda?   | False |
| `st_epoch` | bool | Number epoch of start   | 0 |
| `do_epoch` | bool | How much epoch still do   | 1000 |

## Test

| Argument| Type | Description | Default value |
| ---- | ---- | ---- |---- |
| `batch_size` | int | Batch_size | No|
| `width` | int | Weight image size for net | No |
| `height` | int | Height image size for net | No |
| `model` | str | Path to model | No |
| `data_csv` | str | Path to data CSV   | No not necessary|
| `data` | str | Path to images for dataset   | No |
| `out_path` | str | Path to  out file with submission   | No |
| `cuda` | bool | Use cuda?   | False |
