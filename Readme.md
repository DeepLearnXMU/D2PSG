## D2PSG

This repository includes codes for the paper ''D2PSG: Multi-Party Dialogue Discourse Parsing as Sequence Generation".

#### Data Preparation
* Download datasets (Molweni / STAC) and place under the directory.
* Run "preprocess_data_\*.py". "\* = seq" means for D2PSG-FH model. "\* = stac" means for STAC data. Set the "add_description = True" in scripts as your need.
* The processed data includes two files. "\*.json" will be feeded to model and "\*.idx" will be used for testing.

#### Training
* Use the scripts ("train_\*.sh") to run the models. We use the same setting for the two datasets. For small/base/large, the training steps are no less than 9k/7k/3k (you can set a large epoch directly). Besides, you may need to adjust the per_device_train_batch_size and gradient_accumulation according to your GPU memory size (final_batch_size=per_device_train_batch_size\*gradient_accumulation on single GPU).

#### Testing
* Use "test.sh" to get predictions. Evaluate each saved checkpoint to decide the final model. Usually, the best checkpoint for small/base/large is around 8k/6k/2k.
* Run "postprocess_\*.py" to calculate final F1 scores. Note to set correct path of "generated_predictions.txt" and "\*.idx".