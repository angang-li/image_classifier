# Flower classification

This project uses convolutional neural network to train an image classifier that is able to identify 102 different flower species with 93% testing accuracy. This image classifier can be used to identify flower species from new images, e.g., in a phone app that tells you the name of the flower your camera is looking at.

## 1. Data source

[102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). This dataset contains images of 102 different flower species. Here below are a few examples of the image data.

<img src='assets/Flowers.png' width=500px>

Data file structure:

- `flowers`: folder of image data.
    - `train`, `valid`, `test`: subfolders for training, validating, and testing the image classifier, respectively.
        - `1`, `2`, ..., `102`: 102 subfolders whose names indicate different flower categories. Given the large data size, only very few folders are included here.

## 2. Build an image classifier

[Code](Image_Classifier_Project.ipynb)

- #### Data loading and transformation

    - Training images were transformed (random resizing, rotation, cropping, and flipping) in order to improve model performance.

- #### Model building and training

    - Model is built from a pre-trained network model, Densenet-121 ([reference](https://arxiv.org/pdf/1608.06993.pdf)). An additional hidden layer was added to transfer learnings from the pre-trained model.

    - The model is trained on the training set using Adam (adaptive moment estimation) ([reference](https://arxiv.org/abs/1412.6980)). The loss and accuracy on the validation set are tracked to determine the best hyperparameters.

    - Finally, model is tested on the testing set.

    <img src="assets/inference_example2.png" width=300>

## 3. Build a command line application

- #### Train the image classifier

    [`train.py`](train.py): Train the image classifier, report validation accuracy along training, and save the trained model as a checkpoint.

    - Basic usage:
        - Specify directory of image data: `python train.py flowers`

    - Options:
        - Set directory to save checkpoints: `python train.py flowers --save_dir assets`
        - Choose architecture: `python train.py flowers --arch "vgg13"`
        - Set hyperparameters: `python train.py flowers --learning_rate 0.001 --hidden_units 512 --epochs 20`
        - Use GPU for training: `python train.py flowers --gpu`

- #### Identify flower name from a new image

    [`predict.py`](predict.py): Use the trained image classifier to predict flower name along with the probability of that name

    - Basic usage: 
        - Specify file path of the image and directory name of saved checkpoint: `python predict.py flowers/test/1/image_06743.jpg assets`

    - Options:
        - Return top K most likely classes: `python predict.py flowers/test/1/image_06743.jpg assets --top_k 3`
        - Use a mapping of categories to real names: `python predict.py flowers/test/1/image_06743.jpg assets --category_names cat_to_name.json`
        - Use GPU for inference: `python predict.py flowers/test/1/image_06743.jpg assets --gpu`
