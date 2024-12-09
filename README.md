# Cucumber Disease Classification

This project provides a solution for identifying and classifying various diseases in cucumber plants using deep learning. The model is trained on a dataset of cucumber plant images and is capable of diagnosing different diseases based on the input image.

The `infer.py` script is used to run inference on a given image using a pre-trained model, predicting the disease present in the cucumber plant.

## Features

- **Image Inference**: Predict diseases in cucumber plants from input images.
- **Pre-trained Model**: Uses a pre-trained deep learning model to classify cucumber diseases.
- **Simple Interface**: Run inference with a simple command-line interface.

## Installation

You can install the required dependencies via `pip`:

```bash
pip install -r requirements.txt
```

## Usage

To run inference on an image using the pre-trained model, use the following command:

```bash
python infer.py --image_path <path_to_image> --model_path <path_to_model>
```

### Arguments

- `--image_path` (required): The path to the image file that you want to run inference on (e.g., `img.png`).
- `--model_path` (required): The path to the pre-trained model file (e.g., `model.pth`).

### Example Command

```bash
python infer.py --image_path img.png --model_path model.pth
```

This will load the image `img.png` and run inference using the model `model.pth`.

## Model

The model used in this project is a deep learning model trained on a dataset of cucumber plant images, each labeled with a specific disease. The model can classify the image into different categories, such as:

- **Anthracnose**
- **Bacterial Wilt**
- **Belly Rot**
- **Downy Mildew**
- **Fresh Cucumber**
- **Fresh Leaf**
- **Gummy Stem Blight**
- **Pythium Fruit Rot**

You can replace the `model.pth` file with your own trained model if needed, as long as it is compatible with the script.

## Example Inference Output

Once you run the inference command, the output will display the predicted class (disease) of the cucumber plant in the image, along with the confidence score. For example:

```
Predicted class: Downy Mildew
Confidence: 0.9996

Class probabilities:
Anthracnose: 0.0003
Bacterial Wilt: 0.0000
Belly Rot: 0.0000
Downy Mildew: 0.9996
Fresh Cucumber: 0.0000
Fresh Leaf: 0.0000
Gummy Stem Blight: 0.0000
Pythium Fruit Rot: 0.0000
```

## Notes

- Ensure that the input image is of good quality and clearly shows the cucumber plant for better prediction accuracy.
- The model may not perform well on images significantly different from the training dataset (e.g., poor lighting, different angles, etc.).

## Training the Model

If you wish to train the model on your own dataset of cucumber images, you can modify or create a new training script (main.ipynb) based on the dataset you have. Make sure to preprocess the images in the same way that the pre-trained model was trained.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
