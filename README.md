
# Synthetic Data Generation with Generative AI

This project demonstrates how to use Generative Adversarial Networks (GANs) to generate synthetic data. The dataset used includes daily records of app usage, and the model generates artificial data that closely mimics real-world data.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Training the GAN](#training-the-gan)
- [Generating Synthetic Data](#generating-synthetic-data)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project uses a dataset with app usage details and applies a GAN model to generate synthetic data. The GAN consists of:

- **Generator:** Produces synthetic data.
- **Discriminator:** Evaluates the authenticity of the data (real or synthetic).

The two networks train together, iteratively improving the quality of the generated data.

## Project Structure

This project is executed entirely within Google Colab. Files and dependencies are managed as part of the Colab environment.

```plaintext
├── Screentime-App-Details.csv      # Dataset file (uploaded to Colab)
├── Synthetic_Data_Generation.ipynb # Google Colab notebook
├── README.md                       # Project documentation
```

## Technologies Used

- Python
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn
- Google Colab

## Setup Instructions

1. Open the Google Colab notebook:
   [Synthetic_Data_Generation.ipynb](#)

2. Upload the dataset file (`Screentime-App-Details.csv`) to the Colab environment.

3. Install required libraries (if not already installed):
   ```python
   !pip install tensorflow pandas numpy scikit-learn
   ```

## Usage

### Preprocessing the Dataset

1. Load the dataset and normalize the data:
   ```python
   from sklearn.preprocessing import MinMaxScaler
   import pandas as pd

   data = pd.read_csv('Screentime-App-Details.csv')
   data_gan = data.drop(columns=['Date', 'App'])
   scaler = MinMaxScaler()
   normalized_data = scaler.fit_transform(data_gan)
   ```

### Training the GAN

1. Define the Generator and Discriminator models directly in the notebook.
2. Train the GAN using the provided training loop in the notebook.

### Generating Synthetic Data

1. After training, use the generator to create synthetic data samples by running the corresponding cells in the notebook.

## Training the GAN

The training loop involves:
- Generating fake data using the generator.
- Combining it with real data to train the discriminator.
- Training the generator to improve the quality of the synthetic data.

Progress is displayed in real-time within the notebook, including discriminator and generator losses.

## Generating Synthetic Data

After training, the generator model can produce synthetic data samples by inputting random noise vectors. The generated data is scaled back to the original range for practical use.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
