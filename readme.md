## Image Generation using Autoencoders and GANs from Scratch 

This project explores the exciting world of image generation using two powerful deep learning techniques:

* **Autoencoders** - Neural networks trained to reconstruct their input data, learning efficient representations (compressions) in the process.
* **Generative Adversarial Networks (GANs)** -  A framework where two networks, a generator and a discriminator, compete against each other to produce increasingly realistic synthetic data.

We'll delve into these concepts, implement them from scratch using PyTorch, and train them on the CelebA dataset to generate human faces!

### Project Highlights

* **From the ground up:**  We'll build autoencoder and GAN models without relying on pre-trained components, gaining a deeper understanding of their inner workings.
* **CelebA Dataset:** The project leverages the CelebA dataset, a large-scale collection of celebrity faces, offering a rich source of visual data.
* **Hyperparameter Tuning:** We'll experiment with different hyperparameters to optimize model performance and generate visually appealing faces.
* **Written Analysis:** The project includes a written report section, where we'll reflect on the models' behavior and insights gained during experimentation.

### Getting Started

1. **Prerequisites**
   * Ensure you have PyTorch installed. If not, follow the installation instructions on the official website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   * Install the Pillow (PIL) library for image processing: `pip install Pillow`

2. **Clone the Repository**
   * Clone this repository to your local machine using:
   ```bash
   git clone https://github.com/your-username/image-generation-project.git
   ```

3. **Dataset**
   * Download the CelebA dataset and place it in a directory named `celebA` within your project folder. 

4. **Run the Code**
   * Navigate to the project directory and execute the Python script:
   ```bash
   python image_generation.py 
   ```
   * The code will train the autoencoder and GAN models, and display generated images along with training curves.
   * Feel free to experiment with different hyperparameters and model architectures!

### Code Structure

* **Data Loading:** The `load_celebA` function handles dataset loading and preprocessing.
* **Autoencoder:**
    * `Encoder` class: Defines the encoder network.
    * `Decoder` class: Defines the decoder network.
    * `Autoencoder` class: Combines the encoder and decoder into a complete autoencoder model.
    * `autoencoder_training` function: Implements the training loop for the autoencoder.
* **GAN:**
    * `Discriminator` class: Defines the discriminator network.
    * `Generator` class: Defines the generator network.
    * `training` function: Implements the training loop for the GAN, including separate updates for the generator and discriminator.
* **Hyperparameter Search:** 
    * `k_fold_split` function: Performs k-fold cross-validation for hyperparameter tuning
    * `randomized_grid_search` and `randomized_grid_search_gan` functions: Implement randomized grid search for both autoencoder and GAN.
* **Visualization:** The `plot_image` function displays images using matplotlib.

### Written Report

The `readme.md` file also includes a written report section, where you'll answer insightful questions about the models' behavior and your observations during experimentation.

### Acknowledgments

This project was inspired by various online resources and tutorials. We extend our gratitude to the authors and contributors of these valuable learning materials.

Let's embark on this journey of image generation and unlock the creative potential of deep learning! If you have any questions or suggestions, feel free to open an issue or contribute to the project. Happy coding! 
