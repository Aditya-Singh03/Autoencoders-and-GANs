# %% [markdown]
# ### Homework 4 — CompSci 389 — University of Massachusetts — Spring 2022
# Assigned: April 1, 2024;  Due: April 9, 2024 @ 11:59 PM EST
# 
# ## Instructions!
# 
# In this assignment, you'll be using PyTorch again.  If you need to install PyTorch again, you can find instructions to install it [here](https://pytorch.org/get-started/locally/). Some Windows users have issue using pip to install it so I recommend in that case to use [anaconda](https://docs.conda.io/en/latest/miniconda.html). 
# 
# This time, we'll be using PyTorch to implement two types of neural network -- these are pretty cool.
# 
# The first of these is going to be an *autoencoder*.  An autoencoder is a neural network with a pretty unique structure, which will learn a function that can map an input to itself.  This allows the network to extract important features from an input to effectively compress it, and then reconstruct those important features back into something that approximates the original input closely.
# 
# The second of these is going to be a *GAN (generative adversarial network)*, which is a framework where we train two neural networks - one learns to generate synthetic data based on training data (the generator), and the other learns to distinguish true data from generated data (the discriminator).  This is where the word *adversarial* comes into the picture -- as the discriminator gets better at identifying fake data, the feedback is passed to the generator, which gets better at creating synthetic data.
# 
# One example of a cutting edge GAN that you can quickly check out is [This Person Does Not Exist](https://thispersondoesnotexist.com), which generates synthetic portraits of people!

# %%
# Step 1: Let's import some libraries!
import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import random
from tqdm import tqdm

# %%
# This is a new dependency to load this dataset, so make sure you install this module too
from PIL import Image

# %% [markdown]
# ## Our Dataset
# For this project we're going to use **human faces**! Well, *images* of human faces, but it's still cool. 
# 
# The particular dataset is calleb **CelebA** -- you can find the official website [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and you can find the torchvision documentation [here](http://pytorch.org/vision/main/generated/torchvision.datasets.CelebA.html)
# 

# %%
def load_celebA(batch_size=32, train=True):
    '''
    Dataset loading will be handled for you automatically since it's a bit of a pain
    to work with these large datasets and I'll just give you a subset
    '''

    dataset = []
    batch_counter = 0
    batch = []

    for file in tqdm(os.listdir('./celebA')):
        
        img = Image.open('./celebA/' + file)
        img = np.asarray(img).reshape(3, 109, 89)

        if batch_counter < batch_size:
            batch.append(img)
            batch_counter += 1
        else:
            dataset.append(np.array(batch))
            batch = []
            batch_counter = 0

    return np.array(dataset)


# %% [markdown]
# ### Now let's see what our data looks like!

# %%
def plot_image(image):
    
    '''
    Takes in an image and shows it using matplotlib 
    this is used to visualize the data and also the outputs of our network
    '''

    image = image.reshape(-1, 109,89,3)

    plt.imshow(image[0])
    return

# %%
# This will load the dataset and set the dataset variable to it
# Try to only run this code once since it takes a while (though you may again to change the batch size)

dataset = load_celebA(batch_size=128, train=True)
dataset = torch.from_numpy(dataset)


# %%
# This just displays a random image from the dataset 
ex_image = dataset[random.randint(0,77)]
print("image shape:", ex_image.shape)

plot_image(ex_image)

# %% [markdown]
# ## Autoencoder (40 points)
# 
# Just like we did in HW3, we're doing to build a PyTorch ```nn.Module``` for our autoencoder.  Again, this consists of 2 parts: The initialization (defined in ```__init__()``` -- note that this the python convention for initalizing classes) and the forward pass (defined aptly as ```forward()```)
# 
# Since we're using PyTorch, we can simply define this module and then the gradient can be found *automatically*.
# 
# Documentation for a pytorch module can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
# 
# ##### Adam's little comment: guessing we'll want to put like a description of how to set up layers for the autoencoder here, depends on how much of a bottleneck we want.  I put some starter code below for an autoencoder on MNIST from this tutorial (minor changes for code readability): https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
# 
# Leaving this description blank for now since we haven't covered autoencoder in class...
# 
# 
# 
# 
# In our first model we will just be creating a perceptron which will use a single ```nn.Linear()``` module -- you can find documentation for that [here](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
# 
# In later models we'll use nonlinearities (and that neat convolution thing) -- documentation ```for nn.ReLU()``` can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
# 

# %% [markdown]
# ### Autoencoder Model (10 points)
# 
# Here you are going to make the module for your encoder and decoder -- the encoder will take an image as an input and compress it to some size (its a hyperparameter) and then the decoder will take that compressed (latent) representation of the image and try to reconstruct the original

# %%
class Encoder(nn.Module):

    '''
    This will be the module for your encoder half of your autoencoder
    You may use Linear layers, conv2d layers and anything else you'd like (you just need the first two)

    One thing that might throw you is that input_shape is going to be a tuple which represents the 
    entire shape of the input (i.e. an image is (3,109,89) in CelebA)
    ### TODO celebB change here
    '''

    def __init__(self, input_shape, compression_size):
        super().__init__()

        self.to("cuda")


        # TODO initialize the Encoder network 
        # You must use nn.Conv2d, nn.Linear, and nn.ReLU but you can look for anything else you'd like to use

        #################################################### 
        # 3 channels, 109x89
        self.layers = nn.Sequential(
                                    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # 16 channels, 109x89
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 channels, 55x45
                                    nn.ReLU(), 
                                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 channels, 28x23
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(28*23*64, compression_size) # Linear layer to compress the data to the compression size
        )






        #################################################### 

        self.flatten = nn.Flatten()



    def forward(self, features):

        features = features.view(-1, 3, 109, 89).float()
        features = features.to("cuda")
        
        #################################################### 
        out = self.layers(features)






        #################################################### 

        return out.to("cpu")


# %%
encoder = Encoder((3,109,89), 100).to("cuda")
test_out = encoder(ex_image.to("cuda"))

print(test_out.shape)
print("the shape of the output should be a vector of size (batch_size,100), is it?")


# %%
class Decoder(nn.Module):
    '''
    This is the other bit of the autoencoder
    Likewise you can use whatever you'd like to get from the output of the encoder
    (which should be a vector )
    '''

    def __init__(self, input_size, output_shape):
        super().__init__()

        self.to("cuda")
        

        self.output_shape = output_shape

        # TODO initialize your Decoder
        # You can use Linear or conv layers as you'd like, but since we are expanding the size
        # You may want to look into deconvolution, which is called nn.Conv2Transpose

        ###############################################
        self.layers = nn.Sequential(
                            nn.Linear(input_size, 28*23*64),
                            nn.ReLU(),
                            nn.Unflatten(1, (64, 28, 23)), # 64 channels, 28x23
                            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), # 32 channels, 28x23
                            nn.ReLU(),
                            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1), # 16 channels, 55x45
                            nn.ReLU(),
                            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1), # 3 channels, 109x89
                            # nn.Sigmoid()
                            )




        ###############################################
        



    def forward(self, x):

        # TODO finish the forward pass of your Decoder
        # Input is the output of the encoder


        ###############################################

        x = x.to("cuda")
        out = self.layers(x)


        ###############################################
        
        return out.to("cpu")

# %%
decoder = Decoder(100, (3,109,89)).to("cuda")
test_out = decoder(torch.from_numpy(np.ones((128,100))).float())

print(test_out.shape)
print("the shape of the output should be shape (batch_size,3,109,89), is it?")

# %%
class Autoencoder(nn.Module):

    '''
    This will combine your encoder and decoder modules together
    with their powers combined they make an AUTOENCODER

    You may have issue with the shape of the input to the decoder
    remember that we pass in compression_size which will just be an int
    '''
    
    def __init__(self, input_shape, compression_size):
        super().__init__()
        self.to("cuda")
        
        self.input_size = input_shape

        self.encoder = Encoder(input_shape, compression_size)
        self.decoder = Decoder(compression_size, input_shape)
        self.relu = nn.ReLU()
        
        

    def forward(self, features):
        features = features.to("cuda")
        
        out = self.encoder(features)
        out = self.relu(out)
        out = self.decoder(out)
        
        return out.to("cpu")

# %%
# Shows the prediction of the autoencoder without training
# Not very good huh? (though theres a small chance it is lol)

input_shape = (3,109,89)

test_model = Autoencoder(input_shape, 100).to("cuda") # Takes input of celebA image size and encodes it to size 100 and then decodes it
test_output = test_model(ex_image)

print("The original image")
plot_image(ex_image.byte())
plt.show()
print("Your reconstruction")
plot_image(test_output.detach().byte())

# %% [markdown]
# ### Loss and Optimizer for Autoencoder (5 points)
# 
# The loss for our autoencoder is nice and easy since we can just compare the input and output directly -- MSE, MAE or any other loss you'd like would work, though you can try multiple to see how they behave (or make your own if you're a nerd)
# 
# Likewise we can use 
# 
# 
# 

# %%
## Fill in the loss_function and optimizer below and run this cell to see if they are valid!

model = Autoencoder(input_shape, 100).to("cuda")
ex_image = ex_image.float().view(-1,3,109,89)                               

# TODO fill out the loss_function and optimizer

#############################################

loss_function = nn.MSELoss()                                                    ## You can choose whichever loss function you'd like
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)                                                          ## You can use any optimizer for this, which is defined in torch.optim -- look up some API stuff

#############################################

# This checks that your model, loss and optimizer are valid
print("BEFORE GRADIENT STEP:")
ex_pred = model(ex_image)
ex_label = ex_image


optimizer.zero_grad() # Sets the gradient to 0 so that gradients don't stack together

ex_loss1 = loss_function(ex_pred, ex_label)
print("loss",ex_loss1.item())

ex_loss1.backward() # This gets the gradient of the loss function w.r.t all of your model's params

print()
print("AFTER GRADIENT STEP:")
optimizer.step() # This takes the step to train

ex_pred = model(ex_image)
ex_label = ex_image

ex_loss2 = loss_function(ex_pred, ex_label)
print("loss",ex_loss2.item())

print()
print("Difference in loss:", (ex_loss1 - ex_loss2).item())
print("This should be some positive number to say we reduced loss")


# %% [markdown]
# ### Training Loop (10 points)
# We're ready to train our autoencoder! Complete the ```training()``` function, just like in HW3. You can iterate over your data for 30 epochs to start. 
# 
# [Hint for reseting the optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad)
# 
# [Hint for stepping with the optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step) (You'll have to use .backward() to get the gradient)
# 
# At this point you should record your training and validation *losses* and *accuracies* **(four lists in total)**. You'll need these values for the written section, where you will be plotting them.

# %%
def autoencoder_training(model, loss_function, optimizer, train_data, n_epochs, update_interval):

    '''
    Updates the parameters of the given model using the optimizer of choice to
    reduce the given loss_function

    This will iterate over the dataloader 'n_epochs' times training on each batch of images
    
    To get the gradient (which is stored internally in the model) use .backward() from the loss tensor
    and to apply it use .step() on the optimizer

    In between steps you need to zero the gradient so it can be recalculated -- use .zero_grad for this
    '''
    
    losses = []

    for n in range(n_epochs):
        for i, image in enumerate(tqdm(train_data)):

            image = image.float().view(-1,3,109,89)

            # TODO Complete the training loop using the instructions above
            # Hint: the above code essentially does one training step

            ##############################################################
            optimizer.zero_grad()
            my_output = model(image)
            loss = loss_function(my_output, image)
            loss.backward()
            optimizer.step()







            ##############################################################
        
            if i % update_interval == 0:
                losses.append(round(loss.item(), 2)) # This will append your losses for plotting -- please use "loss" as the name for your loss
        
    return model, losses



# %%
# Plug in your model, loss function, and optimizer 
# Try out different hyperparameters and different models to see how they perform

lr = 0.0002                # The size of the step taken when doing gradient descent
batch_size = 40         # The number of images being trained on at once
update_interval = 10     # The number of batches trained on before recording loss
n_epochs = 10             # The number of times we train through the entire dataset
compression_size = 200    # This is the size of the bottleneck (compression point) of the autoencoder

input_shape = (3,109,89)

dataset = dataset      # The dataset is a pain to load/unload so we want to keep it active

model = Autoencoder(input_shape, compression_size).to("cuda") 
loss_function = nn.MSELoss()                        
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

# autoencoder_training_opt = torch.compile(autoencoder_training, mode="reduce-overhead") # This is compiles the training loop so that it runs faster (the first run will take much longer)

trained_model, losses = autoencoder_training(model, loss_function, optimizer, dataset, n_epochs=n_epochs, update_interval=update_interval)

plt.plot(np.arange(len(losses)) * batch_size * update_interval, losses)
plt.title("training curve")
plt.xlabel("number of images trained on")
plt.ylabel("Reconstruction loss")
plt.show()


# NOTE: It will take a while for this to train (depending on your model)
# You can increase the batch size (way up top) or reduce the size of your model if it takes too long


# %%
# Displays the reconstruction of the (now trained) autoencoder on the same example image
# Notice that it worked and we have a better prediction (if your code works)
ex_image = dataset[random.randint(0,77)]
trained_output = trained_model(ex_image)

print("original image:")
plot_image(ex_image)
plt.show()
print("your (trained) reconstruction")
plot_image(trained_output.detach().byte())

# NOTE: It is very likely that this won't look that good at first
# It may even give the same output independent of the input -- rerun this a few times to check
# Play with the hyperparameters until you're happy with the output

# %% [markdown]
# ### Probably not great huh ^
# 
# Let's tune our hyperparameters so that we get something a bit cooler!

# %% [markdown]
# ### Testing and HyperParameter Search (10 points)
# 
# Since the testing loop and training loop are so similar I'm going to go ahead and just give it to you -- but you gotta promise to at least look at the method to see how similar they are! 

# %%
def testing(model, loss_function, test_data):

    '''
    This function will test the given model on the given test_data
    it will return the accuracy and the test loss (given by loss_function) 
    '''
    
    sum_loss = 0

    for i, image in enumerate(tqdm(test_data)):

        # This is essentially exactly the same as the training loop 
        # without the, well, training, part
        
        image = image.float()
        pred = model(image)
        loss = loss_function(pred, image)
        sum_loss += loss.item()
    
    avg_loss = round(sum_loss / len(test_data), 2)

    print("test loss:", avg_loss )

    return avg_loss

def train_and_test(model, loss_function, optimizer, batch_size, update_interval, n_epochs, train_dataset, test_dataset):

    '''
    This will use your/my methods to create a dataloader, train a gven model, and then test its performance

    Again, since I gave this to you for free you have to promise to look at it and try to understand it
    '''


    # training_opt = torch.compile(autoencoder_training, mode="reduce-overhead") # Once again we are compiling the training loop for speedup

    trained_model, losses = autoencoder_training(model, loss_function, optimizer, train_dataset, n_epochs=n_epochs, update_interval=update_interval)

    test_loss = testing(trained_model, loss_function, test_dataset)

    plt.plot(np.arange(len(losses)) * batch_size * update_interval, losses, color="b", label="train loss")
    plt.hlines(test_loss, 0, len(losses) * batch_size * update_interval, color='r', label="test loss")
    plt.legend()
    plt.title("training curve")
    plt.xlabel("number of images trained on")
    plt.ylabel("loss")
    plt.show()

    return trained_model, test_loss

# %%
avg_test_loss = testing(trained_model, loss_function, dataset[:128]) # you'll need to change this if you changed batch_size
print(avg_test_loss)

# %%
#TODO Implement a hyperparameter search of your choice 
# I'm not going to give any hand-holdy code cause I believe in you! (and NOT because I'm lazy)

#######################################

def k_fold_split(dataset, k=5):
    fold_size = len(dataset) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i+1) * fold_size
        if i == k-1:
            end = len(dataset)
        folds.append(dataset[start:end])
    return folds





def randomized_grid_search(model, loss_function, update_interval, dataset, k=5, n_samples=10):
    best_loss = float('inf')
    best_params = None
    folds = k_fold_split(dataset, k)
    lr_range = np.arange(0.0001, 1, 0.0001)
    compression_size_range = np.arange(50, 500)
    n_epochs_range = np.arange(10, 25)
    batch_size_range = np.arange(32, 128)

    hyperparameters = [{"lr": random.choice(lr_range), "compression_size": random.choice(compression_size_range), "n_epochs": random.choice(n_epochs_range), "batch_size": random.choice(batch_size_range)} for _ in range(n_samples)]

    for i in range(n_samples):
        print(f"Hyper-parameter {i+1}/{n_samples}")
        lr, compression_size, n_epochs, batch_size = hyperparameters[i]["lr"], hyperparameters[i]["compression_size"], hyperparameters[i]["n_epochs"], hyperparameters[i]["batch_size"]
        print(f"lr: {lr}, compression_size: {compression_size}, n_epochs: {n_epochs}, batch_size: {batch_size}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        avg_loss = 0
        for i in range(k):
            print(f"Fold {i+1}/{k}")
            model.__init__(input_shape, compression_size)
            model.to("cuda")
            train_data = torch.cat([folds[j] for j in range(k) if j != i])
            test_data = folds[i]
            model, test_loss = train_and_test(model, loss_function, optimizer, batch_size, update_interval, n_epochs, train_data, test_data)
            avg_loss += test_loss
        avg_loss /= k
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = hyperparameters[i]
    return best_params





#######################################

# %%
# Use your best hyperparameters -- your final test loss should be under 2000

lr = 0.001               # The size of the step taken when doing gradient descent
batch_size = 256        # The number of images being trained on at once
update_interval = 10   # The number of batches trained on before recording loss
n_epochs = 15           # The number of times we train through the entire dataset
compression_size = 400  # The output size of the encoder



model = Autoencoder(input_shape, compression_size).to("cuda") 

loss_function = nn.MSELoss()                        
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

"""
I have commented out my hyperparameter search because the last time I ran it, it took a long time to run, but I found good hyperparameters 
by a combination of trying and hyperparameter search
Below is the (commented out) line of code where I ran hyperparameter search
"""

# lr, compression_size, n_epochs, batch_size = randomized_grid_search(model, loss_function, update_interval, dataset, k=5, n_samples=5)

"""Above is the (commented out) line of code where I ran hyperparameter search"""

train_dataset = dataset[:3 * len(dataset)//4]
test_dataset = dataset[len(dataset)//4:]

print("Best hyperparameters:", lr, compression_size, n_epochs, batch_size)
print("Now training and testing with best hyperparameters")

best_model, _ = train_and_test(model, loss_function, optimizer, batch_size=batch_size, update_interval=update_interval, n_epochs=n_epochs, train_dataset=train_dataset, test_dataset=test_dataset)

ex_image = dataset[random.randint(0,len(dataset)//batch_size)]
trained_output = trained_model(ex_image)

print("original image:")
plot_image(ex_image)
plt.show()
print("your (BEST) reconstruction")
plot_image(trained_output.detach().byte())

# Try to get a reconstruction that you are happy with
# It is difficult though so try to set up a big search and go for a hike or something
# be warned that google collab sometimes cuts off after some time so be careful!

# %% [markdown]
# ## Autoencoder Written Report (10 points)
# Now, lets take a bit of break from implementing models and do some writing (I know you all love that right?)
# Fill out your answer to each question in the empty markdown cell below each question.
# 
# 1. What would happen if the compression size of your autoencoder was as large as the input image? Try it and tell me what you found out!

# %% [markdown]
# My GPU runs out of memory when the compression size is as large as the size of the input image. 
# 
# However, in theory I would beleive that if my compression size is as large as the size of the input image, the encoder might just learn the exact copy of the input image without extracting the important features from the image. This would eventually defeat our purpose of storing the important features of the inputs in a lower dimension, from which the decoder can extract the necessary features. This will lead to overfitting and the model would perform poorly on test data. As already discussed in class, the autoencoder would learn the identity function if the compression size = input size.   

# %% [markdown]
# 2. Was your model able to output any faces that were looking in different directions? Why do you think it would be hard for an autoencoder to learn to output faces with different orientations? 

# %% [markdown]
# So far my model has not been able to generate faces looking in different directions. 
# 
# Since the autoencoders learn the features of the input image (just in a lower dimension), their output would also contain a slight variation of exactly those features (and not a completely different feature). If the input image has a face facing forward (such that the eyes, nose, lips, etc. appear in a specific orientation), then the autoencoder would just learn the lower dimensional features of this orientation. Because the autoencoder does not have an internal mechanism that can manupulate the orientation of the image, the output image would also have a similar face orientation (such that the eyes, nose, lips, etc. appear in a similar orientation as the input) instead of having a face looking in a completely different direction. 

# %% [markdown]
# ## Generative Adversarial Network (60 points)
# 
# Now it's time for us to make a Generative Adversarial Network (GAN)! 
# 
# GANs contain a generator that generates an image based on a given dataset, and a discriminator (classifier) to distinguish whether an image is real or generated.
# 
# GANs are very similar to autoencoders in the sense that we will create two different models, but we will actually train them on different losses!

# %% [markdown]
# ### The GAN model (20 points)
# 
# In this part you will create your model for both the Discriminator and Generator. The discriminator will take in an input the size of the images and output a bit which represents either real or fake (the discriminators geuss as to the input is real data or generated)
# 
# The generator will take in an input of some size (it will end up being noise but this wont come up in the model making part) and then output an "image" that is the same size as the real data. 

# %%
class Discriminator(nn.Module):

    '''
    This will be your discriminator half of you GAN
    it will take in something of the shape of an image of a face

    it will then return either 0 or 1 depending on whether it
    believes the input is from the real distribution or not
    '''

    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.to("cuda")

        # TODO Initialize your discriminator
        # You can linear and conv layers -- as well as anything else you find (don't forget you nonlinearity) 
        # HINT: if it trains too slow try reducing the dimensions for your linera layers somehow
        ##################################### 
        self.layers = nn.Sequential(
                                    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # 16 channels, 109x89
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 channels, 55x45
                                    nn.ReLU(), 
                                    nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1), # 64 channels, 19x15
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(19*15*64, 1) # Linear layer to compress the data to the compression size
        )





        ##################################### 

    def forward(self, x):

        # TODO fill out the forward pass of your model
        # Don't forget nonlinearities!
        ####################################

        x = x.to("cuda")
        x = self.layers(x)



        ####################################

        x = nn.Sigmoid()(x) # This sigmoid will squish the outputs between 0 and 1 (you can change this if you'd like but some things may break)

        return x.to("cpu")


# %%
# This is the performance of the Discriminator (before training) on the example image (which should be 1)
# If you rerun this it should change since we are randomly initializing the model
discriminator = Discriminator((3,109,89)).to("cuda")
ex_output = discriminator(ex_image.float())

plot_image(ex_image)
print("Output of the discriminator given this input:", ex_output[0].detach().numpy()[0])

# %%
# class Generator(nn.Module):
#     def __init__(self, input_size, output_shape):
#         super(Generator, self).__init__()
#         self.to("cuda")

#         # TODO
#        ####################################
        

#         self.layers = nn.Sequential(
#                                     nn.Linear(input_size, 28*23*64),
#                                     nn.ReLU(),
#                                     nn.Unflatten(1, (64, 28, 23)), # 64 channels, 19x15
#                                     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), # 32 channels, 55x45
#                                     nn.ReLU(),
#                                     nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1), # 16 channels, 109x89
#                                     nn.ReLU(),
#                                     nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1), # 3 channels, 109x89
#                                     nn.Sigmoid()
#                                     )

#         ####################################

#     def forward(self, x):
#         # TODO
#        ####################################
        
#         x = x.to("cuda")
#         out = self.layers(x)

#         ####################################
#         return out.to("cpu")

# %%
class Generator(nn.Module):
    def __init__(self, input_size, output_shape):
        super(Generator, self).__init__()
        self.to("cuda")

        # TODO
       ####################################
        

        self.layers = nn.Sequential(
                                    nn.Linear(input_size, 100),
                                    nn.ReLU(),
                                    nn.Linear(100, 250), 
                                    nn.ReLU(), 
                                    nn.Linear(250, 3*109*89),
                                    nn.Unflatten(1, (3, 109, 89)), # 64 channels, 19x15
                                    # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), # 32 channels, 55x45
                                    # nn.ReLU(),
                                    # nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1), # 16 channels, 109x89
                                    # nn.ReLU(),
                                    # nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1), # 3 channels, 109x89
                                    nn.Sigmoid()
                                    )

        ####################################

    def forward(self, x):
        # TODO
       ####################################
        
        x = x.to("cuda")
        out = self.layers(x)

        ####################################
        return out.to("cpu")

# %%
# This will show the output of our generator before training (it's fine if its all black)
test_gen = Generator(128, (3, 109, 89)).to("cuda")
noise = (torch.rand(1, 128) - 0.5) / 0.5
print(noise.shape)
test_output = test_gen(noise)

plot_image(test_output.detach().byte())

# %% [markdown]
# ### This is our generator's attempt at making something before training ^
# 
# Let's train it to see how it can improve! 
# 
# ### Training Loop (20 points)
# The training for a GAN is fundementally the same for all the other models we train with pytorch (zero grad, output, loss, loss backward, optimizer step). But we are going to do 2 seperate updates in each loop, with different losses!
# 
# For each loop you will calculate the loss for both the discriminator and generator and then update those models accordingly. Some code is provided to help you out, but not all of it! 
# 
# You should record your *losses* for both the generator and the discriminator **(two lists in total)**. You'll need these values for the written section, where you will be discussing them.
# 
# Then, use your wisdom from the autoencoder hyperparameter search to find good settings to all the hyperparamers and try your best to get your model to produce a face!
# 
# [Hint for reseting the optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad)
# 
# [Hint for stepping with the optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step) (You'll have to use .backward() to get the gradient)

# %%
def training(generator, discriminator, loss, g_optimizer, d_optimizer, train_dataloader, n_epochs, update_interval, noise_samples):
    
    g_losses = []
    d_losses = []
    
    for epoch in range(n_epochs):
        for i, image in enumerate(tqdm(train_dataloader)):

            # Training the discriminator
            # Real inputs are actual images from the CelebA dataset
            # Fake inputs are from the generator
            # Real inputs should be classified as 1 and fake as 0
            
            image = image.float()/255
            
            real_classifications = discriminator(image)
            real_labels = torch.ones(image.shape[0])

            noise = (torch.rand(image.shape[0], noise_samples) - 0.5) / 0.5
            fake_inputs = generator(noise)
            fake_classifications = discriminator(fake_inputs)
            fake_labels = torch.zeros(image.shape[0])

            classifications = torch.cat((real_classifications, fake_classifications), 0).reshape(len(real_classifications) + len(fake_classifications))
            targets = torch.cat((real_labels, fake_labels), 0)

            # TODO Calculate the loss for the discriminator and apply the gradient
            # This is the same as a normal training loop!
            #######################################################

            d_loss = loss(classifications, targets)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()





            #######################################################
            
            if i % update_interval == 0:
                d_losses.append(round(d_loss.item(), 2))
            

            # We do a seperate forward pass to update the gradient for the generator since 
            # Pytorch doesnt like us reusing the same computation graph (it makes one under the hood)
            noise = (torch.rand(image.shape[0], noise_samples) - 0.5) / 0.5
            fake_inputs = generator(noise)
            fake_classifications = discriminator(fake_inputs)
            fake_labels = torch.zeros(image.shape[0], 1)

            # TODO Calculate the loss for the generator and apply the gradient
            # HINT: the loss for the generator is essentially the opposite of the 
            # discriminators loss but doesnt care about the real examples (they dont go through the generator at all)
            #######################################################
            g_loss = loss(1-fake_classifications, fake_labels)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            




            ########################################################

            if i % update_interval == 0:
                g_losses.append(round(g_loss.item(), 2))
                
    return (generator, discriminator), (g_losses, d_losses) 

# %%
lr = 2e-4               # The size of the step taken when doing gradient descent
batch_size = 64         # The number of images being trained on at once
update_interval = 10   # The number of batches trained on before recording loss
n_epochs = 10            # The number of times we train through the entire dataset
noise_samples = 1024    # The size of the noise input to the Generator

loss_function = nn.BCELoss()

G_model = Generator(noise_samples, (3,109,89)).to("cuda")
D_model = Discriminator((3,109,89)).to("cuda")
G_optimizer = torch.optim.Adam(G_model.parameters(), lr=lr)     # This is an improved version of SGD which decreases the learning rate over time to avoid leaving a minima
D_optimizer = torch.optim.Adam(D_model.parameters(), lr=lr*4)       # This is an improved version of SGD which decreases the learning rate over time to avoid leaving a minima

train_dataset = dataset

models, losses = training(G_model, D_model, loss_function, G_optimizer, D_optimizer, train_dataset, n_epochs, update_interval, noise_samples)

G_model, D_model = models
g_losses, d_losses = losses

plt.plot(np.arange(len(g_losses)) * batch_size * update_interval, g_losses)
plt.title("training curve for generator")
plt.xlabel("number of images trained on")
plt.ylabel("loss")
plt.show()

plt.plot(np.arange(len(d_losses)) * batch_size * update_interval, d_losses)
plt.title("training curve for discriminator")
plt.xlabel("number of images trained on")
plt.ylabel("loss")
plt.show()

# %% [markdown]
# Now let's take a look at the generated images coming out of our trained GAN!

# %%
# This will show the same example as before with the discriminator's new score 
# 0 is fake and 1 is real -- is it good at discriminating?

trained_output = D_model(ex_image.float())

plot_image(ex_image)
print("Output of the discriminator given this input:", trained_output[0].detach().numpy()[0])
plt.show()

noise = (torch.rand(1, noise_samples) - 0.5) / 0.5
trained_gen = G_model(noise)

plot_image(trained_gen.detach())

trained_output = D_model(trained_gen)

print("Output of the discriminator given this generated input:", trained_output[0].detach().numpy()[0])

# %% [markdown]
# ### GAN hyperparameter Search (10 points)
# 
# GANs are notoriously hard to train -- generally you have to do a lot of hyper parameter searching to 
# find good settings. Try it out until you get results above that you're happy with it!
# You're going to have to write a good amount of code for this, but you can base it off of what is above if you want

# %%
# TODO Do a hyper parameter search and find the best settings for your model 
# Again I leave this up to you as to what to do, but I'm gonna warn you that trying them by hand 
# is probably going to be too slow to work
#######################################################

def randomized_grid_search_gan(generator, discriminator, loss, g_optimizer, d_optimizer, train_dataloader, k=5, n_samples=10):
    best_loss = float('inf')
    best_params = None
    folds = k_fold_split(train_dataloader, k)
    lr_range = np.arange(0.0001, 1, 0.0001)
    n_epochs_range = np.arange(10, 11)
    noise_samples_range = np.arange(100, 1000)
    batch_size_range = np.arange(32, 128)

    hyperparameters = [{"lr": random.choice(lr_range), "n_epochs": random.choice(n_epochs_range), "noise_samples": random.choice(noise_samples_range), "batch_size": random.choice(batch_size_range)} for _ in range(n_samples)]

    for i in range(n_samples):
        print(f"Hyper-parameter {i+1}/{n_samples}")
        lr, n_epochs, noise_samples, batch_size = hyperparameters[i]["lr"], hyperparameters[i]["n_epochs"], hyperparameters[i]["noise_samples"], hyperparameters[i]["batch_size"]
        print(f"lr: {lr}, n_epochs: {n_epochs}, noise_samples: {noise_samples}, batch_size: {batch_size}")
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr*4)
        avg_loss = 0
        for i in range(k):
            print(f"Fold {i+1}/{k}")
            generator.__init__(noise_samples, (3, 109, 89))
            discriminator.__init__((3, 109, 89))
            generator.to("cuda")
            discriminator.to("cuda")
            train_data = torch.cat([folds[j] for j in range(k) if j != i])
            test_data = folds[i]
            models, losses = training(generator, discriminator, loss, g_optimizer, d_optimizer, train_data, n_epochs, update_interval, noise_samples)
            generator, discriminator = models
            test_loss = testing(discriminator, loss, test_data)
            avg_loss += test_loss
        avg_loss /= k
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = hyperparameters[i]
    return best_params


#######################################################

# %%
# Finding the best hyperparameters for the GAN

"""
I have just ran this code with very few sets hyperparameters because it takes a long time to run, but it works when it completes properly
"""

lr, n_epochs, noise_samples, batch_size = randomized_grid_search_gan(G_model, D_model, loss_function, G_optimizer, D_optimizer, train_dataset, k=2, n_samples=1)
loss_function = nn.BCELoss()

G_model = Generator(noise_samples, (3,109,89)).to("cuda")
D_model = Discriminator((3,109,89)).to("cuda")
G_optimizer = torch.optim.Adam(G_model.parameters(), lr=lr)     # This is an improved version of SGD which decreases the learning rate over time to avoid leaving a minima
D_optimizer = torch.optim.Adam(D_model.parameters(), lr=lr*4)       # This is an improved version of SGD which decreases the learning rate over time to avoid leaving a minima

train_dataset = dataset

models, losses = training(G_model, D_model, loss_function, G_optimizer, D_optimizer, train_dataset, n_epochs, update_interval, noise_samples)

G_model, D_model = models
g_losses, d_losses = losses

plt.plot(np.arange(len(g_losses)) * batch_size * update_interval, g_losses)
plt.title("training curve for generator")
plt.xlabel("number of images trained on")
plt.ylabel("loss")
plt.show()

plt.plot(np.arange(len(d_losses)) * batch_size * update_interval, d_losses)
plt.title("training curve for discriminator")
plt.xlabel("number of images trained on")
plt.ylabel("loss")
plt.show()


# %%
# This will show the output of our *best* generator after training
noise = (torch.rand(1, 1024) - 0.5) / 0.5
trained_output = G_model(noise)

plot_image(trained_output.detach())

# %% [markdown]
# ## GAN Written Report (10 points)
# More writing, yay! Hopefully these questions will make you think!
# 
# 1. Does your trained discriminator learn to correctly discriminate generated examples -- how does yours perform above? What would you guess is the ideal discriminator performance of a trained GAN? Why? 

# %% [markdown]
# The output of my discriminator highly varies. I tried running it for a couple of images from the dataset, and I got varied ouputs most of the time. (for example I got outputs like - 0.98, 0.94, 0.11, 0.32, 0.97, 0.02, etc.). I beleive that an ideal discriminator's performance should be an accuracy of about (50%), as this level of performance will show that the discriminator is unable to correctly classify between fake and real images almost half of the time. This would mean that the generator has become quite good and generating real like images (which is why the discriminator is only able to catch the generator 50% of the time). This performance of the discriminator can also be seen like the discriminator is acting like a coin flip while detecting fake images (as it correct only 50% of the times). This kind of performance from the discriminator (50% accuracy) will show that the generator has generalized well in learning the features of the faces. 
# 

# %% [markdown]
# 2. Sometimes our generator can produce images that dont look at all like faces (to us) but still fools our discriminator. We call these exmaples *adverserial examples* for our disriminator. Why might our generator produce images like this instead of faces? 

# %% [markdown]
# In some cases the generator might produce pixels that somehow lead to a value that the discriminator considers as real. This might happen when the features generated by the generator are a subset of features of a real image, which may consequentially fool the discriminator. Moreover, we for that several different operations could also lead to the same value (for example, two different matrix multiplications can produce the same value). Therefore, we can also infer from this fact that - some alignment of pixels in an image (that is nowhere even close to a face's image) might lead to the same discriminator value for an image that is real image of a face, hence fooling the discriminator. 
# Moreover, the generator's goal is produce images that the discriminator classifies as real, even if they may not look real at all. Therefore, the generator might learn to produce specific features that are able to fool the discriminator, but don't look real at all. 


