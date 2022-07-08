# leaf-Classification


# Table Of Content
1. [Introduction](#my_first_title)
2. [Method](#my-second-title)
3. [Result](#my-third-title)
4. [Install Requirements](#my-fourth-title)


## Introduction
This Project is a classification task, in which I train an artificial neural network model to classify images of leaves into one of the 14 categories provided.
Each model was then evaluated based on its mean accuracy metric on an unknown test set.
The strategy followed are subdivided into three separate phases:
*  Definition of a basic convolutional neural network, looking how it behaves with changes in the hyperparameters
*  Using the knowledge gained, implementation of the transfer learning + fine-tuning techniques in
a new model
*  Searching in the literature for further improvements to better tune our model

## Method 


As stated by the TensorFlow official documentation, “Achieving peak performance requires an efficient
input pipeline that delivers data for the next step before the current step has finished”.
For this reason, I firstly decided to define two objects for the training and validation sets as
tf.data.Dataset objects, which are recommended by TensorFlow for their efficiency, using the
image_dataset_from_directory function.
Note: in all attempts, I split the data into two sets: training and validation, with ratio 5:1.
Then I applied on them the prefetch transformation, which allows the new images to be loaded in the
background while the GPU is still performing computations on the previous batch of images.
Finally, I did data augmentation as it has been demonstrated from literature that this practice can
dramatically improve the ability of the network to generalize on “unseen” test images. Giving particular
attention to the efficiency of the process, I decided to do data augmentation directly as part of our
model, making use of the Keras image augmentation layers. In fact, with this configuration, the data
augmentation is performed synchronously with the rest of the model execution, meaning that it will
benefit from GPU acceleration.
After having implemented all the above efficiency improvements on the simple CNN architecture seen in
class, it has been possible to notice that the new updated version was 7 times faster on a single epoch with
respect to the default model.



In the next section(s), we will enter into the details of these three steps. Note that all the notebooks are
run on kaggle.com using the GPU accelerator. The amount of free GPU time per week was limited, so we
searched for a way to improve the performance of the code also from a computational point of view.
