# **leaf-Classification**


# **Table Of Content**
1. [Introduction](#my_first_title)
2. [Method](#my-second-title)
3. [Result](#my-third-title)
4. [Install Requirements](#my-fourth-title)


## **Introduction**
This Project is a classification task, in which I train an artificial neural network model to classify images of leaves into one of the 14 categories provided.
Each model was then evaluated based on its mean accuracy metric on an unknown test set.
The strategy followed are subdivided into three separate phases:
*  Definition of a basic convolutional neural network, looking how it behaves with changes in the hyperparameters
*  Using the knowledge gained, implementation of the transfer learning + fine-tuning techniques in
a new model
*  Searching in the literature for further improvements to better tune our model

## **Method**


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


### **First Attempt: A simple CNN**


The model chosen for the first attempt was the same as one used during an exercise lab.
In particular, as with the configuration, it was made up of 5 blocks of convolution + ReLU + batch
normalization + max pooling layers, plus a flatten layer and two fully connected layers on top of them (the
last one being the classifier one) with dropout equal to 0.3. As a callback Early Stopping has been used to
avoid overfitting, with a patience of 10 epochs.
With this simple model different hyperparameters has been tested:
*  The dropout amount
*  The learning rate value
*  The batch size value
*  The number of units in the first fully connected layer
To make the analysis I followed both a train and error approach, as well as using the KerasTuner
library.
It has been concluded that the most relevant hyperparameters that affects the validation accuracy are the
batch size and the learning rate: 1

*  Batch sizes of 8, 16, 32, 64, 128 were tested. The best result was obtained with a batch of size 64,
enough large to be representative of the train dataset and at the same time less RAM demanding
than a 128 sized one.
*  A high learning rate does not allow the optimizer to reach the minima of the loss function, but
constraints it to oscillate around it. At the same time a low learning rate may not be enough for
the network to learn, for example stopping the optimizer iterations on a local minima. A trade-off
approach has been followed for the Transfer Learning + Fine Tuning notebook, which will be
discussed in the following section.








###  **Second Attempt: Transfer Learning + Fine Tuning**


The need for a great validation accuracy score combined with a good efficiency measured in terms of
number of parameters and FLOPS operations led me to choose the EfficientNet network, as it can be
observed in the image below.



![Untitled3](https://user-images.githubusercontent.com/75788150/178113998-1584c9dd-e062-40a7-b68d-893e1f7b269c.png)




In particular I adopted the EfficientNet-B4 model, as it already reaches great performance with a
computational effort comparable with Inception-V2, DenseNet-201 and Xception models. I could
choose even a better accuracy model like B5, B6, B7 or B8 ones, but I preferred to keep working
following the efficiency philosophy already described above.
In order to boost even more the accuracy of the model without increasing the computational effort, as
suggested by the Keras documentation I initialized the weights using the latest



*   #### **Transfer Learning**


The EfficientNet network has been imported in the model right after the augmentation layers, without
its top dense ones. The layers have been set as non-trainable, as I wanted to preserve the feature
extraction capabilities of the pre-trained model. The dense layers have the following configuration:
*   GlobalAveragePooling2D: it allows to reduce overfitting with respect to a simple flatten layer,
acting as a regularizer
*   BatchNormalization: it is a regularizer too, normalizing the output and centering it
*   DenseLayer: added to increase the depth of the network
*   Dropout: it reduces the overfitting as well randomly changing the number of units of the
previous layer
*   OutputLayer: the classifier itself.
The learning rate value chosen is 1e-2. It is a relatively high one, but as written in the Keras
documentation it is a valid option for this phase (warming up phase). Learning rate scheduler has been
tried as well (TriangularCyclicalLearningRate), resulting in better results for the Transfer
Learning part but worse overall.



*   #### **Fine Tunning**



The EfficientNet-B4 network is made up of hundreds of layers subdivided into 7 blocks. During fine
tuning a number of layers get unfrozen and then the model is fitted using a smaller learning rate (1e-4) to
avoid losing the beneficial results obtained during the Transfer Learning phase.
Note:
● I unfroze only the last block of layers (31), since it is recommended to freeze/unfreeze one or
more full blocks. In a second notebook also the 6th block as been unfrozen, leading to worse
results so I chose to continue with only one unfrozen block for the future attempts
● As suggested by keras, the BatchNormalization layers has been kept frozen




## **Result**

The results achieved in the first test set were definitely promising, showing the power of the
improvements adopted. You can see the results in figure below.



![Capture](https://user-images.githubusercontent.com/75788150/178114260-dd1ddf18-a21e-4ea8-aba8-596b0b0afaba.PNG)






In the next section(s), we will enter into the details of these three steps. Note that all the notebooks are
run on kaggle.com using the GPU accelerator. The amount of free GPU time per week was limited, so we
searched for a way to improve the performance of the code also from a computational point of view.
