# Devanagri-Hindi_Digits_Recognition

My Model is Devanagri(Hindi) Handwritten Digits Recognition. So I took dataset of handwritten digits images of size (32, 32)
containing 20000 images in a dataset. Dataset is divided into training and validation set containing 17000 in training and 3000 in validation set. So I have resize image into (130, 130) pixels for better prediction while using app.

I have generated Images using Image Data generator and rescale pixels value from [0, 255] to [0, 1]. Batch Size of 512 and color mode of image is grayscale and class used is categorical as there are 10 category ranging from 0 to 9.

Model Used

I have use CNN Model for recognition as it is good for dealing with images. So I have used sequential model with two convolution layers of 32 and 64 filter size having kernel size of (3, 3) and activation used is relu(rectified non-linear unit). Then  it is passed through MaxPooling layer for halving the size of layer, and it is flatten into 1 long vector using Flatten layer. So finally output layer containing 10 output categories with activation as softmax(used for more than 2 outputs).

Model is compiled using 'Adam' optimizer having 'categorical_crossentropy' loss and then model is trained on 30 epochs with steps_per_epoch = Total training images/batch size of train and validation steps = Total validation images/batch size of validation. Model is evaluated with 100% accuracy and model is saved in the form of json. So is plotted and is showing good results and then model used to predict test images and it shows perfect output.

OpenCV App

So many ML Developer stops at this point but they doesn't show how we can deploy it for predicting mouse-drawn digit.
I have created digits_app for people to draw on canvas and model will predict the digit. I have used OpenCV for prediction of digits using app.

Accuracy is 100% but sometimes it may not predict right as images of datset and drawn images are not od same size so there may be some wrong prediction.
So you can use canvas as:
pressing 's' - To start drawing
pressing 'p' - To predict digit
pressing 'c' - To clear screen
pressing 'q' - To exit out of app

Dataset used:

https://drive.google.com/drive/folders/1bGjf5IIJ793t8PSbeHBqnO8yRbNZQrlz?usp=sharing for digits dataset.

https://archive.ics.uci.edu/ml/machine-learning-databases/00389/ for dataset containing digits with vowels and consonants.  

So you can use this and modified as you want and make your lives better in the world.

