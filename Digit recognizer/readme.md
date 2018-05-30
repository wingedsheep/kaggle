# Kaggle Digit Recognizer

Solving the Kaggle Digit Recognizer competition. And setting up a good working environment for doing machine learning experiments.

https://www.kaggle.com/c/digit-recognizer

## Getting Started

* Install homebrew, follow instructions on https://brew.sh/
* Install python and pip using homebrew
* Install keras and tensorflow
** https://www.tensorflow.org/install/install_mac
** https://keras.io/#installation
* Download the dataset from https://www.kaggle.com/c/digit-recognizer and put it in /data

### Running on Floydhub

https://medium.com/@margaretmz/get-started-with-floydhub-82cfe6735795
https://github.com/floydhub/keras-examples

* Create a Floydhub account.
* Login on https://www.floydhub.com and create a project on Floydhub.
* Create a dataset on Floydhub.
* Install Floydhub client
* Go to code folder and add the code to the Floydhub project
* Go to the data folder and add the data to the Floydhub dataset

```
pip install -U floyd-cli
cd [CODE_FOLDER]
floyd init [USERNAME]/[PROJECT_NAME]
cd [DATA_FOLDER]
floyd data init [USERNAME]/[DATASET_NAME]
floyd data upload 
```

Finally to run the project, use 
```
floyd run <insert-command-here> 
# To use GPU 
floyd run --gpu <insert-command-here> 
# To use a different environment 
floyd run --gpu --env pytorch <insert-command-here> 
```