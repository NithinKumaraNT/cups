Hello potential ML6 colleague!

If you are reading this, you are probably applying for an ML engineering job at ML6. This test will evaluate if you have the right skills for this job. The test should approximately take 2 hours.

In the test, you will try to classify the mugs we drink from at ML6. If you are able to complete this test in a decent way, you might soon be drinking coffee from the black ML6 mug (which is also in the data) together with us.

## The data

As you can see, all data can be found in the data folder. For your purposes, the data has already been split in training data and test data. They are respectively in the train folder and test folder. In those folders, you can find four folders which represent the mugs you'll need to classify. There are four kind of mugs: the white mug, the black mug (the ML6 mug), the blue mug and the transparent mug (glass). This means that the CNN you will build needs 4 output neurons. The input of your CNN will be 64x64 RGB images of mugs. If you want, you can inspect the data, however, the code to load the data of the images into numpy arrays is already written for you.

## The model

In the trainer folder, you will be able to see several python files. The data.py, task.py and final_task.py files are already coded for you. The only file that needs additional code is the model.py file. The comments in this file will indicate which code has to be written.

To test how your model is doing you can execute the following command (you will need to [install](https://cloud.google.com/sdk/docs/) the gcloud command):

```
gcloud ml-engine local train --module-name trainer.task --package-path trainer/ -- --eval-steps 5
```

If you run this command before you wrote any code in the model.py file, you will notice that it returns errors. Your goal is to write code that does not return errors and achieves a test accuracy (the percentage of mugs in the test set that were guessed right by the CNN) that is as high as possible.

The command above will perform 5 evaluation steps during the training. If you want to change this you only have to change the 5 at the end of the command to the number of evaluation steps you like. The batch size and the number of training steps should be defined in the model.py file.

![Data overview](data.png =1x)

The command above uses the task.py file. As you can see in the figure above, this file only uses the mug images in the training folder of this repository and uses the test folder to evaluate the model. This is excellent to test how the model performs but to obtain a better evaluation one can also train upon all available data which should increase the performance on the dataset you will be evaluated. So after you finished coding up model.py you can read on and you'll notice how to train your model on the full dataset.

## Deploying the model

Once you've got the code working you will need to deploy the model to Google Cloud to turn it into an API that can receive new images of mugs and returns its prediction for this mug. Don't worry, the code for this is already written in the final_task.py file. To deploy the model you've just written you only have to run a few commands in your command line.

To export your trained model and to train your model on the training folder and the test folder you have to execute the following command (only do this once you've completed coding the model.py file):

```
gcloud ml-engine local train --module-name trainer.final_task --package-path trainer/
```

Once you've executed this command, you will notice that the output folder was created in the root directory of this repository. This folder contains your saved model that you'll be able to deploy on Google Cloud ML-engine.

To be able to deploy the model on a Google Cloud ML-engine you will need to create a [Google Cloud account](https://cloud.google.com/). You will need a credit card for this but you'll get free credit from Google to run your ML-engine instance.

Once you've created your Google Cloud account, you'll need to deploy your model on a project you've created. You can follow a [Google guideline](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction) for this.

## Finalizing your test

After the model is deployed to a Google Cloud ML-engine, you only need to add us to your project:

* Go to the menu of your project
* Click IAM & admin
* Click Add
* Add stan.callewaert@ml6.eu with the Project Owner role

Once you've added us to your project you've finished the test. ML6 will review it and let you now something as soon as possible.

If you are invited for an interview at ML6 afterwards, you'll have to make sure that you bring your laptop with the code that you've wrote on it so you can explain your model.py file to us.
