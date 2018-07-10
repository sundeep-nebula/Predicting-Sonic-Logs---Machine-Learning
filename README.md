# Predicting Sonic Logs using Machine Learning
This is a tutorial on how to predict sonic logs (DT) using gamma ray (GR) and neutron porosity (NPHI) logs. We will use a simple machine learning method, linear regression with multiple variable, to predict DT. This tutorial is influenced by the Machine Learning course I am currently taking on Coursera, taught by Andrew Ng from Stanford. This idea of prediction using multivariable linear regression is not just limited to DT prediction, but can be use in many other instances where the varialbes are linearly dependent to each other. 

[Predict_DT_notebook](../master/Predict_DT_notebook.ipynb) - Is an ipython notebook that goes through the workflow using the dataset provided in the repository (dt_prediction_dataset.csv). All the required codes (gradient descent, feature normalize etc.) are defined within the ipython notebook. I have also placed the codes as separate python scripts in the repository for everyone to download and use. 

The dataset I provide is in csv format, if you want to go through this workflow on your workstation with LAS files, there is a nice library [LASIO](https://github.com/kinverarity1/lasio) you can use within python to import LAS files. 


Acknowledgements:
I would like ot thank SEG for their impressive list of open-source datasets. 
