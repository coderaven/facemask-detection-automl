Image Classification using TFlite Model from Firebase AutoML
============================================================

# Step 1:
Install Python via Anaconda (if possible, so that most needed packages are already pre-installed)

https://www.anaconda.com/products/individual

# Step 2:
Follow the Tutorial on my Youtube Video for easier step by step annotations.

But Basically:
- Gather Data (Images)
- Put the data into their respective folder with the folder name as the label name like this:
```
    my_training_data.zip
    |____accordion
    | |____001.jpg
    | |____002.jpg
    | |____003.jpg
    |____bass_guitar
    | |____hofner.gif
    | |____p-bass.png
    |____clavier
        |____well-tempered.jpg
        |____well-tempered (1).jpg
        |____well-tempered (2).jpg
```

- Zip the data set
- Upload to Firebase AUTOML
- Train
- Download the resulting Model if happy with the results
- Then use the Model using the sample code given in this repository
