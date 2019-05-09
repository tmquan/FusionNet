# FusionNet

Please install tensorflow, keras, skimage, sklearn, opencv and related libraries using (sudo) pip install if you have the errors related with essential libraries.

Run mkdir models to save the checkpoint of model's weights

- Data.py read images and membrane labels from data folder
- Train.py train the model (define in Model.py) with data augmentation (Augment.py)
- Deploy_full.py predict the result (currently I predicted on the training set, you may set up 
- Utility.py includes all the needed libraries (sorry for my lazy mode). 
