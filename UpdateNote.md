commit 0.1 

fix bugs of batch normalization

0. former version can not be used because this bug is very deadly.


1. fix bugs of batch_normalization, at former version, 

2. record loss value of every logged step to tloss.txt an eloss.txt

3. adding a application (useit.py), which is used to inference SH coefficients from image.

4. reconstruct model.py -- add the LinearNetwork class to make it more easier to build an linear branch of a network
