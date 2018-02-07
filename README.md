# ***Predict Illumination from Image Pair (captured from smart phone)***#

### [Describe] ###
	Name
		fandb : predcit illumination from [F]ront AND [B]ack image
	Date
		2017-02-05 15:59
	Enviroment
		CPU : i7-6800k
		GPU : nvidia gtx 1080
		Sys : Ubuntu 16.04 LTS
		Python : Python3.5.2
		tensorflow: 1.4.1
	CNN Model
    	Trained: Yes
      	Input: two image, front and back 256x192x3
     	Output: 48 sh coefficient (3/4 orders) 16x3 coefficients

### [TrainingDetail] ###
	TrainingSteps : 80k arround
	Training loss : 0.07
	Testing  loss : 0.22 around



|--------------------------------------------------Usage--------------------------------------------------|
v                                                                                                         v
### [Dependencies] ###
	For the Tr
	InputData
    	DataPath: 
	Record
    	RecordPath:

### [Config] ###
	config.py
	    data_path = DataPath_Root
		best_path = RecordPath_best
	useit.py
		Need specify the input image

### [Usage] ###
	Lookup Training details
		Tensorboard --logdir=RecordPath_train(test/best)
		Visualize training/testint/best_loss.txt by R
	Use this model to predict illumination
		Understand useit.py and use it (lol......)
