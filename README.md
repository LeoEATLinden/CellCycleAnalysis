# CellCycleAnalysis
A Deep Learning method to annotate time lapse microscopy data of cells expressing mRuby-PCNA with cell cycle stages.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.


convertSegment.py
	This file converts and segments .tiff files from the microscope
	into several .npy files for each cell trace. It needs to be given
	two additional files. The tracking file containign data on the split events
	(K3_Tracks_...... in this case) And the one with the coordinates (obj_track.txt). More details in comments in the file

trainCNN.py
	This file contains the functions needed to train a neural network.
	Given a folder with segmented images (the folder Images) and a folder
	with labels (the folder Labels). This will train the model on this data.
	It is preconfigured to train for 2 epochs on the available data if the file
	is run from the command line. More info in file. 

test.py
	This file contains functions to test the performance of the methods.
	If run from the command line it will evaluate the performance of the 
	method on all the cross valiidation folds from the project. Data on
	the models can be found in the NNTESTDATA. More info in file.

methodOne.py 
	THis implements the HMM on mean and variance approach. More details in file
	IF ran these will ask for the paths to some folders containign images to be classified and some additional data.

methodTwo.py
	This implements the CNN+HMM methods. More details in file.
	IF ran these will ask for the paths to some folders containign images to be classified and some additional data.

annotate.py
	This file provides methods to annotate the data in the form required for the models to train. More details in files.

THE REST OF THE FILES contain helper methods. If unclear or errors please contact me at either leo.linden@hotmail.se or leo.linden18@imperial.ac.uk

