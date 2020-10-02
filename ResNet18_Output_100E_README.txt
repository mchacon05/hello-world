Purpose:
	
To visualize how the alpha and beta values, for TAct and mTAct, change across several epochs.

Date Created:

08.12.20 - 08.14.20

Created by:

Manuel Chacon while conducting research under Dr. Mario Banuelos.

Research Topic:

Activation functions

Contents:

	1) Name of parameter (alpha or beta)
	2) Value of parameter as a floating point value
	3) Epoch [0,100]
	4) Run [1,5]; each run contains 100 epochs, used to identify the five distinct runs

Alpha:

There are 9 total alpha values per epoch,

	1) f_activation_alpha
	2) layer1.0.f_activation_alpha
	3) layer1.1.f_activation_alpha
	4) layer2.0.f_actiavtion_alpha
	5) layer2.1.f_actiavtion_alpha
	...
	9) layer4.1.f_activation_alpha

Beta:

There are 9 total beta values per epoch. These follow the same pattern as the alpha values,

	1) f_activation_beta
	2) layer1.0.f_activation_beta
	3) ...
	9) layer4.1.f_activation_beta

Notes:

After creation, each CSV file is cleaned and given a header. Cleaning consists of deleting some printout that I wasn't able to prevent from printing out (LR Finder or LR Recorder from fastai) and deleting [^M^M] that prints out at the beginning of each epoch (I don't know why this prints out). The header consists of adding Name, Value, Epoch, and Run at the top of the document.
