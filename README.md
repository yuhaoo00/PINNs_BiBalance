# Parallel Physics-Informed Neural Networks with Bidirectional Balance

Switch to the current folder and enter the following command in the terminal to run.

## Using help
`python main.py -h`

## Test case
`python main.py --inference --netpath checkpoints\PINNcase3_60s_20000e\best_model.pth.tar`
("--Inference" only inference does not do training; "--netpath" gives the path of the trained model file.)

`python main.py --inference --netpath checkpoints\PINN_10s_10000e\best_model.pth.tar --expt 10`
("--expt" sets the maximum prediction time, the default is 60s)

`python main.py --inference --netpath checkpoints\PINNnoFBMcase1_60s_20000e\best_model.pth.tar --no_FBM`
("--no_FBM" declares that this model does not contain FBM)

`python main.py --inference --netpath checkpoints\singlePINNcase3_60s_20000e\best_model.pth.tar --no_PSF --eps`
("--no_PSF" declares that this model does not contain PSF; "--eps" save the result as eps, default png)

...

## Train case
`python main.py --epochs_Adam 20000 --expt 60 --seed 211 --tag case1`
("--Epochs_Adam" sets the number of training epochs; "--expt" sets the maximum time value; "--seed" sets the global random seed; "--tag" marks the folder name)
("PINNcase1_60s_20000e" folder will be generated, if it already exists, it will be overwritten and updated)

`python main.py --epochs_Adam 20000 --expt 60 --seed 211 --tag noBBMcase1 --no_BBM`
("--no_BBM" declares that this model does not contain BBM)

`python main.py --epochs_Adam 20000 --expt 60 --seed 211 --tag case1 --debug`
("--debug "turn on the debug mode, record the gradient to the "grad" folder)

...