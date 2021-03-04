# autoencoder
Autoencoder for EAS radio array data denoising

Neural TRex v0.9.5

### 1. converter.py
*Complete*

Used for convert from ADST to numpy binary format.
arguments
--signal: signal file
--noise: noised signal file
output: 2 numpy files


### 2. creator.py
*In progress*

Used for creation of training or test sets or change vectors dimension for neural network.
Use "upsampling" for create dataset with augmentation.
arguments
--signal: signals
--noise: noised signals
--center: approximate peak position
--window: size of output numpy array
output: 2 numpy files


### 3. trainer.py
*In progress*

Used for create and train CCN via uDocker container
arguments
--signal: signals
--noise: noised signals
--min: min amplitude
--max: max amplitude
--epochs: count of epochs
--arch: structure of cnn in .json file
output: model in h5 file


### 4. denoiser.py
*In progress*

Used for denoising input file.
arguments
--noise: noised signals
--result: output file
--model: model for denoising
output: 1 numpy file

### 5. estimator.py
*In progress*

Creates .csv table which summarizing result.
arguments
--true: true signals
--reco: reco signals
--upsampling: upsampling for transfer count to ns
--mode: snr or simple
output: 1 csv file

### 6. plotter.py
*In progress*

Used for plot graphics based on result from csv table.
arguments
--input: csv files from estimator
output: pdf files


### Simple create CNN pipeline:
1. Download prepared dataset from Google Drive:
https://drive.google.com/open?id=1ESXEmZLb20R-d8ok8n8wczhGpWduhaFx
2. Run trainer.py
python trainer.py --signal cuted_signal.npy --noise cuted_noise.npy --min 100 --max 200 --epochs 100 --arch model_baseline.json
3. Get the model in .h5 format
