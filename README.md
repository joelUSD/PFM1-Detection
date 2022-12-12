Evaluating Deep Learning Models for Landmine Detection:

This project is a part of the AAI-521 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

Description:

This project is an evaluation of the performance of YOLOv5 for the task of detecting PFM-1 scatterable landmines and associated components in orthophoto mosaics that have been captured with various commercial off-the-shelf quadcopters. 

Installation and Dependencies:

The jupyter notebook included in this repository demonstrates the code that we ran to complete this experiment in a virtual environment, to allow for training of models on local GPUs. Different users may want to run this model in different environments, the set up of which is outside of the scope of this readme file. 

In your chosen environment, run the following command line code:  

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
git clone https://github.com/ultralytics/yolov5
pip install -qr yolov5/requirements.txt  # install requirements for yolov5

After installing the requirements in the requirements.txt file, you will need to add the Binghamton data set to the yolov5 file structure. This is done by adding the train, test, valid folders and the binghamton.yaml file to the top level of the yolov5 folder. After addition, the file path for all four items should look like this:
../yolov5/train
../yolov5/test
../yolov5/valid
../yolov5/binghamton.yaml

Add the files with extension “.pt” from this google drive link:

https://drive.google.com/drive/folders/1OibZxxgTHdULaaRAtvp0CKrTmqjwO7s8?usp=sharing

This will allow you to run the same tests that we ran with the same pretrained weights. 
../yolov5/visDroneLarge.pt

Methodology:

Train:
To replicate the results of the model evaluation that we conducted, the following lines can be run to train the model on the Binghamton dataset with the small and large models and different pretrained weights. The pretrained weights were produced by training the YOLOv5 model on the visDrone2019 dataset for 200 epochs using this code:

python train.py --img 640 --batch -1 --epochs 200 --data VisDrone.yaml --weights yolov5s.pt --name PTvisDroneSmall
python train.py --img 640 --batch -1 --epochs 200 --data VisDrone.yaml --weights yolov5l.pt --name PTvisDroneLarge

The pretrained weights are contained in the ‘.pt’ files that can be downloaded from the following google drive:

https://drive.google.com/drive/folders/1OibZxxgTHdULaaRAtvp0CKrTmqjwO7s8?usp=sharing


The following are the training runs that were compared in our accompanying write up:

Yolov5s default weights - pretrained on COCO:
train.py --img 640 --batch 4 --epochs 150 --data Binghamton.yaml --weights yolov5s.pt --name COCOSmall

Yolov5L default weights - pretrained on COCO:
python train.py --img 640 --batch 4 --epochs 150 --data Binghamton.yaml --weights yolov5l.pt --name COCOLarge

Yolov5s fine-tuned on visDrone2019:
python train.py --img 640 --batch 4 --epochs 150 --data Binghamton.yaml --weights yolov5s.pt --name COCOSmall

Yolov5L fine-tuned on visDrone2019:
python train.py --img 640 --batch 4 --epochs 150 --data Binghamton.yaml --weights runs/train/PTvisDroneLarge/weights/best.pt --name visDroneLarge

Detection:

The environment is now set up to replicate the testing that we accomplished with yolov5 and use the model for landmine detection in images. To preprocess new orthomosaics, go to www.roboflow.com and split the ortho into photos that are roughly 700 x 700 pixels. These files are then stored in a source folder in the top level of the yolov5 folder and referenced as follows when run in command line:
Python detect.py –source ‘../source’ –weights best.pt –conf 0.6 –iou 0.45 –augment –project ‘test’ –name ‘detection_test’ 

Citation and Affiliated Links:

Demining Research Group:
https://www.de-mine.com/datasets
de Smet, Timothy; Nikulin, Alex; and Baur, Jasper, "Scatterable Landmine Detection Project Dataset 1-8" (2020). Geological Sciences and Environmental Studies Faculty Scholarship. 5.
https://orb.binghamton.edu/geology_fac/5

yolov5:

https://github.com/ultralytics/yolov5
@software{glenn_jocher_2020_4154370,
author = {Glenn Jocher, Alex Stoken, Jirka Borovec, NanoCode012, ChristopherSTAN, Liu Changyu, Laughing, tkiana,Adam Hogan,lorenzomammana, yxNONG, AlexWang1900, Laurentiu Diaconu, Marc, wanghaoyang0106, ml5ah, Doug, Francisco Ingham, Frederik, Guilhen, Hatovix, Jake Poznanski, Jiacong Fang, Lijun Yu 于力军, changyu98, Mingyu Wang, Naman Gupta, Osama Akhtar,PetrDvoracek,Prashant Rai},
title  = {{ultralytics/yolov5: v3.1 - Bug Fixes and Performance Improvements}},
month = oct,
year = 2020,
publisher = {Zenodo},
version = {v3.1},
doi = {10.5281/zenodo.4154370},
url = {https://doi.org/10.5281/zenodo.4154370}
}

visDrone:

https://github.com/VisDrone/VisDrone-Dataset	
@ARTICLE{9573394,
author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan,    Heng and Hu, Qinghua and Ling, Haibin},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
title={Detection and Tracking Meet Drones Challenge}, 
year={2021},
volume={},
number={},
pages={1-1},
doi={10.1109/TPAMI.2021.3119563}}


License: 

This project is released under the Apache 2.0 license.
