<div align="center">

# RDLCM
</div>

## About RDLCM
Regional-Division Local Contrast Measure (RDLCM) is specially designed for the infrared small dim targets detection appearing near the SSL.
This code aims to reproduce the results of RDLCM in our paper. Download the datasets [here](https://drive.google.com/file/d/1g9x0jbV7yjp7qeEcxOHHzufkZHCrYk16/view?usp=sharing), 
and unzip them in ./data/. The labeled sequence file is in ./jsonFile/ and the trained model for sea-sky line detection is included in ./experiments/Horizon/models.
 
## About the dataset

The images containing targets appearing near the SSL are extracted from the open-source dataset [InfML-HDD](https://github.com/FJsRepo/InfML-HDD), 
these images were divided into three sequences based on the imaging environment characteristics. 
All images are in PNG format with a resolution of 288$\times$384 pixels and shot with an infrared camera $Xinfrared$ with band 8$\sim$14 $\mu m$. 
The labeling tool is Labelme \cite{labelme}.
## Run detect.py to validate RDLCM
To run the code you should first configure the correct environment, the details could refer to [InfML-HDD](https://github.com/FJsRepo/InfML-HDD).

The whole project is based on Python 3 and PyTorch, the version we used is Python-3.6.13 and PyTorch-1.7.1, 
the project and other key packages could be downloaded and installed by:

```
$ git clone https://github.com/FJsRepo/RDLCM.git
$ cd RDLCM
$ pip install -r requirements.txt
```

If your IDE is Pycharm, open the "Edit configuration" and set the parameters as follows:
```
--exp_name Horizon --cfg TargetDetection.yaml
```

