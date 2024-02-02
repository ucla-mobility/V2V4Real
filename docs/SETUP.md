# Setup


## **1. Conda environment:**
```shell
conda create -n dmstrack python=3.7
conda activate dmstrack
```


## **2. V2V4Real setup:**

Note: My setup steps are different from the official V2V4Real's instruction because the official one does not work in my machine.
Make sure you follow my steps and DO NOT SKIP the first pip install spconv-cu113 even though it looks unnecessary! Very important! 
In the end of this step, you will see the error message about a dependency conflict. 
That is expected and don't worry about it. 
We will continue solving the dependency problem and it will not affect running V2V4Real's detection inference in the end.

```shell
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install spconv-cu113
cd V2V4Real
pip install -r requirements.txt
pip install open3d==0.16.0
python setup.py develop
python opencood/utils/setup.py build_ext --inplace
pip install spconv-cu113==2.1.25
cd ../
```

The expected error message is:  
"  
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.  
cumm 0.4.11 requires pccm>=0.4.2, but you have pccm 0.3.4 which is incompatible.  
"


## **3. AB3DMOT setup:**

Note: You will also see message about dependency conflicts due to the differences between the AB3DMOT and V2V4Real code bases.
That is expected and don't worry about it. 
We will continue solving the dependency problem and it will not affect running V2V4Real's detection inference and AB3DMOT's tracking and evaluation in the end.

```shell
cd AB3DMOT
pip install -r requirements.txt
cd Xinshuo_PyToolbox
pip install -r requirements.txt
cd ../../
```

The expected error message after the first pip install is:  
"  
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.  
open3d 0.16.0 requires matplotlib>=3, but you have matplotlib 2.2.3 which is incompatible.  
open3d 0.16.0 requires pyyaml>=5.4.1, but you have pyyaml 5.4 which is incompatible.  
open3d 0.16.0 requires scikit-learn>=0.21, but you have scikit-learn 0.19.2 which is incompatible.  
v2v4real 0.1.0 requires matplotlib~=3.3.3, but you have matplotlib 2.2.3 which is incompatible.  
v2v4real 0.1.0 requires numba==0.49.0, but you have numba 0.56.2 which is incompatible.  
v2v4real 0.1.0 requires opencv-python~=4.5.1.48, but you have opencv-python 4.2.0.32 which is incompatible.  
"


## **4. DMSTrack setup:**

Note: We will solve the dependency conflicts as many as possible in the step. 
You will still see one message about a dependency conflict.
That is expected and don't worry about it. 
This conda environment is good enough to run all the commands we need: 
V2V4Real cooperative detection inference, AB3DMOT regular tracking and evaluation, DMSTrack cooperative tracking's training, evaluation, visualization.

```shell
pip install -r requirements.txt
```

The expected error message is:  
"  
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.  
v2v4real 0.1.0 requires numba==0.49.0, but you have numba 0.56.2 which is incompatible.  
"


## **5. PYTHONPATH setup:** 

Set the PYTHONPATH depending on your PATH_TO_THE_CODE. You will need to run this every time when you run "conda activate dmstrack".

```shell
export PYTHONPATH=${PYTHONPATH}:${PATH_TO_THE_CODE}/DMSTrack/AB3DMOT
export PYTHONPATH=${PYTHONPATH}:${PATH_TO_THE_CODE}/DMSTrack/AB3DMOT/Xinshuo_PyToolbox
export PYTHONPATH=${PYTHONPATH}:${PATH_TO_THE_CODE}/DMSTrack/V2V4Real
export PYTHONPATH=${PYTHONPATH}:${PATH_TO_THE_CODE}/DMSTrack
```
