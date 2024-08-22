#conda activate ab3dmot

conda activate dp3dmoct
# for gcp
#export MY_HOME=${HOME}
# for psc
export MY_HOME=${PROJECT}

export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/my_cooperative_tracking/AB3DMOT
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/my_cooperative_tracking/AB3DMOT/Xinshuo_PyToolbox
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/my_cooperative_tracking/V2V4Real
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/my_cooperative_tracking

export JUPYTER_PATH=${PYTHONPATH}
