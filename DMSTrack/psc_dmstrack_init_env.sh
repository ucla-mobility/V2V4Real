conda activate dmstrack

# for gcp
#export MY_HOME=${HOME}
# for psc
export MY_HOME=${PROJECT}

export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/my_cooperative_tracking/AB3DMOT
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/my_cooperative_tracking/AB3DMOT/Xinshuo_PyToolbox
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/my_cooperative_tracking/V2V4Real
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/my_cooperative_tracking/DMSTrack
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/my_cooperative_tracking/

echo $PYTHONPATH

export JUPYTER_PATH=${PYTHONPATH}
