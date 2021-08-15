#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python setup.py install --user

# NOTE: For GRNet 

# Cubic Feature Sampling
cd $HOME/extensions/cubic_feature_sampling
python setup.py install --user

# Gridding & Gridding Reverse
cd $HOME/extensions/gridding
python setup.py install --user

# Gridding Loss
cd $HOME/extensions/gridding_loss
python setup.py install --user

