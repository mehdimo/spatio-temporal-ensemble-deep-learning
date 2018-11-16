# Ensemble Deep Learning using Spatio-Temporal Patterns in IoT Data

This is an implementation of this paper: 

M. Mohammadi, A. Al-Fuqaha, "Exploiting the Spatio-Temporal Patterns in IoT Data to Establish a Dynamic Ensemble of Distributed Learners," IEEE Access, Vol. PP, No. 99, 2018.

## How to use:

* `$ python ensemble_model.py --dataset [electricity|traffic|BLE] [--i]`

  * Choose one of datasets electricity, traffic, or BLE to train and evaluate
  * `--i` force running the individual models to be trained and generate predictions.
