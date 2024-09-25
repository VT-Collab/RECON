# RECON: Reducing Causal Confusion with Human-Placed Markers

In this repository we provide our implmentation of RECON for the Dynamic 2D simulated environment

# Install 

The code was written and tested in python 3.11.0

1. (Optional) Crea a conda environment 

```
conda create -n RECON python==3.8.10
conda activate RECON
```

2. Clone the repository 

```
https://github.com/VT-Collab/RECON.git
```

3. Install dependencies

```
pip install -r requirements.txt
```

# Training model 

A set of trained model has been provided. If you want to retrain all model a bash files is include. The bash file will generate 10 scenarios, and train all models.

```
./bash.sh
```

For retraining and individual model run:

```
python train_vision_model.py --beacon [Model]
```

The model list includes: ['Exact', 'Partial', 'Other', 'Random']. For training the baseline run the script without the ``` --beacon ``` argument.

# Testing models

By default the testing file will run for 100 iterations and render 10 of those iterations. At the end it will print the average reward results on the terminal. It is possible to modify the number of iterations and render frequency by using the ``` --scenarios ``` and ``` --render_freq ``` arguments.

```
python test_models.py --scenarios 100 --render_freq 10
```
