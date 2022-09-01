# Datathon2019_team4
Subject : Ache index from biosignal 

## This is code for KOREA CLINICAL DATATHON 2019
- Development of pain indicator for patients

## Data
- Bio signal data, 1-D array
- Data is private as patient's privacy.  

## Model Architecture
- 1-D convolutional layers and linear layers with activations.
```
.
├── data.py
├── main.py
├── model.py
├── __pycache__
│   ├── data.cpython-37.pyc
│   ├── main.cpython-37.pyc
│   ├── model.cpython-37.pyc
│   ├── train.cpython-37.pyc
│   └── utils.cpython-37.pyc
├── train.py
└── utils.py

```

## Requirements
pytorch 1.0.0+\
python 3.6+ \
numpy 1.2+
  
## Usage
```
bash
cd BIOSIG
```
```
python main.py --data_dir __USER DATA__ --save_dir __USER DIR__ --save_name __USER NAME__
```
