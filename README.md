# Federated Learning With Opacus (PyTorch)

In this repository, I used PyTorch and Opacus to simulate the __fedavg__ and __fedprox__ algorithm of federated learning. And add __differential privacy__ in the models.


# Requirments

* python == 3.8.0
* pytorch == 1.12.1
* torchvision == 0.13.1
* opacus == 1.4.0
* numpy == 1.23.2
* matplotlib == 3.2.0
* scikit-learn == 1.2.2

# Running the experiments
* Running fedavg algorithm with differential privacy
```
python main.py --dp True
```
* Running fedprox algorithm with differential privacy
```
python main.py --dp True --fedprox True
```

# Advantage
Compared to the fedavg done by others, my model is more featured, you can see my code.
