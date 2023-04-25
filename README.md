# Federated Learning With Opacus (PyTorch)

In this repository, I used PyTorch and Opacus to simulate the fedavg algorithm of federated learning. And add differential privacy in the models.


# Requirments

* Python3
* Pytorch
* Torchvision
* Opacus


# Running the experiments
* Running fedavg algorithm with differential privacy
```
python main.py
```
* Running fedprox algorithm with differential privacy
```
python main.py --fedprox True
```

# Advantage
Compared to the fedavg done by others, my model is more featured, you can see my code.
