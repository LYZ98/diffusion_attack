# Adversarial-Diffusion-Attacks-on-Graph-based-Traffic-Prediction-Models
The codes includes two parts: model_train and model attack.

In model_train, you could run the file "main.py" to train your model. Here we provide 3 types of models: st-gcn, a3t-gcn, st-gcn.
(For t-gcn and a3t-gcn, the codes are based on codes in https://github.com/lehaifeng/T-GCN.
 For st-gcn, the codes are based on codes in https://github.com/VeritasYin/STGCN_IJCAI-18).
You could also train correspoding models with Dropout, Dropnode and Dropedge.
The models information are in file "out".
 
In model_attack, you could run attack_algorithm_comparision.py to attack the models with different algorithms in our paper. Here we provide trained models in file "out".
If you want to attack a new models, you could train it in "model_train/main.py" file, then get results in "model_train/out" file. You could copy this "out" file from "model_train" to "model_attack". The file name should be the same if you want to restore models.
The codes of SPSA algorithm are based on https://github.com/TheBugger228/SPSA.
 
 
