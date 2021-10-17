# Adversarial-Diffusion-Attacks-on-Graph-based-Traffic-Prediction-Models
Our paper is available at http://arxiv.org/abs/2104.09369.
Author: Lyuyi Zhu (3170103586@zju.edu.cn), Kairui Feng (kairuif@princeton.edu), Ziyuan Pu (ziyuan.pu@monash.edu), Wei Ma (wei.w.ma@polyu.edu.hk).

The code includes two parts: model_train and model_attack.

In model_train, you could run the file "main.py" to train your model. Here we provide 3 types of models: st-gcn, t-gcn, a3t-gcn.
(For t-gcn and a3t-gcn, the code is based on codes in https://github.com/lehaifeng/T-GCN.
 For st-gcn, the code is based on codes in https://github.com/VeritasYin/STGCN_IJCAI-18).
You can also train correspoding models with Dropout, Dropnode and Dropedge.
The model information is generated in folder "model_train/out".
 
In model_attack, you could run "attack_algorithm_comparision.py" to attack the models with different algorithms in our paper. Here we provide trained models in our Google drive https://drive.google.com/drive/folders/1sVoQxd7yH0PVR-g1Ni1HMM2vREjtp1l8?usp=sharing.
If you want to attack new models which are trained by yourself, you could train them in "model_train/main.py" file, then get results in "model_train/out" folder. You could copy this "out" folder from "model_train/out" to "model_attack/out", then run file "attack_algorithm_comparision.py". The file name should be the same if you want to restore models sucessfully.

The code of SPSA algorithm is based on https://github.com/TheBugger228/SPSA.
 
 
