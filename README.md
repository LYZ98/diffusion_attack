# Adversarial-Diffusion-Attacks-on-Graph-based-Traffic-Prediction-Models
The code consists of two parts: model_train and model_attack. In the model_train section, you can execute the "main.py" file to train your model. We offer three types of models: st-gcn, t-gcn, and a3t-gcn. For t-gcn and a3t-gcn, the code is based on the code available at https://github.com/lehaifeng/T-GCN. As for st-gcn, the code is based on the code provided at https://github.com/VeritasYin/STGCN_IJCAI-18. Additionally, you have the option to train corresponding models with Dropout, Dropnode, and Dropedge. The model information will be generated in the "model_train/out" folder.

The SPSA algorithm code is derived from https://github.com/TheBugger228/SPSA. In the model_attack section, you can run the "attack_algorithm_comparision.py" file to launch attacks on the models using different algorithms mentioned in our paper. We have shared the trained models on our Google Drive at https://drive.google.com/drive/folders/1sVoQxd7yH0PVR-g1Ni1HMM2vREjtp1l8?usp=sharing. If you wish to attack new models that you have trained yourself, you can train them using the "model_train/main.py" file and obtain the results in the "model_train/out" folder. Simply copy the "out" folder from "model_train/out" to "model_attack/out" and ensure that the file names match. Finally, execute the "attack_algorithm_comparision.py" file.


<p>Please cite our work if you utilize datasets or codes from our research：</p>
<pre><code>@ARTICLE{10167720,
  author={Zhu, Lyuyi and Feng, Kairui and Pu, Ziyuan and Ma, Wei},
  journal={IEEE Internet of Things Journal}, 
  title={Adversarial Diffusion Attacks on Graph-Based Traffic Prediction Models}, 
  year={2024},
  volume={11},
  number={1},
  pages={1481-1495},
  doi={10.1109/JIOT.2023.3290401}
 }

</code></pre>


