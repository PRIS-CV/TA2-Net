# Task-aware Adaptive Learning for Cross-domain Few-shot Learning

Code release for "Task-aware Adaptive Learning for Cross-domain Few-shot Learning" 

**Abstract**: Although existing few-shot learning works yield promising results for in-domain queries, they still suffer from weak cross-domain generalization. Limited support data requires effective knowledge transfer, but domain-shift makes this harder.  Towards this emerging challenge, researchers improved adaptation by introducing task-specific parameters, which are directly optimized and estimated for each task. However, adding a fixed number of additional parameters fails to consider the diverse domain shifts between target tasks and the source domain, limiting efficacy. In this paper, we first observe the dependence of task-specific parameter configuration on the target task. 
Abundant task-specific parameters may over-fit, and insufficient task-specific parameters may result in under-adaptation -- but the optimal task-specific configuration varies for different test tasks.
Based on these findings, we propose the Task-aware Adaptive Network (TA$^2$-Net), which is trained by reinforcement learning to adaptively estimate the optimal task-specific parameter configuration for each test task. It learns, for example, that tasks with significant domain-shift usually have a larger need for task-specific parameters for adaptation.
We evaluate our model on the Meta-dataset. Empirical results show that our model outperforms existing state-of-the-art methods.



## Dependencies
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.0 or greater
* TensorFlow 1.14 or greater

## Data & Pre-trained Weights
* Clone or download this repository.
* Configure Meta-Dataset:

You may refer to this [repo](https://github.com/VICO-UoE/URL) to download the datasets and pre-trained model weights as we followed most of the settings in **Cross-domain Few-shot Learning with Task-specific Adapters**.

## Initialization

  Before doing anything, first run the following commands as this [repo](https://github.com/VICO-UoE/URL).
    
    ulimit -n 50000
    export META_DATASET_ROOT=<root directory of the cloned or downloaded Meta-Dataset repository>
    export RECORDS=<the directory where tf-records of MetaDataset are stored>

    
    Note the above commands need to be run every time you open a new command shell.

## Training
-   load the pre-trained backbone before training
- `bash scripts/meta_train_resnet18_TA2_Net.sh` for training the proposed TA^2-Net
 

## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- guoyurong@bupt.edu.cn

## Acknowledgement
Our code is mainly built upon [Cross-domain Few-shot Learning with Task-specific Adapters](https://github.com/VICO-UoE/URL). We appreciate their unreserved sharing.
