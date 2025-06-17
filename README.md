# CARDS: Caching-Offloading-Routing Dynamic Synergistic Edge Network

The growing need for real-time service delivery, such as in vehicular networks and latency-sensitive applications, has made edge networks a promising solution. 
However, current caching and offloading schemes in edge networks fall short in managing the entire processing of requests with distinct granularity. 
This paper proposes CARDS, an innovative three-time-scale online edge network architecture predicated on the integration of caching, offloading, and routing, aimed at enhancing service facilitation in edge computing. 
Our key idea is to decompose the edge network into three time granularities ranging from fine to coarse, and to map these granularities one-to-one with computational, storage, and communication resources, as well as with offloading, caching, and routing.
The main challenge lies in the synergy of three granularities, along with their reliable online solving and deployment.
We employ two core technologies: a polynomial-time solution with guaranteed lower bounds based on submodular functions, and a dynamic programming working as a bridge between constraint programming and deep reinforcement learning. 
Additionally, we provide a framework that considers inter-frame interactions, along with distributed and adaptive deployment strategies. 
We have implemented a CARDS prototype in testbed experiments and demonstrated that it delivers up to 3.2$\times$ improvement in served requests and 37.6\% reduction in scheduling time over baseline solutions. 

## Content of the repository

For each problem that we have considered, you can find:

* A single-frame caching method SFC.
* * without coloring greedy
* * with coloring greedy improving
* A multi-frame caching method DOMF. 
* * simply migrate single-frame greedy
* * optimized single-frame greedy
* * DOMF with flooding greedy and single-frame greedy
* A request offloading & routing method.
* * based on DRL and CP


```bash
.
├── single-frame-caching # SFC
    ├── submodel_algorithm_v1.py  # submodular greedy
    ├── submodel_algorithm_v2.py  # submodular coloring greedy
    ├── yen.py  # Yen algorithm for solving dnmop
    ├── requirements.txt  # environment
    ├── output_data  # output
    ├── lingo/
        ├── mymethord.lg4 # lingo solving example 1
        └── mymethord2.lg4 # lingo solving example 2
    └── data_dir/
        ├── lingoExample.txt # lingo data input (manually)
        └── pythonExample.txt # python data input (through exec in python script)
├── multi-frame-caching # DOMF
    ├── submodel_algorithm_multiple_frame_v1.py # common submodular greedy
    ├── submodel_algorithm_multiple_frame_v2.py # optimized submodular greedy
    ├── submodel_algorithm_multiple_frame_v3.py # flooding greedy + optimized submodular greedy
    ├── submodel_algorithm_threads.py # multi threads caching
    ├── requirements.txt  # environment
    ├── output_data  # output
    ├── lingo/
        ├── mymethord.lg4 # single-frame lingo solving
        └── mymethord_multiple_frame.lg4 # multi-frame lingo solving
    └── data_dir/framedataExample/
        ├── allframeExample.txt # all frame data in one file (python + lingo)
        ├── frame1Example.txt # data split by frame
        :
        └── frame5Example.txt # data split by frame
├── fine-grained-problem # DRL
    ├── environmentwyb.yml  # environment
    ├── requirements.txt # environment
    ├── text.py # test environment
    ├── run_training_x_y.sh  # script for running the training. enter the parameters here 
    ├── trained_models/  # directory for saving models
    ├── selected-models/  # select models for using
    └── src/ 
        ├── architecture/ # implementation of the NN used
            └── util/  #  util code (as the memory replay and yen for data generating)
        └── problem/  # cards problem
            └── cards/ 
                ├── main_training_x_y.py  # main file for training a model for the problem y using algorithm x
                ├── solving/ # solve problem
                ├── environment/ # the generator, and the DP model, acting also as the RL environment
                └── training/  # training algorithms   
└── docker
    └── dockerfile # docker environment
```

## quick start
### single frame caching (SFC)
#### environment preparation
```shell
conda create -n singleframe python=3.8
conda activate singleframe
cd single-frame-caching
pip install -r requirements.txt
```
#### run code
```shell
cd single-frame-caching
python submodel_algorithm_v2.py
```

### multi-frame caching (DOMF)
```shell
conda create -n multiframe python=3.8
conda activate multiframe
cd multi-frame-caching
pip install -r requirements.txt
```
#### run code
```shell
cd multi-frame-caching
python submodel_algorithm_multiple_frame_v3.py
```

### offloading & routing (DRL)
#### environment preparation
```shell
cd fine-grained-problem
conda env create -f environmentwyb.yml 
conda env list
conda activate pytorch_env
```
note: our environment is with cuda 11.8

or 
```shell
cd fine-grained-problem
pip install -r requirements.txt
```

#### run code
```shell
cd fine-grained-problem
bash run_training_ppo_cards.sh 
```