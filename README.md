# Leapfrog Diffusion Model for Stochastic Trajectory Prediction （LED）

Official **PyTorch** code for CVPR'23 paper "Leapfrog Diffusion Model for Stochastic Trajectory Prediction".


## 1. Overview

<div align="center">  
  <img src="./results/fig/overview.png" width = "60%" alt="system overview"/>
</div>

**Abstract**: To model the indeterminacy of human behaviors, stochastic trajectory prediction requires a sophisticated multi-modal distribution of future trajectories. Emerging diffusion models have revealed their tremendous representation capacities in numerous generation tasks, showing potential for stochastic trajectory prediction. However, expensive time consumption prevents diffusion models from real-time prediction, since a large number of denoising steps are required to assure sufficient representation ability. To resolve the dilemma, we present **LEapfrog Diffusion model (LED)**, a novel diffusion-based trajectory prediction model, which provides  real-time, precise, and diverse predictions. The core of the proposed LED is to leverage a trainable leapfrog initializer to directly learn an expressive multi-modal distribution of future trajectories, which skips a large number of denoising steps, significantly accelerating inference speed. Moreover, the leapfrog initializer is trained to appropriately allocate correlated samples to provide a diversity of predicted future trajectories, significantly improving prediction performances. Extensive experiments on four real-world datasets, including NBA/NFL/SDD/ETH-UCY, show that LED consistently improves performance and achieves **23.7\%/21.9\%** ADE/FDE improvement on NFL. The proposed LED also speeds up the inference **19.3/30.8/24.3/25.1** times compared to the standard diffusion model on NBA/NFL/SDD/ETH-UCY, satisfying real-time inference needs.

<div  align="center">  
  <img src="./results/fig/mean_var_estimation.png" width = "50%" alt="mean and variance estimation"/>
</div>
Here, we present an example (above) to illustrate the mean and variance estimation in the leapfrog initializer under four scenes on the NBA dataset. We see that the variance estimation can well describe the scene complexity for the current agent by the learned variance, showing the rationality of our variance estimation.


## 2. Code Guidance

Overall project structure:
```text
----LED\   
    |----README.md
    |----requirements.txt # packages to install                    
    |----main_led_nba.py  # [CORE] main file
    |----trainer\ # [CORE] main training files, we define the denoising process HERE!
    |    |----train_led_trajectory_augment_input.py 
    |----models\  # [CORE] define models under this file
    |    |----model_led_initializer.py                    
    |    |----model_diffusion.py    
    |    |----layers.py
    |----utils\ 
    |    |----utils.py 
    |    |----config.py
    |----data\ # preprocessed data (~200MB) and dataloader
    |    |----files\
    |    |    |----nba_test.npy
    |    |    |----nba_train.npy
    |    |----dataloader_nba.py
    |----cfg\ # config files
    |    |----nba\
    |    |    |----led_augment.yml
    |----results\ # store the results and checkpoints (~100MB)
    |----visualization\ # some visualization codes
```

Please download the data and results from [Google Drive](https://drive.google.com/drive/folders/1Uy8-WvlCp7n3zJKiEX0uONlEcx2u3Nnx?usp=sharing). 

**TODO list**:

- [ ] add training/evaluation for diffusion models (in two weeks).
- [ ] more detailed descripition in trainers (in one month).
- [ ] transfer the parameters in models into yaml.
- [ ] other fast sampling methods (DDIM and PD).


### 2.1. Environment
We train and evaluate our model on `Ubuntu=18.04` with `RTX 3090-24G`.

Create a new python environment (`led`) using `conda`:
```
conda create -n led python=3.7
conda activate led
```

Install required packages using Command 1 or 2:
```bash
# Command 1 (recommend):
pip install -r requirements.txt

# Command 2:
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install easydict
pip install glob2
```



### 2.2. Training

You can use the following command to start training the initializer.
```bash
python main_led_nba.py --cfg <-config_file_name_here-> --gpu <-gpu_index_here-> --train 1 --info <-experiment_information_here->

# e.g.
python main_led_nba.py --cfg led_augment --gpu 5 --train 1 --info try1
```

And the results are stored under the `./results` folder.



### 2.3. Evaluation

We provide pretrained models under the `./checkpoints` folder.

**Reproduce**. Using the command `python main_led_nba.py --cfg led_augment --gpu 5 --train 0 --info reproduce` and you will get the following results:
```text
[Core Denoising Model] Trainable/Total: 6568720/6568720
[Initialization Model] Trainable/Total: 4634721/4634721 
./checkpoints/led_new.p  
--ADE(1s): 0.1766       --FDE(1s): 0.2694
--ADE(2s): 0.3693       --FDE(2s): 0.5642
--ADE(3s): 0.5817       --FDE(3s): 0.8366
--ADE(4s): 0.8095       --FDE(4s): 1.0960 
```



## 3. Citation
If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{mao2023leapfrog,
  title={Leapfrog Diffusion Model for Stochastic Trajectory Prediction},
  author={Mao, Weibo and Xu, Chenxin and Zhu, Qi and Chen, Siheng and Wang, Yanfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5517--5526},
  year={2023}
}
```



## 4. Acknowledgement

Some code is borrowed from [MID](https://github.com/Gutianpei/MID), [NPSN](https://github.com/InhwanBae/NPSN) and [GroupNet](https://github.com/MediaBrain-SJTU/GroupNet). We thank the authors for releasing their code.

[![Star History Chart](https://api.star-history.com/svg?repos=MediaBrain-SJTU/LED&type=Date)](https://star-history.com/#MediaBrain-SJTU/LED&Date)
