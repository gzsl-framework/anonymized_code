## On the Transferability of Visual Features in Generalized Zero-Shot Learning

### Framework for large scale analysis of Generalized Zero-Shot Learning (GZSL) methods

#### About
Our work provides a comprehensive benchmark for Generalized Zero-Shot Learning (GZSL). We benchmark extensively the utility of different GZSL methods which we characterize as embedding-based, generative-based, and based on semantic disentanglement. We particularly investigate how these previous methods for GZSL fare against CLIP, a more recent large scale pretrained model that claims zero-shot performance by means of being trained with internet scale multimodal data. Our findings indicate that through prompt engineering over an off-the-shelf CLIP model, it is possible to surpass all previous methods on standard benchmarks for GZSL: CUB (Birds), SUN (scenes), and AWA2 (animals). While it is possible that CLIP has actually seen many of the unseen categories in these benchmarks, we also show that GZSL methods in combination with the feature backbones obtained through CLIP contrastive pretraining (e.g. ViT~L/14) still provide advantages in standard GZSL benchmarks over off-the-shelf CLIP with prompt engineering. In summary, some GZSL methods designed to transfer information from seen categories to unseen categories still provide valuable gains when paired with a comparable feature backbone such as the one in CLIP. Surprisingly, we find that generative-based GZSL methods provide more advantages compared to more recent methods based on semantic disentanglement. We release a well-documented codebase which both replicates our findings and provides a modular framework for analyzing representation learning issues in GZSL.

#### Requirements
- python >= 3.7.7 
- pytorch > 1.5.0
- torchvision
- tensorflow-gpu==1.14
- torchcontrib

#### FAQ
- Code is based on original authors implementations, including seed and hyperparameter selection.
- Codebase should be used to reproduce the results we report.
- Run the command below to reproduce the CADA-VAE results on CUB using the RN101 features:
```
python main.py --method CADA --dataset CUB
```

Everything you need to run is in main.py.
The Wrapper class contains all the main functions to create the model, prepare the dataset, and train your model. The arguments you pass are handled by the Wrapper.

```python
usage: main.py [-h] [--dataset DATASET]
               [--feature_backbone {resnet101,CLIP,resnet152,resnet50,resnet50_moco,googlenet,vgg16,alexnet,shufflenet,vit,vit_large,adv_inception_v3,inception_v3,resnet50_clip,resnet101_clip,resnet50x4_clip,resnet50x16_clip,resnet50x64_clip,vit_b32_clip,vit_b16_clip,vit_l14_clip,virtex,virtex2,mlp_mixer,mlp_mixer_l16,vit_base_21k,vit_large_21k,vit_huge,deit_base,dino_vitb16,dino_resnet50,biggan_138k_128size,biggan_100k_224size,vq_vae_fromScratch,soho,combinedv1,combinedv2,vit_l14_clip_finetune_v2,vit_l14_clip_finetune_classAndAtt,vit_l14_clip_finetune_class200Epochs,vit_l14_clip_finetune_trainsetAndgenerated_100Epochs,vit_l14_clip_finetune_trainsetAndgenerated_200Epochs,vit_l14_clip_finetuned_classAndAtt_200Epochs,vit_l14_clip_finetuned_setAndgenerated_classAndAtt_100Epochs,vit_l14_clip_finetuned_setAndgenerated_classAndAtt_200Epochs,clip_l14_finetune_classes_200epochs,clip_l14_finetun_atts_200epochs,clip_l14_finetun_atts_200epochs,clip_l14_finetune_classes_200epochs_frozenAllExc1Layer,clip_l14_finetun_atts_200epochs_frozenAllExc1Layer,clip_l14_finetune_classAndAtt_200epochs_frozenAllExc1Layer,clip_l14_finetune_classes_200epochs_frozenTextE,clip_l14_finetun_atts_200epochs_frozenTextE,clip_l14_finetune_classAndAtt_200epochs_frozenTextE,clip_l14_finetun_atts_fromMAT_200epochs,clip_l14_finetun_classAndatts_fromMAT_200epochs,clip_l14_finetun_class_fromMAT_200epochs,vit_large_finetune_classes_200epochs}]
               [--methods {DEVISE,ESZSL,ALE,CADA,tfVAEGAN,CE,SDGZSL,FREE,UPPER_BOUND}]
               [--finetuned_features] [--data_path DATA_PATH]
               [--workers WORKERS] [--dropout DO] [--optimizer OPTIMIZER]
               [--epochs N] [--start_epoch N] [-b N] [--lr LR]
               [--initial_lr LR] [--lr_rampup EPOCHS]
               [--lr_rampdown_epochs EPOCHS] [--momentum M] [--nesterov]
               [--weight-decay W] [--doParallel] [--print_freq N]
               [--root_dir ROOT_DIR] [--add_name ADD_NAME] [--exp_dir EXP_DIR]
               [--load_from_epoch LOAD_FROM_EPOCH] [--seed SEED]

```

#### Features

| Datasets | Backbone Types | GZSL Families |
| :-----------: | :-----------: | :-----------: |
| CUB | CNN | Embedding-based |
| SUN | ViT | Generative-based |
| AWA2 | MLP-Mixer | Disentanglement-based |

<!-- <br/> -->

_______

- :white_check_mark: All 54 visual features for all datasets are [available here!](https://drive.google.com/drive/folders/14NQE2px2GPh6aucMk6aPfuiikdiSGduI?usp=sharing) :star: 
- :white_check_mark: Initial codebase is now available! :arrow_double_up:
- :black_square_button: Please expect regular updates and commits of this repo.
