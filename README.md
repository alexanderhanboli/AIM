# LAI: Latent-likelihood Adversarial Inference

![](https://img.shields.io/badge/python-3.5.2-brightgreen.svg)

<center>
<img src="https://github.com/AlexanderYogurt/LAI/blob/master/demo/reconstructed_generate_animation.gif" width="128">
 </center>
## Example
1. MNIST
```bash
python main.py --dataset 'mnist' --root 'your/root/directory' --epoch 100 --batch_size 64 --beta1 0.5
```

2. Mixed Gaussian
```bash
python main.py --dataset 'mixed-Gaussian' --root 'your/root/directory' --epoch 200 --batch_size 100 --beta1 0.8
```

3. dcGAN structure LAI
```bash
python main.py --model_name 'dcLAI' --dataset 'svhn' --root 'your/root/directory' --epoch 20 --batch_size 128 --beta1 0.5 --z_dim 100
```
