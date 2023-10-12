# PHDiffusion-Painterly-Image-Harmonization

This is the official repository for the following paper:

> **Painterly Image Harmonization using Diffusion Model**  [\[arXiv\]](https://arxiv.org/pdf/2308.02228.pdf)<br>
>
> Lingxiao Lu, Jiangtong Li, Junyan Cao, Li Niu, Liqing Zhang<br>
> Accepted by **ACM MM 2023**.

### Code and model

1.  Dependencies

    *   Python == 3.8
    *   Pytorch == 1.11.0
    *   Run

        ```bash
        pip install -r requirements.txt
        ```
2.  Download Models

    Please download the following models to the `pretrained_model/` folder.

    *   sd-v1-4. We trained our adapter and dual encoder fusion model based on Stable Diffusion v1.4.&#x20;
    *   [vgg\_normalised](). For training, loss is calculated with the help of VGG.
    *   [PHDiffusionWithRes](). The best checkpoint of our adapter with residual and dual encoder fusion module.
    *   [PHDiffusionWithoutRes](). The best checkpoint of our adapter without residual and dual encoder fusion module.
3.  Train

    You can run this to train adapter and dual encoder fusion module:

    ```bash
    CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 train.py
    ```
4.  Test

    You can run this to test using adapter with residual:

    ```bash
    CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 test.py --strength 0.7 --model_resume_path pretrained_models/PHDiffusionWithRes.pth
    ```

    And run this to test using adapter without residual

    ```bash
    CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 test.py --strength 0.7 --model_resume_path pretrained_models/PHDiffusionWithoutRes.pth --no_residual
    ```

### Experimental results

Our method can significantly outperform GAN-based methods when the background has dense textures or abstract style.

<p align="center">  
  <img src="./examples/results1.jpg" width="90%" />
</p>

<p align="center">  
  <img src="./examples/results2.jpg" width="90%" />
</p>

## Other Resources

*   [Awesome-Image-Harmonization](https://github.com/bcmi/Awesome-Image-Harmonization)
*   [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)

