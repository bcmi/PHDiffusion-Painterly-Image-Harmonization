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

    *   sd-v1-4. Our model is based on Stable Diffusion v1.4.&#x20;
    *   Our pretrained model. You can download **one** of them for testing. The main difference between the two models is that **PHDiffusionWithoutRes** removes the residual structure in its adapter, while **PHDiffusionWithRes** retains it. Note that **PHDiffusionWithoutRes** perform better in some dense texture styles, learning textures that are more similar to the original ones. While **PHDiffusionWithRes** can preserve better content. You can make selections based on your needs.

        *   [PHDiffusionWithoutRes](https://drive.google.com/file/d/1mP9fUXF58jJGOB28YB0hoi1yK-L5xOj_/view?usp=sharing). The best checkpoint of our adapter **without** residual and dual encoder fusion module.
        *   [PHDiffusionWithRes](https://drive.google.com/file/d/1cJy4N7kzEcjsp5c__--ymmGTjvL2w1Cs/view?usp=sharing). The best checkpoint of our adapter **with** residual and dual encoder fusion module.
    *   [VGG19](https://drive.google.com/file/d/1pZpi45kJi-vnTfQIPrin69MLsUJO0Y_x/view?usp=sharing)(Optional, only needed for training). Loss is calculated with the help of VGG.


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

    And run this to test using adapter without residual:

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

