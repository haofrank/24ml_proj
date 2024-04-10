# Multimodal LLMs for Annotating Complex Cultural Memes in Movie Dialogues
## - NYU 24 Spring Machine Learning Group Project

## Contributors:
> Yihan Li (@Litou-lyh), Hao Li (@haofrank), Jiayu Gu (@), Guanshi Wang (@oyster14)

> {yl10798, hl5262, jg7956, gw2310}@nyu.edu

## Tentative plan
- Week 0401
  - [X] Built conda env image on Greene. By: yihan
  > 我在greene上配了个镜像，在/scratch/yl10798/ml_env，已经装好vllm了。你们可以复制一份，然后用这个命令跑：singularity exec --overlay /scratch/yl10798/ml_env/vllm.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash
  > 可以参考这里https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda

  - [ ] Run vllm or any LLM to get reasonable output (on greene). Assign: Hao
  - [ ] Find movie dialogue (script) dataset. Assign:
  - [ ] Design prompts (append to dialogues). Can test using GPT4 / Claude 3 first. Assign:
  - [ ] Survey on serving techs (e.g. quantization). Assign:

- Week 0408
  - [ ] Write streaming input scripts. Assign:
  - [ ] Benchmark mean response time with streaming input. (may also include memory usage). Assign:
  - [ ] Try current serving techs. Assign:
  - [ ] Try distributed inference. Assign:
- Week 0415
- Week 0422
- Week 0429

## Task-specific improvement
- KV Cache sharing
  - Consider the Complex decoding algorigthms mentioned in vLLM paper: can we share the KV caches?
  - The answer is YES.
  - For our task, we need to input with sliding context window of dialogues.
    - Also, relocate context start once a meme is detected. Think substring matching!
