# Multimodal LLMs for Annotating Complex Cultural Memes in Movie Dialogues
## - NYU 24 Spring Machine Learning Group Project

## Contributors:
> Yihan Li (@Litou-lyh), Hao Li (@haofrank), Jiayu Gu (@), Guanshi Wang (@oyster14)

> {yl10798, hl5262, jg7956, gw2310}@nyu.edu


## conda镜像
我在greene上配了个镜像，在/scratch/yl10798/ml_env，已经装好vllm了
  
你们可以复制一份: `cp /scratch/yl10798/ml_env/vllm.ext3 /scratch/<NetID>/your/path/`
  
运行 `ssh burst`, 然后申请机器：`srun --account=csci_ga_2565-2024sp --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=04:00:00 --pty /bin/bash`, 参考brightspace上ML课的announcement
也可以用 `sbatch`
  
进入机器后，运行 `singularity exec --overlay /scratch/<NetID>/your/path/vllm.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash`
  
进入Singularity容器后需要运行 `source /ext3/env.sh`，激活conda环境
  
使用 `python`, `import vllm`, 验证vllm已安装
  
配置镜像可以参考这里: https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda

[vllm](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
  
[llama2+vllm](https://github.com/meta-llama/llama-recipes/blob/main/recipes/inference/model_servers/llama-on-prem.md)

## vllm load llama
0.复制conda镜像到自己的目录

1.连接burst：
ssh burst

2.申请机器：
srun --account=csci_ga_2565-2024sp --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=04:00:00 --pty /bin/bash


scp将conda镜像传到slurm集群，把{}替换为自己的：
scp /scratch/{yl10798/ml_env}/vllm.ext3 {b-3-17}:/scratch/{yl10798/ml_env}/vllm.ext3


运行cuda镜像, 其中ext3中为自己的conda镜像：
./run-cuda-12.1.bash

下载llama模型：参考 https://zhuanlan.zhihu.com/p/651444120

git config --global credential.helper store

huggingface-cli login
填入huggingface创建的token
然后 python 跑下面脚本，下载到默认缓存地址：
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

根据vllm get started运行测试

nvidia-smi查看现存占用：13G左右

## Tentative plan

  - [ ] Run vllm or any LLM to get reasonable output (on greene). Assign: Hao
  - [X] Find movie dialogue (script) dataset. Assign:
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
  - The answer is YES. For our task, we need to input with sliding context window of dialogues.
  - Would reduce memory usage, and might speed up
  - Also, relocate context start once a meme is detected. Think substring matching! This can speed up inference!
    
- Consider the different lengths of context windows
  - Three senerios:
    - Seperate jokes / terms
    - Consecutive jokes
    - Callback (大海捞针)
- Parallelized execution of different size of contexts
  - Parallelize short contexts, ensure completion earlier than the whole context one
  - Dynamically append REMINDERs to the input sequence that containing the whole context
    - Candidates of REMINDERs:
      - Repeating occurance
      - Detected jokes
      - 专有名词（英语不会）

