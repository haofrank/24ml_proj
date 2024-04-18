# Multimodal LLMs for Annotating Complex Cultural Memes in Movie Dialogues
## - NYU 24 Spring Machine Learning Group Project

## Contributors:
> Yihan Li (@Litou-lyh), Hao Li (@haofrank), Jiayu Gu (@), Guanshi Wang (@oyster14)

> {yl10798, hl5262, jg7956, gw2310}@nyu.edu


## PATH 1: vllm

### conda镜像
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

### load llama
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

### Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

根据vllm get started运行测试

nvidia-smi查看现存占用：13G左右

## PATH 2: TensorRT LLM + Triton Server

This part, we convert the original model (meta-llama/Llama-2-7b-chat-hf) to TensorRT LLM, The use Triton Server load it as a http server.

### Env initialization (only once)

Use the Triton latest release image version [24-03](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-24-03.html#rel-24-03) as our's work ENV. Setup follow these steps:

1. Get the machine on burst.
    
    `srun --mpi=pmi2 --account=csci_ga_3033_077-2024sp --partition=n1s8-v100-1 --gres=gpu:1 --time=01:00:00 --pty /bin/bash`

2. Pull or Copy the image to the scratch path.


    ``` shell
    cd /scratch/{NetID}
    singularity pull  docker://nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3
    
    # or copy tritonserver_24.03-trtllm-python-py3.sif
    # cp -r /scratch/hl5262/tritonserver_24.03-trtllm-python-py3.sif /scratch/{NetID} 
    ```

3. singularity run this image.
    Here is the **run** shell file. You can copy the contens in to one file. Then execute this file **get in the container**.

    `vim ~/run-triton3.bash`
    ``` shell
    #!/bin/bash

    args=
    for i in "$@"; do
      i="${i//\\/\\\\}"
      args="${args} \"${i//\"/\\\"}\""
    done

    if [ "${args}" == "" ]; then args="/bin/bash"; fi

    if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

    export SINGULARITY_BINDPATH=/home,/scratch,/share/apps

    singularity exec ${nv} \
    /scratch/hl5262/tritonserver_24.03-trtllm-python-py3.sif \
    /bin/bash -c "
    unset -f which
    ${args}
    "
    ```

    `sh ~/run-triton3.bash`

4. Pull code repos.
    Just pull This repo rename as `ml_proj` in your container. The TensorRT-LLM, tensorrtllm_backend and Llama-2-7b-chat-hf are submodule.

    ``` shell
    git clone --recursive git@github.com:haofrank/24ml_proj.git ml_proj

    # or copy /scratch/hl5262/ml_proj
    # mkdir ~/ml_proj && cp /scratch/hl5262/ml_proj/* ~/ml_proj
    ```

6. Install Python requirements.

    ``` shell
    pip install -r TensorRT-LLM/requirements.txt
    pip install -r TensorRT-LLM/examples/llama/requirements.txt
    pip install -r tensorrtllm_backend/requirements.txt
    pip3 install tensorrt_llm==0.8.0 --extra-index-url https://pypi.nvidia.com
    ```

### Run Triton Server (when you need)
1. copy the model_repo (*gpt_model*) that you need. **If** your `ml_proj` was copied, **DO NOT** need do this step.

    `cp -r /scratch/hl5262/tmp tmp`


2. update the path config.

  - we need update **tokenizer_dir** in these file using the `Llama-2-7b-chat-hf` absolute path:

    ```
      models/model_repo/llama_ifb/preprocessing/config.pbtxt
      models/model_repo/llama_ifb/postprocessing/config.pbtxt
      models/model_repo/llama_ifb/tensorrt_llm_bls/config.pbtxt
    ```

  - we need update **gpt_model_path** in this file using the `tmp` dir copyed in step 1:

    ```
      models/model_repo/llama_ifb/tensorrt_llm/config.pbtxt
    ```

3. Run.

    There is a `models` dir in this repo, these models have been coverd to TRT format. Your can run it to test it.


    `python3 tensorrtllm_backend/scripts/launch_triton_server.py --world_size 1 --model_repo=/home/{NetId}/ml_proj/models/model_repo/llama_ifb/`

4. Test.

    Here is an curl test case:

    ```
     curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "纽约大学怎么样?", "max_tokens": 200, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
    ```

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
  - 前端：guanshi
    - 视频播放
    - 字幕预加载
  - 后端：hao
    - 单机多卡
    - 服务暴露  
  - 测试&调研 jiayu，yihan
    - 效果
    - contex的记忆
    - 设计 system prompt
    - 时间
  - 炼丹：yihan
    - trt模型转化的参数
    - 模型启动（triton server）的参数
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

