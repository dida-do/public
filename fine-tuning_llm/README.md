# Fine Tuning LLM with DPO

The training script to fine tuning an LLM with DPO.


To run the training script you need to install the requirements first. You also need a GPU that does support `flash_attention_2`, otherwise you can set

```python
attn_implementation=None
```

To learn more about Direct Preference Optimization visit the our [blog](https://dida.do/blog).

You can try the code with the following command

```
accelerate launch --num_processes 1 train-dpo.py
```

if you want to use multiple gpus for the training replace the number accordingly. If you have a pretrained model locally, and do not want to downlaod the weights again, you can replace 

```python
model_cfg = ModelConfig(
    model_name_or_path=<PATH_TO_YOUR_WEIGHTs>,
```
If you have memory problems you can try to reduce the rank of lora by reducing `lora_r`. The current version of the code does support training with ZeRO Data Parallelism, so you can also try out `ZeRo Stage 2` in combination with offloading parameters to CPUs. `ZeRo Stage 3` does not support 4 bit training yet.
