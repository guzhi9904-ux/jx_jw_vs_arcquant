from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, Qwen2ForCausalLM
from datasets import load_dataset
import torch.nn as nn
import gc
import torch
from collections import defaultdict
import functools
from typing import List
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import torch.nn.functional as F
from sklearn.cluster import KMeans
import sys
import os
from pathlib import Path
from model.quantize import *
from model.kv_cache import *


@torch.no_grad()
def get_reorder_index(model, act_scales, metric='mean'):
    """为每个 Linear 输入生成通道置换。

    ``act_scales`` 的 key 形如 ``layers.0.self_attn.q_proj.input``。排序采用
    升序，所以低重要性通道在前、高重要性（ARC 候选）通道在后。返回值
    中每个一维张量都必须是 ``[0, K)`` 的完整排列，而不是 top-k 索引。

    注意：当前有效代码直接排序 ``act_scales``，``metric`` 只保留作接口
    兼容；注释掉的分支曾考虑把权重范数也纳入重要性。
    """
    act_orders = {}
    def is_permutation(x: torch.Tensor) -> bool:
        if not torch.is_tensor(x) or x.dim() != 1:
            return False
            
        if x.dtype.is_floating_point:
            return False
    
        n = len(x)
    
        if n == 0:
            return True
    
        expected = torch.arange(n, device=x.device, dtype=x.dtype)
        
        return torch.equal(torch.sort(x).values, expected)
    def reorder_tensor(tensor):
        # assert dimension == 1
        assert tensor.dim() == 1, "Choosing outliers must be 1 dimensional"
        # 升序排列使最重要/离群的通道落在最后 select_num 个位置。
        sorted_tensor, sorted_index = torch.sort(tensor, descending=False) # For putting outliers at last
        # _, sorted_index = torch.sort(tensor, descending=True) # For putting outliers at first
        assert is_permutation(sorted_index)
        return sorted_index
        # return torch.arange(tensor.shape[0])
        
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            m.name = name
            # Reorder Index of each layer's input
            # Used to reorder the weight and previous layer's output
            inputName = name + ".input"
            # act_orders[inputName] = reorder_tensor(act_scales[inputName])
            # if metric == 'frobenius': 
            #     importance = torch.linalg.norm(m.weight.data, ord=2, dim=0) * act_scales[inputName]
            # else: 
            #     importance = act_scales[inputName]
            act_orders[inputName] = reorder_tensor(act_scales[inputName])
            # act_orders[inputName] = reorder_tensor(importance)

            assert act_orders[inputName].dim() == 1, "Return Index must be 1 dimensional"

    return act_orders



def load_model(model_path):
    """加载校准用的原始 HF 模型和 tokenizer，并关闭生成时 KV cache。"""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    kwargs = {"torch_dtype": "auto", "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, **kwargs)
    model.eval()
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
    return model, enc



@torch.no_grad()
def get_act_stats(model, dataloader, device_, metric='mean', seqlen=2048, reorder_index=None):
    """收集每个 Linear 输入/输出的逐通道统计量。

    参数：
      model: 尚未量化的 Hugging Face CausalLM。
      dataloader: ``nsamples`` 条定长 token 样本。
      device_: 校准时逐层搬运到的设备。
      metric: ``hessian`` 使用二阶矩对角；``score`` 使用 NVFP4 残差 L2；
        其他值（CLI 暴露的 mean/frobenius/max）当前都使用绝对值的无穷范数。
      seqlen: Catcher 为每条样本预分配的序列长度。

    为控制显存，先截获第一层输入，再一次只把一个 DecoderLayer 放到 GPU。
    forward hook 在每个 Linear 处累计统计，最后返回 CPU 上的字典。
    """
    nsamples = len(dataloader)
    device = device_
    act_scales = {}

    def stat_tensor(name, tensor, weight=None, reorder_index=None):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()

        if metric == 'hessian':
            # 对角项与每个输入通道的平方和成正比。
            tensorH = math.sqrt(2 / nsamples) * tensor.float().t()
            comming_H = tensorH.matmul(tensorH.t())
            comming_scales = torch.diag(comming_H)
        elif metric == 'score':
            # 直接用该通道在 NVFP4 fake quant 下的残差能量排序。
            if reorder_index is not None:
                tensor = torch.index_select(tensor, 1, reorder_index)
                    
            tensorE = tensor - quantize_nvfp4_tensor(tensor, group_size=16)
            # if weight is not None:
            #     if reorder_index is not None:
            #         weight = torch.index_select(weight.to(tensor.device, non_blocking=True), 1, reorder_index)
            #     weight_norm = torch.linalg.norm(weight.to(tensor.device, non_blocking=True), ord=2, dim=0).float()
            #     tensor_norm = torch.linalg.norm(tensorE, ord=2, dim=0).float()
            #     comming_scales = (tensor_norm * weight_norm).cpu()
            # else:
            comming_scales = torch.linalg.norm(tensorE, ord=2, dim=0).float().cpu()
        else:
            # comming_scales = torch.mean(tensor.abs(), dim=0).float().cpu()
            # 当前默认路径：跨 token/样本取每个通道的最大绝对激活。
            comming_scales = torch.linalg.norm(tensor.abs(), ord=float('inf'), dim=0).float().cpu()

        if name in act_scales:
            if metric == 'hessian':
                act_scales[name] += comming_scales
            else:
                act_scales[name] = torch.max(act_scales[name], comming_scales)
        else:
            act_scales[name] = comming_scales

    def stat_input_hook(m, x, y, name, weight_for_input_stat=None, reorder_index=None):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.Tensor)
        if isinstance(y, tuple):
            y = y[0]
            assert isinstance(y, torch.Tensor)

        inputName = name + ".input"
        outputName = name + ".output"
        if reorder_index is not None:
            # stat_tensor(inputName, x[:, reorder_index[inputName].to(torch.int32)], weight=weight_for_input_stat[:, reorder_index[inputName].to(torch.int32)])
            stat_tensor(inputName, x, weight=weight_for_input_stat, reorder_index=reorder_index)
        else:
            stat_tensor(inputName, x, weight=weight_for_input_stat)
        stat_tensor(outputName, y)

    # q/k/v 共享同一输入，因此统计时组合权重的逻辑仅服务于已注释的
    # frobenius 方案；当前默认 max 方案实际只读取输入激活。
    hooks = []
    nameTemplate = 'layers.{}.{}.{}.{}'
    
    for layer_idx, layer in enumerate(model.model.layers):
        

        attn_block = layer.self_attn
        
        # The default ARC calibration metric only reads activations.  Keeping a
        # GPU copy of every layer's concatenated weights alive through hook
        # closures made 7/8B calibration needlessly consume many extra GiB.
        qkv_weight_combined = (
            torch.cat([
                attn_block.q_proj.weight.data,
                attn_block.k_proj.weight.data,
                attn_block.v_proj.weight.data,
            ], dim=0).to(device=device, non_blocking=True)
            if metric == 'frobenius'
            else None
        )
        
        for proj_name, proj_module in [('q_proj', attn_block.q_proj), ('k_proj', attn_block.k_proj), ('v_proj', attn_block.v_proj)]:
            name = f'layers.{layer_idx}.self_attn.{proj_name}'
            index_key = nameTemplate.format(layer_idx, 'self_attn', proj_name, 'input')
            index = reorder_index[index_key].to(device=device, dtype=torch.int32) if (reorder_index is not None and index_key in reorder_index) else None
            
            hooks.append(
                proj_module.register_forward_hook(
                    functools.partial(stat_input_hook, name=name, weight_for_input_stat=qkv_weight_combined, reorder_index=index)
                )
            )
            
        o_proj_name = f'layers.{layer_idx}.self_attn.o_proj'
        o_proj_weight_for_hook = attn_block.o_proj.weight.data if 'o_proj' in o_proj_name and metric == 'frobenius' else None
        
        index_key = nameTemplate.format(layer_idx, 'self_attn', 'o_proj', 'input')
        index = reorder_index[index_key].to(device=device, dtype=torch.int32) if (reorder_index is not None and index_key in reorder_index) else None
        
        hooks.append(
            attn_block.o_proj.register_forward_hook(
                functools.partial(stat_input_hook, name=o_proj_name, weight_for_input_stat=o_proj_weight_for_hook, reorder_index=index)
            )
        )
        
        
        if hasattr(layer, 'block_sparse_moe'):
            moe_block = layer.block_sparse_moe
            
            gate_layer = moe_block.gate
            gate_name = f'layers.{layer_idx}.block_sparse_moe.gate'
            
            index_key = f"{gate_name}.input" 
            index = reorder_index[index_key].to(device=device, dtype=torch.int32) if (reorder_index is not None and index_key in reorder_index) else None
    
            hooks.append(
                gate_layer.register_forward_hook(
                    functools.partial(stat_input_hook, name=gate_name, weight_for_input_stat=gate_layer.weight.data, reorder_index=index)
                )
            )
    
            for expert_idx, expert in enumerate(moe_block.experts):

                gate_up_weight_combined = (
                    torch.cat([
                        expert.w1.weight.data,
                        expert.w3.weight.data,
                    ], dim=0).to(device=device, non_blocking=True)
                    if metric == 'frobenius'
                    else None
                )
                
                for proj_name, proj_module in [('w1', expert.w1), ('w3', expert.w3)]:
                    name = f'layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{proj_name}'
                    
                    index_key = f"{name}.input"
                    index = reorder_index[index_key].to(device=device, dtype=torch.int32) if (reorder_index is not None and index_key in reorder_index) else None
    
                    hooks.append(
                        proj_module.register_forward_hook(
                            functools.partial(stat_input_hook, name=name, weight_for_input_stat=gate_up_weight_combined, reorder_index=index)
                        )
                    )
    
                down_proj_name = f'layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2'
                down_proj_weight_for_hook = expert.w2.weight.data if metric == 'frobenius' else None
                
                index_key = f"{down_proj_name}.input"
                index = reorder_index[index_key].to(device=device, dtype=torch.int32) if (reorder_index is not None and index_key in reorder_index) else None
    
                hooks.append(
                    expert.w2.register_forward_hook(
                        functools.partial(stat_input_hook, name=down_proj_name, weight_for_input_stat=down_proj_weight_for_hook, reorder_index=index)
                    )
                )
    
        elif hasattr(layer, 'mlp'):
            mlp_block = layer.mlp
            
            gate_up_weight_combined = (
                torch.cat([
                    mlp_block.gate_proj.weight.data,
                    mlp_block.up_proj.weight.data,
                ], dim=0).to(device=device, non_blocking=True)
                if metric == 'frobenius'
                else None
            )
            
            for proj_name, proj_module in [('gate_proj', mlp_block.gate_proj), ('up_proj', mlp_block.up_proj)]:
                name = f'layers.{layer_idx}.mlp.{proj_name}'
                index_key = nameTemplate.format(layer_idx, 'mlp', proj_name, 'input')
                index = reorder_index[index_key].to(device=device, dtype=torch.int32) if (reorder_index is not None and index_key in reorder_index) else None
                
                hooks.append(
                    proj_module.register_forward_hook(
                        functools.partial(stat_input_hook, name=name, weight_for_input_stat=gate_up_weight_combined, reorder_index=index)
                    )
                )
            
            down_proj_name = f'layers.{layer_idx}.mlp.down_proj'
            down_proj_weight_for_hook = mlp_block.down_proj.weight.data if 'down_proj' in down_proj_name and metric == 'frobenius' else None
            
            index_key = nameTemplate.format(layer_idx, 'mlp', 'down_proj', 'input')
            index = reorder_index[index_key].to(device=device, dtype=torch.int32) if (reorder_index is not None and index_key in reorder_index) else None
            
            hooks.append(
                mlp_block.down_proj.register_forward_hook(
                    functools.partial(stat_input_hook, name=down_proj_name, weight_for_input_stat=down_proj_weight_for_hook, reorder_index=index)
                )
            )

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, 'norm') and not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        """截获 embedding 后的首层输入，并用异常提前终止完整 forward。"""
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            hidden_states = inp[0] if isinstance(inp, tuple) else inp
            inps[cache['i']] = hidden_states.squeeze(0)
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError  # 控制流信号，不表示校准失败。

    layers[0] = Catcher(layers[0])
    
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    assert cache['i'] == nsamples, "Captured samples should be equal to nsamples"
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, 'norm') and not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    # 顺序执行各层，并交换 inps/outs，使下一层拿到真实的上一层输出。
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps
        torch.cuda.empty_cache()
        gc.collect()

    for h in hooks:
        h.remove()

    return act_scales

    

def _load_wikitext2_split(split):
    """Load a cached Arrow split when ARCQUANT_WIKITEXT_CACHE_DIR is set."""
    cache_dir = os.environ.get("ARCQUANT_WIKITEXT_CACHE_DIR")
    if cache_dir:
        from datasets import Dataset

        arrow_path = Path(cache_dir) / f"wikitext-{split}.arrow"
        if not arrow_path.is_file():
            raise FileNotFoundError(
                f"Missing cached WikiText2 split: {arrow_path}"
            )
        print(f"Loading local WikiText2 {split} split from {arrow_path}")
        return Dataset.from_file(str(arrow_path))

    from datasets import load_dataset

    return load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = _load_wikitext2_split('train')
    testdata = _load_wikitext2_split('test')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
  
    import random
    random.seed(seed)
    trainloader = []
    inps = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
    return trainloader, inps 

def get_c4(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    import random
    import torch

    traindata = load_dataset(
        'allenai/c4', 'en', 
        split='validation', 
        trust_remote_code=True
    )
    
    random.seed(seed)
    trainloader = []
    inps = []
    
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]['text']
            
            encoded = tokenizer(text, return_tensors='pt')
            
            if encoded.input_ids.shape[1] >= seqlen:
                i = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
                inp = encoded.input_ids[:, i : i + seqlen]
                break
        
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
        
    return trainloader, inps

def get_pile(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    import random
    
    try:
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    except:
        print("Falling back to pile-10k")
        dataset = load_dataset("NeelNanda/pile-10k", split="train")

    dataset = dataset.shuffle(seed=seed)

    trainloader = []
    inps = []
    
    for data in dataset:
        if len(trainloader) == nsamples:
            break
            
        text = data['text']
        enc = tokenizer(text, return_tensors='pt')
        
        if enc.input_ids.shape[1] >= seqlen:
            i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = enc.input_ids[:, i:j]
            
            tar = inp.clone()
            tar[:, :-1] = -100 # Mask out context
            
            trainloader.append((inp, tar))
            inps.append(inp)
            
    return trainloader, inps

def get_humaneval(nsamples, seed, seqlen, tokenizer):
    import random
    
    try:
        from human_eval.data import read_problems
        problems = read_problems()  
        dataset = list(problems.values())
    except ImportError:
        print("=" * 80)
        print("run 'pip install humaneval'")
        print("=" * 80)
        return [], []
    except Exception as e:
        print(f" 'humaneval' loading error: {e}")
        return [], []

    text_corpus = "\n\n".join([sample['prompt'] for sample in dataset])
    trainenc = tokenizer(text_corpus, return_tensors='pt')

    random.seed(seed)
    trainloader = []
    inps = []
    for _ in range(nsamples):
        if trainenc.input_ids.shape[1] <= seqlen:
            print(f"warning: HumanEval total length ({trainenc.input_ids.shape[1]}) <= seqlen ({seqlen}).")
            inp = trainenc.input_ids
        else:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]

        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
        
        if trainenc.input_ids.shape[1] <= seqlen:
            break 

    return trainloader, inps


@torch.no_grad()
def search_select_proportions(model, dataloader, device_, seqlen, reorder_index):
    """估计每个 Linear 需要追加的 ARC 通道数 ``select_num``。

    先按 ``reorder_index`` 排列输入，再统计大于“每行最大值的 1/8”的元素
    比例。该比例乘输入维度后向上对齐到 64，得到 CUDA kernel 的 ``KE``。
    估算位宽按 NVFP4 每元素 4.5 bit 计算：4.5 * (K + KE) / K。

    这里使用 32 条预处理样本（由调用方决定），与第一阶段的 ``samples``
    数量不同；返回字典的 key 同样以 ``.input`` 结尾。
    """
    nsamples = len(dataloader)
    device = device_
    
    select_nums = {}
    average_bits = {}
    
    print("Preparing inputs...")
    layers = model.model.layers
    
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    
    dtype = next(iter(model.parameters())).dtype
    
    cache = {'attention_mask': None, 'position_ids': None}
    class Catcher(nn.Module):
        """一次性截获整批首层输入，供后续逐层搜索使用。"""
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            cache['inps'] = inp
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError  # 控制流信号：拿到 hidden states 后停止整模型 forward。
            
    layers[0] = Catcher(layers[0])
    
    if isinstance(dataloader, list):
         dataloader = torch.stack(dataloader, dim=0).squeeze(1)
    
    try:
        model(dataloader.to(device))
    except ValueError:
        pass 
    
    layers[0] = layers[0].module
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
        
    torch.cuda.empty_cache()

    inps = cache['inps']
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    total_elements = 0
    total_bits = 0

    def stat_input_hook(m, x, y, name, act_scales_dict):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(y, tuple):
            y = y[0]
        act_scales_dict[name + ".input"] = x 
        # act_scales_dict[name + ".output"] = y 

    print("Processing layers...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        layer = layer.to(device) 
        
        act_scales = {} 
        hooks = []
        
        layer_prefix = f"layers.{i}"
        
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                full_name = f"{layer_prefix}.{name}"
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=full_name, act_scales_dict=act_scales)
                    )
                )

        inps = inps.to(device)
        if attention_mask is not None: attention_mask = attention_mask.to(device)
        if position_ids is not None: position_ids = position_ids.to(device)

        with torch.no_grad():
            inps = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0]

        for name, keys in act_scales.items():
            if 'output' in name:
                continue
            
            keys = keys.reshape(-1, keys.shape[-1]).contiguous()
            seqlen_dim, in_features = keys.shape
            
            if name in reorder_index:
                idx = reorder_index[name].to(device).to(torch.int32) 
                keys = keys[:, idx]
            else:
                print(f"Warning: {name} not found in reorder_index")
                continue

            # 论文实现中的经验阈值：逐 token 最大激活的 1/8。
            threshold = keys.max(dim=-1, keepdim=True)[0] * 0.125
            select_ratio = (keys > threshold).sum() / keys.numel()
            # 64 对齐既满足后端 kernel 布局，也避免每层使用任意细粒度 KE。
            select_num = math.ceil(in_features * select_ratio / 64) * 64
            
            if select_num > in_features: select_num = in_features
            
            select_ratio_val = select_num / in_features
            # ARC 把内积维度从 K 扩成 K+KE；4.5 同时近似计入 FP4 值和 scale。
            avg_bits = 4.5 * (in_features + select_num) / in_features
            
            average_bits[name] = avg_bits
            select_nums[name] = select_num
            
            total_elements += in_features
            total_bits += 4.5 * (in_features + select_num)
            
            print(f'{name}: {select_ratio_val*100:.2f}%, avg:{avg_bits:.2f}')
            
            del keys 

        for h in hooks:
            h.remove()
        
        del act_scales
        del hooks
        
        layer = layer.cpu() 
        gc.collect()
        torch.cuda.empty_cache()

    print(f'Average bits is {(total_bits / total_elements):.2f}')
    return select_nums, average_bits

