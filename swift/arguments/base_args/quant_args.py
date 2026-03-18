# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from dataclasses import dataclass
from typing import Literal, Optional

from swift.model import get_model_processor
from swift.utils import HfConfigFactory, get_modules_to_not_convert


@dataclass
class QuantizeArguments:
    """A dataclass that holds the configuration for model quantization.

    Args:
        quant_method (Optional[str]): The quantization method to use when loading the model. Can be one of {'bnb',
            'hqq', 'eetq', 'quanto', 'fp8'}. Note: This is not required for QLoRA training on pre-quantized AWQ/GPTQ
            models. Defaults to None.
        quant_bits (Optional[Union[int, str]]): The number of bits for quantization, e.g., {1, 2, 3, 4, 8, 'float8'}.
            Defaults to None.
        hqq_axis (Optional[int]): The quantization axis for HQQ quantization. Defaults to None.
        bnb_4bit_compute_dtype (Optional[str]): The compute data type for 4-bit BNB quantization. Can be one of {
            'float16', 'bfloat16', 'float32'}. Defaults to None, which will use the model's `torch_dtype`.
        bnb_4bit_quant_type (str): The quantization type for 4-bit BNB quantization. Can be one of {'fp4', 'nf4'}.
            Defaults to 'nf4'.
        bnb_4bit_use_double_quant (bool): Whether to use double quantization for 4-bit BNB quantization.
            Defaults to True.
        bnb_4bit_quant_storage (Optional[str]): The storage type for packing quantized 4-bit parameters in BNB.
            Defaults to None.
    """
    # awq, gptq, and aqlm need to be pre-quantized models.
    #   It can be detected automatically, without the need to pass in.
    # while bnb, hqq, and eetq can be quantized during SFT using the original models.
    quant_method: Literal['bnb', 'hqq', 'eetq', 'quanto', 'fp8'] = None
    # bnb: 4,8; hqq: 1,2,3,4,8'; eetq: 8
    # awq: 4; gptq: 2,3,4,8
    quant_bits: Literal[1, 2, 3, 4, 8, 'float8'] = None
    # hqq
    hqq_axis: Optional[int] = None
    # bnb
    bnb_4bit_compute_dtype: Literal['float16', 'bfloat16', 'float32', None] = None
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: Optional[str] = None

    def get_quantization_config(self):
        if self.quant_method is None or self.quant_method in {'awq', 'gptq', 'gptq_v2'}:
            return None
        assert self.quant_method in {'bnb', 'hqq', 'eetq', 'quanto', 'fp8'}
        if self.quant_method != 'fp8' and self.quant_bits is None:
            raise ValueError(f'Please set the quant_bits. args.quant_bits: {self.quant_bits}')
        if self.quant_method == 'bnb':
            if self.quant_bits == 4:
                load_in_4bit, load_in_8bit = True, False
            elif self.quant_bits == 8:
                load_in_4bit, load_in_8bit = False, True
            else:
                raise ValueError(f'bnb not support quant_bits: {self.quant_bits}')

            from transformers import BitsAndBytesConfig
            llm_int8_skip_modules = self.get_modules_to_not_convert()
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_quant_storage=self.bnb_4bit_quant_storage,
                llm_int8_skip_modules=llm_int8_skip_modules)
        elif self.quant_method == 'fp8':
            if not hasattr(self, 'model_info'):
                return
            from transformers import FineGrainedFP8Config
            with torch.device('meta'):
                hf_model, _ = get_model_processor(self.model_dir, model_type=self.model_type, return_dummy_model=True)
            modules_to_not_convert = get_modules_to_not_convert(hf_model)
            modules_to_not_convert_tmp = [
                "lm_head",
                "model.language_model.embed_tokens",
                "model.language_model.layers.0.linear_attn.conv1d",
                "model.language_model.layers.0.linear_attn.in_proj_a",
                "model.language_model.layers.0.linear_attn.in_proj_b",
                "model.language_model.layers.1.linear_attn.conv1d",
                "model.language_model.layers.1.linear_attn.in_proj_a",
                "model.language_model.layers.1.linear_attn.in_proj_b",
                "model.language_model.layers.10.linear_attn.conv1d",
                "model.language_model.layers.10.linear_attn.in_proj_a",
                "model.language_model.layers.10.linear_attn.in_proj_b",
                "model.language_model.layers.12.linear_attn.conv1d",
                "model.language_model.layers.12.linear_attn.in_proj_a",
                "model.language_model.layers.12.linear_attn.in_proj_b",
                "model.language_model.layers.13.linear_attn.conv1d",
                "model.language_model.layers.13.linear_attn.in_proj_a",
                "model.language_model.layers.13.linear_attn.in_proj_b",
                "model.language_model.layers.14.linear_attn.conv1d",
                "model.language_model.layers.14.linear_attn.in_proj_a",
                "model.language_model.layers.14.linear_attn.in_proj_b",
                "model.language_model.layers.16.linear_attn.conv1d",
                "model.language_model.layers.16.linear_attn.in_proj_a",
                "model.language_model.layers.16.linear_attn.in_proj_b",
                "model.language_model.layers.17.linear_attn.conv1d",
                "model.language_model.layers.17.linear_attn.in_proj_a",
                "model.language_model.layers.17.linear_attn.in_proj_b",
                "model.language_model.layers.18.linear_attn.conv1d",
                "model.language_model.layers.18.linear_attn.in_proj_a",
                "model.language_model.layers.18.linear_attn.in_proj_b",
                "model.language_model.layers.2.linear_attn.conv1d",
                "model.language_model.layers.2.linear_attn.in_proj_a",
                "model.language_model.layers.2.linear_attn.in_proj_b",
                "model.language_model.layers.20.linear_attn.conv1d",
                "model.language_model.layers.20.linear_attn.in_proj_a",
                "model.language_model.layers.20.linear_attn.in_proj_b",
                "model.language_model.layers.21.linear_attn.conv1d",
                "model.language_model.layers.21.linear_attn.in_proj_a",
                "model.language_model.layers.21.linear_attn.in_proj_b",
                "model.language_model.layers.22.linear_attn.conv1d",
                "model.language_model.layers.22.linear_attn.in_proj_a",
                "model.language_model.layers.22.linear_attn.in_proj_b",
                "model.language_model.layers.24.linear_attn.conv1d",
                "model.language_model.layers.24.linear_attn.in_proj_a",
                "model.language_model.layers.24.linear_attn.in_proj_b",
                "model.language_model.layers.25.linear_attn.conv1d",
                "model.language_model.layers.25.linear_attn.in_proj_a",
                "model.language_model.layers.25.linear_attn.in_proj_b",
                "model.language_model.layers.26.linear_attn.conv1d",
                "model.language_model.layers.26.linear_attn.in_proj_a",
                "model.language_model.layers.26.linear_attn.in_proj_b",
                "model.language_model.layers.28.linear_attn.conv1d",
                "model.language_model.layers.28.linear_attn.in_proj_a",
                "model.language_model.layers.28.linear_attn.in_proj_b",
                "model.language_model.layers.29.linear_attn.conv1d",
                "model.language_model.layers.29.linear_attn.in_proj_a",
                "model.language_model.layers.29.linear_attn.in_proj_b",
                "model.language_model.layers.30.linear_attn.conv1d",
                "model.language_model.layers.30.linear_attn.in_proj_a",
                "model.language_model.layers.30.linear_attn.in_proj_b",
                "model.language_model.layers.32.linear_attn.conv1d",
                "model.language_model.layers.32.linear_attn.in_proj_a",
                "model.language_model.layers.32.linear_attn.in_proj_b",
                "model.language_model.layers.33.linear_attn.conv1d",
                "model.language_model.layers.33.linear_attn.in_proj_a",
                "model.language_model.layers.33.linear_attn.in_proj_b",
                "model.language_model.layers.34.linear_attn.conv1d",
                "model.language_model.layers.34.linear_attn.in_proj_a",
                "model.language_model.layers.34.linear_attn.in_proj_b",
                "model.language_model.layers.36.linear_attn.conv1d",
                "model.language_model.layers.36.linear_attn.in_proj_a",
                "model.language_model.layers.36.linear_attn.in_proj_b",
                "model.language_model.layers.37.linear_attn.conv1d",
                "model.language_model.layers.37.linear_attn.in_proj_a",
                "model.language_model.layers.37.linear_attn.in_proj_b",
                "model.language_model.layers.38.linear_attn.conv1d",
                "model.language_model.layers.38.linear_attn.in_proj_a",
                "model.language_model.layers.38.linear_attn.in_proj_b",
                "model.language_model.layers.4.linear_attn.conv1d",
                "model.language_model.layers.4.linear_attn.in_proj_a",
                "model.language_model.layers.4.linear_attn.in_proj_b",
                "model.language_model.layers.40.linear_attn.conv1d",
                "model.language_model.layers.40.linear_attn.in_proj_a",
                "model.language_model.layers.40.linear_attn.in_proj_b",
                "model.language_model.layers.41.linear_attn.conv1d",
                "model.language_model.layers.41.linear_attn.in_proj_a",
                "model.language_model.layers.41.linear_attn.in_proj_b",
                "model.language_model.layers.42.linear_attn.conv1d",
                "model.language_model.layers.42.linear_attn.in_proj_a",
                "model.language_model.layers.42.linear_attn.in_proj_b",
                "model.language_model.layers.44.linear_attn.conv1d",
                "model.language_model.layers.44.linear_attn.in_proj_a",
                "model.language_model.layers.44.linear_attn.in_proj_b",
                "model.language_model.layers.45.linear_attn.conv1d",
                "model.language_model.layers.45.linear_attn.in_proj_a",
                "model.language_model.layers.45.linear_attn.in_proj_b",
                "model.language_model.layers.46.linear_attn.conv1d",
                "model.language_model.layers.46.linear_attn.in_proj_a",
                "model.language_model.layers.46.linear_attn.in_proj_b",
                "model.language_model.layers.48.linear_attn.conv1d",
                "model.language_model.layers.48.linear_attn.in_proj_a",
                "model.language_model.layers.48.linear_attn.in_proj_b",
                "model.language_model.layers.49.linear_attn.conv1d",
                "model.language_model.layers.49.linear_attn.in_proj_a",
                "model.language_model.layers.49.linear_attn.in_proj_b",
                "model.language_model.layers.5.linear_attn.conv1d",
                "model.language_model.layers.5.linear_attn.in_proj_a",
                "model.language_model.layers.5.linear_attn.in_proj_b",
                "model.language_model.layers.50.linear_attn.conv1d",
                "model.language_model.layers.50.linear_attn.in_proj_a",
                "model.language_model.layers.50.linear_attn.in_proj_b",
                "model.language_model.layers.52.linear_attn.conv1d",
                "model.language_model.layers.52.linear_attn.in_proj_a",
                "model.language_model.layers.52.linear_attn.in_proj_b",
                "model.language_model.layers.53.linear_attn.conv1d",
                "model.language_model.layers.53.linear_attn.in_proj_a",
                "model.language_model.layers.53.linear_attn.in_proj_b",
                "model.language_model.layers.54.linear_attn.conv1d",
                "model.language_model.layers.54.linear_attn.in_proj_a",
                "model.language_model.layers.54.linear_attn.in_proj_b",
                "model.language_model.layers.56.linear_attn.conv1d",
                "model.language_model.layers.56.linear_attn.in_proj_a",
                "model.language_model.layers.56.linear_attn.in_proj_b",
                "model.language_model.layers.57.linear_attn.conv1d",
                "model.language_model.layers.57.linear_attn.in_proj_a",
                "model.language_model.layers.57.linear_attn.in_proj_b",
                "model.language_model.layers.58.linear_attn.conv1d",
                "model.language_model.layers.58.linear_attn.in_proj_a",
                "model.language_model.layers.58.linear_attn.in_proj_b",
                "model.language_model.layers.6.linear_attn.conv1d",
                "model.language_model.layers.6.linear_attn.in_proj_a",
                "model.language_model.layers.6.linear_attn.in_proj_b",
                "model.language_model.layers.60.linear_attn.conv1d",
                "model.language_model.layers.60.linear_attn.in_proj_a",
                "model.language_model.layers.60.linear_attn.in_proj_b",
                "model.language_model.layers.61.linear_attn.conv1d",
                "model.language_model.layers.61.linear_attn.in_proj_a",
                "model.language_model.layers.61.linear_attn.in_proj_b",
                "model.language_model.layers.62.linear_attn.conv1d",
                "model.language_model.layers.62.linear_attn.in_proj_a",
                "model.language_model.layers.62.linear_attn.in_proj_b",
                "model.language_model.layers.8.linear_attn.conv1d",
                "model.language_model.layers.8.linear_attn.in_proj_a",
                "model.language_model.layers.8.linear_attn.in_proj_b",
                "model.language_model.layers.9.linear_attn.conv1d",
                "model.language_model.layers.9.linear_attn.in_proj_a",
                "model.language_model.layers.9.linear_attn.in_proj_b",
                "model.visual.blocks.0.attn.proj",
                "model.visual.blocks.0.attn.qkv",
                "model.visual.blocks.0.mlp.linear_fc1",
                "model.visual.blocks.0.mlp.linear_fc2",
                "model.visual.blocks.1.attn.proj",
                "model.visual.blocks.1.attn.qkv",
                "model.visual.blocks.1.mlp.linear_fc1",
                "model.visual.blocks.1.mlp.linear_fc2",
                "model.visual.blocks.10.attn.proj",
                "model.visual.blocks.10.attn.qkv",
                "model.visual.blocks.10.mlp.linear_fc1",
                "model.visual.blocks.10.mlp.linear_fc2",
                "model.visual.blocks.11.attn.proj",
                "model.visual.blocks.11.attn.qkv",
                "model.visual.blocks.11.mlp.linear_fc1",
                "model.visual.blocks.11.mlp.linear_fc2",
                "model.visual.blocks.12.attn.proj",
                "model.visual.blocks.12.attn.qkv",
                "model.visual.blocks.12.mlp.linear_fc1",
                "model.visual.blocks.12.mlp.linear_fc2",
                "model.visual.blocks.13.attn.proj",
                "model.visual.blocks.13.attn.qkv",
                "model.visual.blocks.13.mlp.linear_fc1",
                "model.visual.blocks.13.mlp.linear_fc2",
                "model.visual.blocks.14.attn.proj",
                "model.visual.blocks.14.attn.qkv",
                "model.visual.blocks.14.mlp.linear_fc1",
                "model.visual.blocks.14.mlp.linear_fc2",
                "model.visual.blocks.15.attn.proj",
                "model.visual.blocks.15.attn.qkv",
                "model.visual.blocks.15.mlp.linear_fc1",
                "model.visual.blocks.15.mlp.linear_fc2",
                "model.visual.blocks.16.attn.proj",
                "model.visual.blocks.16.attn.qkv",
                "model.visual.blocks.16.mlp.linear_fc1",
                "model.visual.blocks.16.mlp.linear_fc2",
                "model.visual.blocks.17.attn.proj",
                "model.visual.blocks.17.attn.qkv",
                "model.visual.blocks.17.mlp.linear_fc1",
                "model.visual.blocks.17.mlp.linear_fc2",
                "model.visual.blocks.18.attn.proj",
                "model.visual.blocks.18.attn.qkv",
                "model.visual.blocks.18.mlp.linear_fc1",
                "model.visual.blocks.18.mlp.linear_fc2",
                "model.visual.blocks.19.attn.proj",
                "model.visual.blocks.19.attn.qkv",
                "model.visual.blocks.19.mlp.linear_fc1",
                "model.visual.blocks.19.mlp.linear_fc2",
                "model.visual.blocks.2.attn.proj",
                "model.visual.blocks.2.attn.qkv",
                "model.visual.blocks.2.mlp.linear_fc1",
                "model.visual.blocks.2.mlp.linear_fc2",
                "model.visual.blocks.20.attn.proj",
                "model.visual.blocks.20.attn.qkv",
                "model.visual.blocks.20.mlp.linear_fc1",
                "model.visual.blocks.20.mlp.linear_fc2",
                "model.visual.blocks.21.attn.proj",
                "model.visual.blocks.21.attn.qkv",
                "model.visual.blocks.21.mlp.linear_fc1",
                "model.visual.blocks.21.mlp.linear_fc2",
                "model.visual.blocks.22.attn.proj",
                "model.visual.blocks.22.attn.qkv",
                "model.visual.blocks.22.mlp.linear_fc1",
                "model.visual.blocks.22.mlp.linear_fc2",
                "model.visual.blocks.23.attn.proj",
                "model.visual.blocks.23.attn.qkv",
                "model.visual.blocks.23.mlp.linear_fc1",
                "model.visual.blocks.23.mlp.linear_fc2",
                "model.visual.blocks.24.attn.proj",
                "model.visual.blocks.24.attn.qkv",
                "model.visual.blocks.24.mlp.linear_fc1",
                "model.visual.blocks.24.mlp.linear_fc2",
                "model.visual.blocks.25.attn.proj",
                "model.visual.blocks.25.attn.qkv",
                "model.visual.blocks.25.mlp.linear_fc1",
                "model.visual.blocks.25.mlp.linear_fc2",
                "model.visual.blocks.26.attn.proj",
                "model.visual.blocks.26.attn.qkv",
                "model.visual.blocks.26.mlp.linear_fc1",
                "model.visual.blocks.26.mlp.linear_fc2",
                "model.visual.blocks.3.attn.proj",
                "model.visual.blocks.3.attn.qkv",
                "model.visual.blocks.3.mlp.linear_fc1",
                "model.visual.blocks.3.mlp.linear_fc2",
                "model.visual.blocks.4.attn.proj",
                "model.visual.blocks.4.attn.qkv",
                "model.visual.blocks.4.mlp.linear_fc1",
                "model.visual.blocks.4.mlp.linear_fc2",
                "model.visual.blocks.5.attn.proj",
                "model.visual.blocks.5.attn.qkv",
                "model.visual.blocks.5.mlp.linear_fc1",
                "model.visual.blocks.5.mlp.linear_fc2",
                "model.visual.blocks.6.attn.proj",
                "model.visual.blocks.6.attn.qkv",
                "model.visual.blocks.6.mlp.linear_fc1",
                "model.visual.blocks.6.mlp.linear_fc2",
                "model.visual.blocks.7.attn.proj",
                "model.visual.blocks.7.attn.qkv",
                "model.visual.blocks.7.mlp.linear_fc1",
                "model.visual.blocks.7.mlp.linear_fc2",
                "model.visual.blocks.8.attn.proj",
                "model.visual.blocks.8.attn.qkv",
                "model.visual.blocks.8.mlp.linear_fc1",
                "model.visual.blocks.8.mlp.linear_fc2",
                "model.visual.blocks.9.attn.proj",
                "model.visual.blocks.9.attn.qkv",
                "model.visual.blocks.9.mlp.linear_fc1",
                "model.visual.blocks.9.mlp.linear_fc2",
                "model.visual.merger.linear_fc1",
                "model.visual.merger.linear_fc2",
                "model.visual.patch_embed.proj",
                "model.visual.pos_embed",
                "mtp.fc"
            ]
            quantization_config = FineGrainedFP8Config(modules_to_not_convert=modules_to_not_convert_tmp)
        elif self.quant_method == 'hqq':
            from transformers import HqqConfig
            quantization_config = HqqConfig(nbits=self.quant_bits, axis=self.hqq_axis)
        elif self.quant_method == 'quanto':
            from transformers import QuantoConfig
            if self.quant_bits == 8:
                weights = 'int8'
            elif self.quant_bits == 'float8':
                weights = 'float8'
            elif self.quant_bits == 4:
                weights = 'int4'
            elif self.quant_bits == 2:
                weights = 'int2'
            else:
                raise ValueError('quanto quantization only support quant bits 2/4/8/float8')
            quantization_config = QuantoConfig(weights=weights)
        else:  # 'eetq'
            from transformers import EetqConfig
            quantization_config = EetqConfig(f'int{self.quant_bits}')

        return quantization_config

    def get_modules_to_not_convert(self):
        if not hasattr(self, 'model_meta') or not hasattr(self, 'model_info'):
            return None
        model_arch = self.model_meta.model_arch
        res = []
        if self.model_info.is_moe_model:
            res += ['mlp.gate', 'mlp.shared_expert_gate']
        if model_arch is not None:
            for key in ['vision_tower', 'aligner']:
                value = getattr(model_arch, key, None)
                if value:
                    res += value
        if not res:
            return None
        res.append('lm_head')
        return res

    def __post_init__(self):
        if self.bnb_4bit_compute_dtype is None:
            if self.torch_dtype in {torch.float16, torch.float32}:
                self.bnb_4bit_compute_dtype = torch.float32
            elif self.torch_dtype == torch.bfloat16:
                self.bnb_4bit_compute_dtype = torch.bfloat16
        self.bnb_4bit_compute_dtype: torch.dtype = HfConfigFactory.to_torch_dtype(self.bnb_4bit_compute_dtype)
