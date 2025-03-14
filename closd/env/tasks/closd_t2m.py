# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from closd.env.tasks import closd_task
from isaacgym.torch_utils import *
from closd.utils.closd_util import STATES
import random

class CLoSDT2M(closd_task.CLoSDTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.init_state = STATES.TEXT2MOTION
        self.hml_data_buf_size = max(self.fake_mdm_args.context_len, self.planning_horizon_20fps)
        self.hml_prefix_from_data = torch.zeros([self.num_envs, 263, 1, self.hml_data_buf_size], dtype=torch.float32, device=self.device)
        
        # 添加自定义提示词支持
        self.custom_prompts = None
        print("Debug: Checking custom prompts configuration...")
        print(f"Debug: cfg.env.test = {self.cfg.env.test}")
        print(f"Debug: hasattr(cfg.env, 'custom_prompts') = {hasattr(cfg.env, 'custom_prompts')}")
        if hasattr(cfg.env, 'custom_prompts'):
            print(f"Debug: cfg.env.custom_prompts = {cfg.env.custom_prompts}")
        
        if hasattr(cfg.env, 'custom_prompts') and cfg.env.custom_prompts is not None:
            self.custom_prompts = cfg.env.custom_prompts
            print(f"Debug: Loaded custom prompts: {self.custom_prompts}")
        return
    
    def update_mdm_conditions(self, env_ids):  
        super().update_mdm_conditions(env_ids)
        
        # 获取gt_motion，无论是自定义提示词还是数据集模式都需要
        try:
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        except StopIteration:
            del self.mdm_data_iter
            self.mdm_data_iter = iter(self.mdm_data)
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        
        # 如果启用了自定义提示词模式
        if self.custom_prompts is not None and self.cfg.env.test:
            print("Debug: Using custom prompts mode")
            for i in env_ids:
                # 从自定义提示词列表中随机选择一个
                prompt = random.choice(self.custom_prompts)
                self.hml_prompts[int(i)] = prompt
                # 使用默认长度，因为这是测试模式
                self.hml_lengths[int(i)] = 40
                # 使用默认token，因为这是测试模式
                self.hml_tokens[int(i)] = torch.zeros(1, dtype=torch.long, device=self.device)
                self.db_keys[int(i)] = "custom"
                print(f'Environment {i}: Custom Prompt = {prompt}')
        else:
            print("Debug: Using dataset prompts mode")
            for i in env_ids:
                self.hml_prompts[int(i)] = model_kwargs['y']['text'][int(i)]
                self.hml_lengths[int(i)] = model_kwargs['y']['lengths'][int(i)]  
                self.hml_tokens[int(i)] = model_kwargs['y']['tokens'][int(i)]  
                self.db_keys[int(i)] = model_kwargs['y']['db_key'][int(i)]  
                print(f'Environment {i}: Dataset Prompt = {self.hml_prompts[int(i)]}')
        
        self.hml_prefix_from_data[env_ids] = gt_motion[..., :self.hml_data_buf_size].to(self.device)[env_ids]
        if self.cfg['env']['dip']['debug_hml']:
            print(f'in update_mdm_conditions: 1st 10 env_ids={env_ids[:10].cpu().numpy()}, prompts={self.hml_prompts[:2]}')
        return
    
    def get_cur_done(self):
        # Done signal is not in use for this task
        return torch.zeros([self.num_envs], device=self.device, dtype=bool)
    

