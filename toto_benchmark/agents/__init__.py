def init_agent_from_config(config, device='cpu', normalization=None):
    """ Create an agent according to the config """

    agent_type = config.agent.type
    if agent_type in ['bc', 'bcimage_pre']:
        from .BCAgent import _init_agent_from_config
        return _init_agent_from_config(config, device, normalization)
    
    elif agent_type == 'bcimage':
        from .BCImageAgent import _init_agent_from_config
        return _init_agent_from_config(config, device, normalization)
    
    elif agent_type == 'knn_image':
        from .KNNImageAgent import _init_agent_from_config
        return _init_agent_from_config(config, device)
    
    elif agent_type == 'd3rlpy':
        from .D3Agent import _init_agent_from_config
        return _init_agent_from_config(config, device)

    elif agent_type == 'Diffusion_policy_unet':
        from .Diffusion_Agent_Unet import _init_agent_from_config
        return _init_agent_from_config(config, device, normalization)
    
    elif agent_type == 'Diffusion_policy_transformer':
        from .Diffusion_Agent_Transformer import _init_agent_from_config
        return _init_agent_from_config(config, device, normalization)

    assert f"[ERROR] Unknown agent type {agent_type}"
