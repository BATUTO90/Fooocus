import torch
import ldm_patched.modules.model_management
import ldm_patched.modules.samplers
import ldm_patched.modules.conds
import ldm_patched.modules.utils
import math
import numpy as np

def prepare_noise(latent_image, seed, noise_inds=None):
    """Creates random noise given a latent image and a seed.
    Args:
        latent_image (torch.Tensor): The latent image.
        seed (int): The random seed.
        noise_inds (np.ndarray, optional): The noise indices. Defaults to None.
    Returns:
        torch.Tensor: The random noise.
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
    
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

def prepare_mask(noise_mask, shape, device):
    """Ensures noise mask is of proper dimensions.
    Args:
        noise_mask (torch.Tensor): The noise mask.
        shape (tuple): The shape of the noise mask.
        device (torch.device): The device to create the noise mask on.
    Returns:
        torch.Tensor: The prepared noise mask.
    """
    noise_mask = torch.nn.functional.interpolate(noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    noise_mask = ldm_patched.modules.utils.repeat_to_batch_size(noise_mask, shape[0])
    noise_mask = noise_mask.to(device)
    return noise_mask

def get_models_from_cond(cond, model_type):
    """Gets a list of models from a conditioning.
    Args:
        cond: The conditioning.
        model_type (str): The type of model to get.
    Returns:
        list: A list of models.
    """
    models = []
    for c in cond:
        if model_type in c:
            models += [c[model_type]]
    return models

def convert_cond(cond):
    """Converts a conditioning to a different format.
    Args:
        cond: The conditioning to convert.
    Returns:
        The converted conditioning.
    """
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = ldm_patched.modules.conds.CONDCrossAttn(c[0]) #TODO: remove
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        out.append(temp)
    return out

def get_additional_models(positive, negative, dtype):
    """Loads additional models in positive and negative conditioning.
    Args:
        positive: The positive conditioning.
        negative: The negative conditioning.
        dtype: The dtype of the models.
    Returns:
        A tuple of the additional models and the inference memory requirements.
    """
    control_nets = set(get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control"))

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = get_models_from_cond(positive, "gligen") + get_models_from_cond(negative, "gligen")
    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory

def cleanup_additional_models(models):
    """Cleans up additional models that were loaded.
    Args:
        models (list): A list of models to clean up.
    """
    for m in models:
        if hasattr(m, 'cleanup'):
            m.cleanup()

def prepare_sampling(model, noise_shape, positive, negative, noise_mask):
    """Prepares for sampling.
    Args:
        model: The model.
        noise_shape (tuple): The shape of the noise.
        positive: The positive conditioning.
        negative: The negative conditioning.
        noise_mask (torch.Tensor): The noise mask.
    Returns:
        A tuple of the real model, the positive conditioning, the negative conditioning, the noise mask, and the additional models.
    """
    device = model.load_device
    positive = convert_cond(positive)
    negative = convert_cond(negative)

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise_shape, device)

    real_model = None
    models, inference_memory = get_additional_models(positive, negative, model.model_dtype())
    ldm_patched.modules.model_management.load_models_gpu([model] + models, model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:])) + inference_memory)
    real_model = model.model

    return real_model, positive, negative, noise_mask, models


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    """Samples from a model.
    Args:
        model: The model to sample from.
        noise (torch.Tensor): The noise to start with.
        steps (int): The number of steps to sample for.
        cfg (float): The classifier-free guidance scale.
        sampler_name (str): The name of the sampler to use.
        scheduler (str): The name of the scheduler to use.
        positive: The positive conditioning.
        negative: The negative conditioning.
        latent_image (torch.Tensor): The latent image to start with.
        denoise (float, optional): The denoising strength. Defaults to 1.0.
        disable_noise (bool, optional): Whether to disable noise. Defaults to False.
        start_step (int, optional): The step to start sampling from. Defaults to None.
        last_step (int, optional): The step to stop sampling at. Defaults to None.
        force_full_denoise (bool, optional): Whether to force full denoising. Defaults to False.
        noise_mask (torch.Tensor, optional): The noise mask. Defaults to None.
        sigmas (torch.Tensor, optional): The sigmas to use. Defaults to None.
        callback (function, optional): A callback function. Defaults to None.
        disable_pbar (bool, optional): Whether to disable the progress bar. Defaults to False.
        seed (int, optional): The random seed. Defaults to None.
    Returns:
        torch.Tensor: The sampled images.
    """
    real_model, positive_copy, negative_copy, noise_mask, models = prepare_sampling(model, noise.shape, positive, negative, noise_mask)

    noise = noise.to(model.load_device)
    latent_image = latent_image.to(model.load_device)

    sampler = ldm_patched.modules.samplers.KSampler(real_model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(ldm_patched.modules.model_management.intermediate_device())

    cleanup_additional_models(models)
    cleanup_additional_models(set(get_models_from_cond(positive_copy, "control") + get_models_from_cond(negative_copy, "control")))
    return samples

def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=None, callback=None, disable_pbar=False, seed=None):
    """Samples from a model using a custom sampler.
    Args:
        model: The model to sample from.
        noise (torch.Tensor): The noise to start with.
        cfg (float): The classifier-free guidance scale.
        sampler: The sampler to use.
        sigmas (torch.Tensor): The sigmas to use.
        positive: The positive conditioning.
        negative: The negative conditioning.
        latent_image (torch.Tensor): The latent image to start with.
        noise_mask (torch.Tensor, optional): The noise mask. Defaults to None.
        callback (function, optional): A callback function. Defaults to None.
        disable_pbar (bool, optional): Whether to disable the progress bar. Defaults to False.
        seed (int, optional): The random seed. Defaults to None.
    Returns:
        torch.Tensor: The sampled images.
    """
    real_model, positive_copy, negative_copy, noise_mask, models = prepare_sampling(model, noise.shape, positive, negative, noise_mask)
    noise = noise.to(model.load_device)
    latent_image = latent_image.to(model.load_device)
    sigmas = sigmas.to(model.load_device)

    samples = ldm_patched.modules.samplers.sample(real_model, noise, positive_copy, negative_copy, cfg, model.load_device, sampler, sigmas, model_options=model.model_options, latent_image=latent_image, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(ldm_patched.modules.model_management.intermediate_device())
    cleanup_additional_models(models)
    cleanup_additional_models(set(get_models_from_cond(positive_copy, "control") + get_models_from_cond(negative_copy, "control")))
    return samples

