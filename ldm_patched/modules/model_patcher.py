import torch
import copy
import inspect

import ldm_patched.modules.utils
import ldm_patched.modules.model_management

class ModelPatcher:
    """A class for patching models."""
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False):
        """Initializes a ModelPatcher.
        Args:
            model: The model to patch.
            load_device (torch.device): The device to load the model on.
            offload_device (torch.device): The device to offload the model to.
            size (int, optional): The size of the model. Defaults to 0.
            current_device (torch.device, optional): The current device of the model. Defaults to None.
            weight_inplace_update (bool, optional): Whether to update the weights in place. Defaults to False.
        """
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options":{}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

        self.weight_inplace_update = weight_inplace_update

    def model_size(self):
        """Gets the size of the model.
        Returns:
            int: The size of the model in bytes.
        """
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        self.size = ldm_patched.modules.model_management.module_size(self.model)
        self.model_keys = set(model_sd.keys())
        return self.size

    def clone(self):
        """Clones the ModelPatcher.
        Returns:
            ModelPatcher: The cloned ModelPatcher.
        """
        n = ModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        return n

    def is_clone(self, other):
        """Checks if another ModelPatcher is a clone of this one.
        Args:
            other (ModelPatcher): The other ModelPatcher.
        Returns:
            bool: True if the other ModelPatcher is a clone of this one, False otherwise.
        """
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

    def memory_required(self, input_shape):
        """Gets the memory required for the model.
        Args:
            input_shape (tuple): The input shape.
        Returns:
            int: The memory required for the model in bytes.
        """
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        """Sets the model's sampler CFG function.
        Args:
            sampler_cfg_function (function): The sampler CFG function.
            disable_cfg1_optimization (bool, optional): Whether to disable CFG1 optimization. Defaults to False.
        """
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"]) #Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        """Sets the model's sampler post-CFG function.
        Args:
            post_cfg_function (function): The sampler post-CFG function.
            disable_cfg1_optimization (bool, optional): Whether to disable CFG1 optimization. Defaults to False.
        """
        self.model_options["sampler_post_cfg_function"] = self.model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        """Sets the model's UNet function wrapper.
        Args:
            unet_wrapper_function (function): The UNet function wrapper.
        """
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_patch(self, patch, name):
        """Sets a model patch.
        Args:
            patch: The patch to set.
            name (str): The name of the patch.
        """
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        """Sets a model patch replacement.
        Args:
            patch: The patch to set.
            name (str): The name of the patch.
            block_name (str): The name of the block.
            number (int): The number of the block.
            transformer_index (int, optional): The index of the transformer. Defaults to None.
        """
        to = self.model_options["transformer_options"]
        if "patches_replace" not in to:
            to["patches_replace"] = {}
        if name not in to["patches_replace"]:
            to["patches_replace"][name] = {}
        if transformer_index is not None:
            block = (block_name, number, transformer_index)
        else:
            block = (block_name, number)
        to["patches_replace"][name][block] = patch

    def set_model_attn1_patch(self, patch):
        """Sets a model attention1 patch.
        Args:
            patch: The patch to set.
        """
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        """Sets a model attention2 patch.
        Args:
            patch: The patch to set.
        """
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        """Sets a model attention1 replacement.
        Args:
            patch: The patch to set.
            block_name (str): The name of the block.
            number (int): The number of the block.
            transformer_index (int, optional): The index of the transformer. Defaults to None.
        """
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        """Sets a model attention2 replacement.
        Args:
            patch: The patch to set.
            block_name (str): The name of the block.
            number (int): The number of the block.
            transformer_index (int, optional): The index of the transformer. Defaults to None.
        """
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch):
        """Sets a model attention1 output patch.
        Args:
            patch: The patch to set.
        """
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        """Sets a model attention2 output patch.
        Args:
            patch: The patch to set.
        """
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        """Sets a model input block patch.
        Args:
            patch: The patch to set.
        """
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        """Sets a model input block patch after the skip connection.
        Args:
            patch: The patch to set.
        """
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        """Sets a model output block patch.
        Args:
            patch: The patch to set.
        """
        self.set_model_patch(patch, "output_block_patch")

    def add_object_patch(self, name, obj):
        """Adds an object patch.
        Args:
            name (str): The name of the patch.
            obj: The object to patch.
        """
        self.object_patches[name] = obj

    def model_patches_to(self, device):
        """Moves the model patches to a device.
        Args:
            device (torch.device): The device to move the patches to.
        """
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        """Gets the dtype of the model.
        Returns:
            torch.dtype: The dtype of the model.
        """
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        """Adds patches to the model.
        Args:
            patches (dict): A dictionary of patches to add.
            strength_patch (float, optional): The strength of the patch. Defaults to 1.0.
            strength_model (float, optional): The strength of the model. Defaults to 1.0.
        Returns:
            list: A list of the patched keys.
        """
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches

        return list(p)

    def get_key_patches(self, filter_prefix=None):
        """Gets the key patches.
        Args:
            filter_prefix (str, optional): A prefix to filter the keys by. Defaults to None.
        Returns:
            dict: A dictionary of the key patches.
        """
        ldm_patched.modules.model_management.unload_model_clones(self)
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            if k in self.patches:
                p[k] = [model_sd[k]] + self.patches[k]
            else:
                p[k] = (model_sd[k],)
        return p

    def model_state_dict(self, filter_prefix=None):
        """Gets the model's state dict.
        Args:
            filter_prefix (str, optional): A prefix to filter the keys by. Defaults to None.
        Returns:
            dict: The model's state dict.
        """
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_model(self, device_to=None, patch_weights=True):
        """Patches the model.
        Args:
            device_to (torch.device, optional): The device to move the model to. Defaults to None.
            patch_weights (bool, optional): Whether to patch the weights. Defaults to True.
        Returns:
            The patched model.
        """
        for k in self.object_patches:
            old = getattr(self.model, k)
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old
            setattr(self.model, k, self.object_patches[k])

        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    print("could not patch. key doesn't exist in model:", key)
                    continue

                weight = model_sd[key]

                inplace_update = self.weight_inplace_update

                if key not in self.backup:
                    self.backup[key] = weight.to(device=self.offload_device, copy=inplace_update)

                if device_to is not None:
                    temp_weight = ldm_patched.modules.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
                else:
                    temp_weight = weight.to(torch.float32, copy=True)
                out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
                if inplace_update:
                    ldm_patched.modules.utils.copy_to_param(self.model, key, out_weight)
                else:
                    ldm_patched.modules.utils.set_attr(self.model, key, out_weight)
                del temp_weight

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def calculate_weight(self, patches, weight, key):
        """Calculates the weight of a patch.
        Args:
            patches (list): A list of patches.
            weight: The weight to patch.
            key (str): The key of the weight.
        Returns:
            The patched weight.
        """
        for p in patches:
            alpha = p[0]
            v = p[1]
            strength_model = p[2]

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                v = (self.calculate_weight(v[1:], v[0].clone(), key), )

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                w1 = v[0]
                if alpha != 0.0:
                    if w1.shape != weight.shape:
                        print("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(key, w1.shape, weight.shape))
                    else:
                        weight += alpha * ldm_patched.modules.model_management.cast_to_device(w1, weight.device, weight.dtype)
            elif patch_type == "lora": #lora/locon
                mat1 = ldm_patched.modules.model_management.cast_to_device(v[0], weight.device, torch.float32)
                mat2 = ldm_patched.modules.model_management.cast_to_device(v[1], weight.device, torch.float32)
                if v[2] is not None:
                    alpha *= v[2] / mat2.shape[0]
                if v[3] is not None:
                    #locon mid weights, hopefully the math is fine because I didn't properly test it
                    mat3 = ldm_patched.modules.model_management.cast_to_device(v[3], weight.device, torch.float32)
                    final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                    mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
                try:
                    weight += (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))).reshape(weight.shape).type(weight.dtype)
                except Exception as e:
                    print("ERROR", key, e)
            elif patch_type == "lokr":
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w1_a, weight.device, torch.float32),
                                  ldm_patched.modules.model_management.cast_to_device(w1_b, weight.device, torch.float32))
                else:
                    w1 = ldm_patched.modules.model_management.cast_to_device(w1, weight.device, torch.float32)

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w2_a, weight.device, torch.float32),
                                      ldm_patched.modules.model_management.cast_to_device(w2_b, weight.device, torch.float32))
                    else:
                        w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                          ldm_patched.modules.model_management.cast_to_device(t2, weight.device, torch.float32),
                                          ldm_patched.modules.model_management.cast_to_device(w2_b, weight.device, torch.float32),
                                          ldm_patched.modules.model_management.cast_to_device(w2_a, weight.device, torch.float32))
                else:
                    w2 = ldm_patched.modules.model_management.cast_to_device(w2, weight.device, torch.float32)

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha *= v[2] / dim

                try:
                    weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(weight.dtype)
                except Exception as e:
                    print("ERROR", key, e)
            elif patch_type == "loha":
                w1a = v[0]
                w1b = v[1]
                if v[2] is not None:
                    alpha *= v[2] / w1b.shape[0]
                w2a = v[3]
                w2b = v[4]
                if v[5] is not None: #cp decomposition
                    t1 = v[5]
                    t2 = v[6]
                    m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      ldm_patched.modules.model_management.cast_to_device(t1, weight.device, torch.float32),
                                      ldm_patched.modules.model_management.cast_to_device(w1b, weight.device, torch.float32),
                                      ldm_patched.modules.model_management.cast_to_device(w1a, weight.device, torch.float32))

                    m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      ldm_patched.modules.model_management.cast_to_device(t2, weight.device, torch.float32),
                                      ldm_patched.modules.model_management.cast_to_device(w2b, weight.device, torch.float32),
                                      ldm_patched.modules.model_management.cast_to_device(w2a, weight.device, torch.float32))
                else:
                    m1 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w1a, weight.device, torch.float32),
                                  ldm_patched.modules.model_management.cast_to_device(w1b, weight.device, torch.float32))
                    m2 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w2a, weight.device, torch.float32),
                                  ldm_patched.modules.model_management.cast_to_device(w2b, weight.device, torch.float32))

                try:
                    weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
                except Exception as e:
                    print("ERROR", key, e)
            elif patch_type == "glora":
                if v[4] is not None:
                    alpha *= v[4] / v[0].shape[0]

                a1 = ldm_patched.modules.model_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, torch.float32)
                a2 = ldm_patched.modules.model_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, torch.float32)
                b1 = ldm_patched.modules.model_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, torch.float32)
                b2 = ldm_patched.modules.model_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, torch.float32)

                weight += ((torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)) * alpha).reshape(weight.shape).type(weight.dtype)
            else:
                print("patch type not recognized", patch_type, key)

        return weight

    def unpatch_model(self, device_to=None):
        """Unpatches the model.
        Args:
            device_to (torch.device, optional): The device to move the model to. Defaults to None.
        """
        keys = list(self.backup.keys())

        if self.weight_inplace_update:
            for k in keys:
                ldm_patched.modules.utils.copy_to_param(self.model, k, self.backup[k])
        else:
            for k in keys:
                ldm_patched.modules.utils.set_attr(self.model, k, self.backup[k])

        self.backup = {}

        if device_to is not None:
            self.model.to(device_to)
            self.current_device = device_to

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            setattr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup = {}
