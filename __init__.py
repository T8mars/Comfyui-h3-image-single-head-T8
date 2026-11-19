"""Reusable static single-head experiment. This is NOT recovered teacher weights."""
import hashlib
from pathlib import Path
import torch
from safetensors.torch import load_file
from comfy_api.latest import ComfyExtension,io

HEAD=Path(__file__).with_name("mean_head.safetensors")
HEAD_SHA="fab15aa9a5e0e9d78bb85d3adb5e86b17f3d697b6e29e0fe8ac7f72715ad65b9"

def replacement_tensors():
    with HEAD.open("rb") as stream:
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
    if digest!=HEAD_SHA:raise ValueError("Verified H3 image mean-head asset changed")
    data=load_file(HEAD,device="cpu")
    if set(data)!={"weight","bias"} or data["weight"].shape!=(96,5376) or data["bias"].shape!=(96,):
        raise ValueError("Unexpected mean-head asset")
    result={}
    for suffix,value in data.items():
        if value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Finite FP32 head required")
        target=torch.zeros((value.shape[0]*6,*value.shape[1:]),dtype=value.dtype)
        target[:value.shape[0]]=value
        result["diffusion_model.final_layer.video_out."+suffix]=target
    return result

class H3ImageSingleHead(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="H3ImageSingleHead",display_name="H3 Image 单头多步（实验）",
            category="H3 Image/Experimental",inputs=[io.Model.Input("model")],outputs=[io.Model.Output()])

    @classmethod
    def execute(cls,model):
        config=model.model.model_config.unet_config
        if config.get("image_model")!="minimax_h3_image" or config.get("pdd_steps")!=6:
            raise ValueError("Requires a published six-head H3 Image model; validated on full BF16/FP32 only")
        if model.patches or model.get_injections("bypass_lora"):
            raise ValueError("Place this node directly after UNETLoader, before any LoRA")
        values=replacement_tensors()
        state=model.model.state_dict()
        for key,value in values.items():
            if key not in state or state[key].shape!=value.shape:
                raise ValueError("Output-head target shape differs")
        clone=model.clone()
        patches={key:("diff",(value,)) for key,value in values.items()}
        accepted=clone.add_patches(patches,strength_patch=1.,strength_model=0.)
        if set(accepted)!=set(patches):raise ValueError("Both complete head-bank replacements must attach")
        return io.NodeOutput(clone)

class Extension(ComfyExtension):
    async def get_node_list(self):return [H3ImageSingleHead]

async def comfy_entrypoint():return Extension()
