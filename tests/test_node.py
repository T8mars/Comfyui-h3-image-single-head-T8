"""CPU checks against ComfyUI's actual ModelPatcher, without model loading."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

COMFY = Path(os.environ['COMFYUI_PATH']).resolve()
sys.path.insert(0, str(COMFY))
from comfy.cli_args import args
args.cpu = True
import torch
import comfy.lora
from comfy.model_patcher import ModelPatcher
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h3_single_head_under_test', ROOT / '__init__.py')
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


def model():
    obj = torch.nn.Module()
    obj.model_config = types.SimpleNamespace(unet_config={'image_model': 'minimax_h3_image', 'pdd_steps': 6})
    obj.diffusion_model = torch.nn.Module()
    obj.diffusion_model.final_layer = torch.nn.Module()
    obj.diffusion_model.final_layer.video_out = torch.nn.Linear(5376, 576, device='meta')
    return ModelPatcher(obj, torch.device('cpu'), torch.device('cpu'))


class SingleHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_asset_and_head_layout(self):
        original = load_file(ROOT / 'mean_head.safetensors')
        values = node.replacement_tensors()
        self.assertEqual(set(original), {'weight', 'bias'})
        self.assertEqual(len(values), 2)
        for suffix, value in original.items():
            expanded = values['diffusion_model.final_layer.video_out.' + suffix]
            self.assertEqual(expanded.dtype, torch.float32)
            self.assertTrue(torch.equal(expanded[:96], value))
            self.assertEqual(torch.count_nonzero(expanded[96:]).item(), 0)

    def test_native_patch_exact_and_source_unchanged(self):
        source = model()
        output = node.H3ImageSingleHead.execute(source)[0]
        self.assertIsNot(output, source)
        self.assertFalse(source.patches)
        for key, target in node.replacement_tensors().items():
            actual = comfy.lora.calculate_weight(output.patches[key], torch.ones_like(target), key)
            self.assertTrue(torch.equal(actual, target))

    def test_reject_wrong_model(self):
        source = model()
        source.model.model_config.unet_config['pdd_steps'] = 1
        with self.assertRaises(ValueError):
            node.H3ImageSingleHead.execute(source)

    def test_reject_existing_patches(self):
        source = model()
        source.patches = {'existing': []}
        with self.assertRaises(ValueError):
            node.H3ImageSingleHead.execute(source)

    def test_reject_changed_or_missing_asset(self):
        original = node.HEAD
        try:
            with tempfile.TemporaryDirectory() as tmp:
                node.HEAD = Path(tmp) / 'missing.safetensors'
                with self.assertRaises(FileNotFoundError):
                    node.replacement_tensors()
                node.HEAD.write_bytes(b'invalid asset')
                with self.assertRaises(ValueError):
                    node.replacement_tensors()
        finally:
            node.HEAD = original


if __name__ == '__main__':
    unittest.main()
