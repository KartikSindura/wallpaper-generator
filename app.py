import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping."""
    try:
        return obj[index]
    except (KeyError, IndexError):
        if isinstance(obj, dict) and "result" in obj:
            return obj["result"][index]
        elif isinstance(obj, list) and len(obj) > 0:
            return obj[0]
        else:
            # Return the object itself if we can't extract an element
            return obj


def find_path(name: str, path: str = None) -> str:
    """Recursively looks at parent folders starting from the given path until it finds the given name."""
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)

    if parent_directory == path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """Add 'ComfyUI' to the sys.path"""
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path."""
    try:
        from main import load_extra_path_config
    except ImportError:
        print("Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.")
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def setup_environment():
    """Setup the environment by adding necessary paths and importing custom nodes."""
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    import_custom_nodes()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS"""
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


checkpoint_cache = {}
loraloader_cache = {}


def get_cached_checkpoint():
    from nodes import NODE_CLASS_MAPPINGS
    """Check for cached checkpoint."""
    if "sd_xl_base_1.0.safetensors" not in checkpoint_cache:
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"](
        )
        checkpoint_cache["sd_xl_base_1.0.safetensors"] = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )
    return checkpoint_cache["sd_xl_base_1.0.safetensors"]


def get_cached_lora():
    from nodes import NODE_CLASS_MAPPINGS
    """Check for cached Lora loader."""
    if "sdxl-dreambooth-lora.safetensors" not in loraloader_cache:
        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        loraloader_cache["sdxl-dreambooth-lora.safetensors"] = loraloader.load_lora(
            lora_name="sdxl-dreambooth-lora.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(get_cached_checkpoint(), 0),
            clip=get_value_at_index(get_cached_checkpoint(), 1),
        )
    return loraloader_cache["sdxl-dreambooth-lora.safetensors"]


def generate_image(positive_prompt, negative_prompt, width, height, steps):
    """Generate an image using ComfyUI with the provided parameters."""
    from nodes import NODE_CLASS_MAPPINGS
    import folder_paths
    import os
    from PIL import Image

    with torch.inference_mode():
        checkpoint = get_cached_checkpoint()  # Get the cached checkpoint
        loraloader_1 = get_cached_lora()  # Get the cached Lora loader
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_3 = cliptextencode.encode(
            text=positive_prompt, clip=get_value_at_index(loraloader_1, 1)
        )

        cliptextencode_8 = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(loraloader_1, 1),
        )

        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_12 = emptylatentimage.generate(
            width=width, height=height, batch_size=1
        )

        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        ksampler_4 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=steps,
            cfg=8,
            sampler_name="euler",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(loraloader_1, 0),
            positive=get_value_at_index(cliptextencode_3, 0),
            negative=get_value_at_index(cliptextencode_8, 0),
            latent_image=get_value_at_index(emptylatentimage_12, 0),
        )

        vaedecode_5 = vaedecode.decode(
            samples=get_value_at_index(ksampler_4, 0),
            vae=get_value_at_index(checkpoint, 2),
        )

        # Instead of using SaveImage, we'll directly get the image tensor and convert it
        image_tensor = get_value_at_index(vaedecode_5, 0)

        # Convert tensor to PIL image
        # The tensor is usually in the format [batch, height, width, channels]
        # We need to take the first image in the batch
        if len(image_tensor.shape) == 4:
            # Take the first image from the batch
            image_tensor = image_tensor[0]

        # Convert from [0,1] float to [0,255] uint8
        image_np = (image_tensor.cpu().numpy() * 255).astype('uint8')

        # Create a PIL image
        pil_image = Image.fromarray(image_np)

        # Save the image to a temporary file
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"generated_{
                                   random.randint(1000, 9999)}.png")
        pil_image.save(output_path)

        return output_path


def create_gradio_interface():
    """Create and launch the Gradio interface."""
    with gr.Blocks(title="Wallpaper generator") as demo:
        gr.Markdown("# Wallpaper generator")
        gr.Markdown("Generate wallpapers using SDXL with custom prompts")
        gr.Markdown("Works best with higher resolutions")

        with gr.Row():
            with gr.Column():
                positive_prompt = gr.Textbox(
                    label="Positive Prompt",
                    placeholder="Enter what you want in the image...",
                    value="yellow, black, sky, night"
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Enter what you don't want in the image...",
                    value="blurry, low quality, low resolution, distorted, deformed, unrealistic cars"
                )
                width = gr.Slider(
                    minimum=100, maximum=2560, value=800, step=64,
                    label="Width"
                )
                height = gr.Slider(
                    minimum=100, maximum=1440, value=600, step=64,
                    label="Height"
                )
                steps = gr.Slider(
                    minimum=1, maximum=20, value=20, step=1,
                    label="Steps"
                )
                generate_btn = gr.Button("Generate Image")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")

        generate_btn.click(
            fn=generate_image,
            inputs=[positive_prompt, negative_prompt, width, height, steps],
            outputs=output_image
        )

    return demo


if __name__ == "__main__":
    # Setup ComfyUI environment
    setup_environment()

    # Create and launch Gradio interface
    demo = create_gradio_interface()
    demo.launch(share=False)
