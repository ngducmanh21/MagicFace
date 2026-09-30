
import argparse
import json
import os
from pathlib import Path

from PIL import Image
from mgface.au import AU_NAMES, parse_au_request
from mgface.verification import add_report_arguments, report_from_args, validate_scale

# AU mapping
ind_dict = {name: index for index, name in enumerate(AU_NAMES)}

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a MagicFace test script.")
    # /home/mengting/Desktop/diffusion_models/stable-diffusion-v1-5
    # sd-legacy/stable-diffusion-v1-5
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='sd-legacy/stable-diffusion-v1-5',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument("--seed", type=int, default=424,
                        help="Inference seed, reset for every AU variant for comparison.")

    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--denoising_unet_path",
        type=str,
        default='mengtingwei/magicface',
    )
    
    parser.add_argument(
        "--ID_unet_path",
        type=str,
        default='mengtingwei/magicface',
    )

    parser.add_argument(
        "--au_test",
        type=str,
        default='',
    )

    parser.add_argument(
        "--AU_variation",
        type=str,
        action='append',
        required=True,
        help="AU changes separated by '+'. Repeat this option to compare multiple edits.",
    )

    parser.add_argument(
        "--img_path",
        type=str,
        default='',
    )

    parser.add_argument(
        "--bg_path",
        type=str,
        default='',
    )

    parser.add_argument(
        "--saved_path",
        type=str,
        default='edited_images',
    )
    parser.add_argument('--verify', action='store_true',
                        help='Export a visual report with measured AU intensities. Automatic for multiple edits.')
    parser.add_argument('--verification_dir', default=None)
    add_report_arguments(parser)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    try:
        args.au_requests = [parse_au_request(args.au_test, value) for value in args.AU_variation]
        validate_scale(args.au_delta_scale)
        if args.inference_steps <= 0:
            raise ValueError('--inference_steps must be positive.')
        for path in (args.img_path, args.bg_path):
            if not path or not Path(path).is_file():
                raise ValueError(f'Input image does not exist: {path!r}')
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return args


def make_data(args):
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()

    img_name = args.img_path
    bg_name = args.bg_path

    with Image.open(img_name) as image:
        source = transform(image.convert('RGB'))
    with Image.open(bg_name) as image:
        bg = transform(image.convert('RGB'))
    if source.shape != bg.shape:
        raise ValueError('Source and background images must have matching dimensions.')

    return source, bg


def tokenize_captions(tokenizer, captions, max_length):

    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs.input_ids


def load_pipeline(args):
    """Load pretrained weights once; reusable across all images in a dataset."""
    import torch
    from diffusers import AutoencoderKL, UniPCMultistepScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from mgface.pipelines_mgface.pipeline_mgface import MgPipeline as MgPipelineInference
    from mgface.pipelines_mgface.unet_ID_2d_condition import UNetID2DConditionModel
    from mgface.pipelines_mgface.unet_deno_2d_condition import UNetDeno2DConditionModel

    device = 'cuda'
    denoising_unet_path = args.denoising_unet_path
    ID_unet_path = args.ID_unet_path

    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            cache_dir='./'
        ).to(device)
    text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
        cache_dir='./'
        ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
        cache_dir='./'
        )

    unet_ID = UNetID2DConditionModel.from_pretrained(
            ID_unet_path,
            subfolder='ID_enc',
            # torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            cache_dir='./',
        )

    # 
    unet_deno = UNetDeno2DConditionModel.from_pretrained(
            denoising_unet_path,
            subfolder='denoising_unet',
            # torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
        cache_dir='./',
        )

    unet_deno.requires_grad_(False)
    unet_ID.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    

    weight_dtype = torch.float16


    pipeline = MgPipelineInference.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet_ID=unet_ID,
        unet_deno=unet_deno,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    ).to(device)
    
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    prompt = 'A close up of a person.'
    prompt_embeds = text_encoder(tokenize_captions(tokenizer, [prompt], 2).to(device))[0]
    return pipeline, prompt_embeds


def generate_edits(args, pipeline, prompt_embeds, images=None, on_case=None):
    import numpy as np
    import torch
    from time import perf_counter

    device = pipeline._execution_device
    source, bg = images if images is not None else make_data(args)
    source = source.unsqueeze(0)
    bg = bg.unsqueeze(0)
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    image_path = Path(args.img_path)
    cases = []
    for index, requested in enumerate(args.au_requests):
        # Reuse the initial noise for every variant: only the AU condition changes.
        generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
        au_prompt = np.array([requested.get(name, 0.0) for name in AU_NAMES], dtype=np.float32)
        tor_exp = torch.from_numpy(au_prompt).unsqueeze(0)
        started = perf_counter()
        sample = pipeline(
            prompt_embeds=prompt_embeds, source=source, bg=bg, au=tor_exp,
            num_inference_steps=args.inference_steps, generator=generator,
        ).images[0]
        elapsed = perf_counter() - started
        filename = (image_path.name if len(args.au_requests) == 1
                    else f'{image_path.stem}_edit_{index + 1:03d}.png')
        result_path = Path(saved_path) / filename
        if result_path.resolve() in (image_path.resolve(), Path(args.bg_path).resolve()):
            raise ValueError('--saved_path would overwrite an input image. Choose another directory.')
        sample.save(result_path)
        label = ', '.join(f'{name} {value:+g}' for name, value in requested.items())
        cases.append({'source': str(image_path.resolve()), 'result': str(result_path.resolve()),
                      'requested_aus': requested, 'label': label, 'seed': args.seed,
                      'inference_steps': args.inference_steps,
                      'generation_seconds': elapsed, 'background': str(Path(args.bg_path).resolve())})
        print(f'Saved {label}: {result_path}')
        if on_case is not None:
            on_case(cases[-1], index)
    return cases


def model_metadata(args):
    return {'base_model': args.pretrained_model_name_or_path, 'ID_unet': args.ID_unet_path,
            'denoising_unet': args.denoising_unet_path, 'revision': args.revision,
            'variant': args.variant, 'prompt': 'A close up of a person.'}


def main(args):
    # Validate/read images before downloading model weights.
    images = make_data(args)
    pipeline, prompt_embeds = load_pipeline(args)
    cases = generate_edits(args, pipeline, prompt_embeds, images=images)
    image_path = Path(args.img_path)
    saved_path = args.saved_path

    if args.verify or args.evidence or len(cases) > 1:
        report_dir = Path(args.verification_dir or Path(saved_path) / f'{image_path.stem}_verification')
        report_dir.mkdir(parents=True, exist_ok=True)
        metadata = {**model_metadata(args), 'background': str(Path(args.bg_path).resolve())}
        # Persist before scoring, so reports can be rebuilt without generating images again.
        (report_dir / 'manifest.json').write_text(json.dumps({'cases': cases, 'metadata': metadata}, indent=2))
        report_from_args(cases, report_dir, args, metadata=metadata)
    print('done')

if __name__ == "__main__":
    args = parse_args()

    main(args)
