import os
import torch
import argparse
from PIL import Image
import cv2
import gradio as gr

from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

from src.eyes_utils import create_eye_mask_with_dnn, PROTOTXT_PATH, MODEL_PATH
from src.mouth_utils import create_mouth_mask_for_non_smiling_faces
from src.inpainting_utils import run_inpainting_stage, is_mask_empty, create_blend_mask


def process_image(input_image: Image.Image) -> Image.Image:
    tmp_dir = "data/outputs"
    os.makedirs(tmp_dir, exist_ok=True)
    input_path = os.path.join(tmp_dir, "gr_input.png")
    input_image.save(input_path)

    eyes_mask_path = os.path.join(tmp_dir, "eyes_mask.png")
    mouth_mask_path = os.path.join(tmp_dir, "mouth_mask.png")
    stage1_output_path = os.path.join(tmp_dir, "opened_eyes_only.png")
    final_output_path = os.path.join(tmp_dir, "gr_result.png")

    eye_mask = create_eye_mask_with_dnn(input_path, PROTOTXT_PATH, MODEL_PATH)
    if eye_mask is None:
        raise RuntimeError("Не вдалося створити маску очей.")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eye_mask = cv2.dilate(eye_mask, kernel, iterations=3)
    cv2.imwrite(eyes_mask_path, eye_mask)

    mouth_mask = create_mouth_mask_for_non_smiling_faces(input_path)
    if mouth_mask is None:
        raise RuntimeError("Не вдалося створити маску рота.")
    cv2.imwrite(mouth_mask_path, mouth_mask)

    with Image.open(input_path) as img:
        target_wh = img.size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "stabilityai/stable-diffusion-2-inpainting"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype, scheduler=scheduler
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(43)
    blend_mask = create_blend_mask(tile_size=512, overlap=256)

    if is_mask_empty(eyes_mask_path):
        result_s1 = Image.open(input_path).convert("RGB")
    else:
        result_s1 = run_inpainting_stage(
            pipeline=pipeline, generator=generator, device=device,
            image_input=input_path, mask_path=eyes_mask_path, output_path=stage1_output_path,
            prompt=(
                "Ultra-photorealistic inpainting: Generate natural, bright, perfectly open human eyes with detailed, "
                "realistic iris color and texture, lifelike catchlights, and individual eyelashes. Ensure seamless "
                "integration with the surrounding area, precisely matching the original image's lighting, skin texture, "
                "style, and focus. Maintain facial symmetry; if one eye is open, match the other eye exactly. If glasses "
                "are present, render eyes open behind lenses, keeping glasses intact. Sharp focus, high detail, balanced "
                "lighting, no visible seams or artifacts."
            ),
            negative_prompt=(
                "Closed eyes, squinting, half-closed, sleepy eyes, droopy eyelids, uneven eyes, asymmetrical eyes, "
                "crossed eyes, cartoon, anime, illustration, sketch, deformed, mutated, blurry, artifacts, noisy."
            ),
            stage_name="Eyes", tile_size=512, overlap=256, num_steps=120, guidance=10,
            max_initial_size=1024, blend_mask=blend_mask, target_original_wh=target_wh
        )
    result_s1.save(stage1_output_path)

    if is_mask_empty(mouth_mask_path):
        final = result_s1
    else:
        final = run_inpainting_stage(
            pipeline=pipeline, generator=generator, device=device,
            image_input=result_s1, mask_path=mouth_mask_path, output_path=final_output_path,
            prompt=(
                "Ultra-photorealistic inpainting: Generate a natural and inviting wide smile with a gentle, "
                "upward curve of the lips that clearly reveals realistic upper teeth. The lips must display a natural "
                "hue and texture, blending seamlessly with the adjacent skin. Ensure impeccable integration with the "
                "image's lighting, skin texture, style, and focus. The resulting smile should appear balanced, "
                "unmistakably visible, and free of artifacts."
            ),
            negative_prompt=(
                "Avoid sad, frowning, angry, neutral. No closed lips, grimacing, cartoonish or exaggerated smiles, "
                "visible tongue, blurry, low-res, artifacts."
            ),
            stage_name="Mouth_Smile", tile_size=512, overlap=256, num_steps=120, guidance=10,
            max_initial_size=1024, blend_mask=blend_mask, target_original_wh=target_wh
        )

    return final


def main():
    parser = argparse.ArgumentParser(description="Automated face expression inpainting pipeline")
    parser.add_argument("--input_path", type=str, help="Path to the input image")
    parser.add_argument("--output_path", type=str, help="Path to save the final output image")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio web interface")
    args = parser.parse_args()

    if args.gradio:
        iface = gr.Interface(
            fn=process_image,
            inputs=gr.Image(type="pil"),
            outputs=gr.Image(type="pil"),
            title="Корекція виразу обличчя",
            description="Завантажте фото з заплющеними очима або без усмішки — отримаєте виправлене зображення."
        )
        iface.launch()
    else:
        if not args.input_path or not args.output_path:
            parser.error("Потрібно вказати --input_path та --output_path або --gradio")
        img = Image.open(args.input_path).convert("RGB")
        result = process_image(img)
        result.save(args.output_path)
        print(f"Результат збережено в: {args.output_path}")


if __name__ == "__main__":
    main()
