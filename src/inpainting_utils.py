import math
import time
import os
from PIL import Image
import numpy as np


def pad_to_multiple_of_8(image: Image.Image) -> tuple:
    w, h = image.size
    new_w = (w + 7) // 8 * 8
    new_h = (h + 7) // 8 * 8
    if new_w == w and new_h == h:
        return image, 0, 0
    mode = image.mode
    if 'A' in mode:
        fill_color = (0, 0, 0, 0)
    elif mode == 'L':
        fill_color = 0
    else:
        fill_color = (0, 0, 0)
    padded = Image.new(mode, (new_w, new_h), fill_color)
    offset_x = (new_w - w) // 2
    offset_y = (new_h - h) // 2
    padded.paste(image, (offset_x, offset_y))
    return padded, offset_x, offset_y


def unpad_image(padded_image: Image.Image, offset_x: int, offset_y: int, target_w: int, target_h: int) -> Image.Image:
    left = offset_x
    top = offset_y
    right = min(offset_x + target_w, padded_image.width)
    bottom = min(offset_y + target_h, padded_image.height)
    cropped = padded_image.crop((left, top, right, bottom))
    if cropped.size != (target_w, target_h):
        mode = padded_image.mode
        if 'A' in mode:
            fill_color = (0, 0, 0, 0)
        elif mode == 'L':
            fill_color = 0
        else:
            fill_color = (0, 0, 0)
        final_image = Image.new(mode, (target_w, target_h), fill_color)
        paste_x = (target_w - cropped.width) // 2
        paste_y = (target_h - cropped.height) // 2
        final_image.paste(cropped, (paste_x, paste_y))
        return final_image
    else:
        return cropped


def load_and_resize_input(image_input, mask_path, max_size=1024):
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.copy().convert("RGB")
    else:
        raise TypeError("image_input must be a file path (str) or a PIL Image object")

    msk = Image.open(mask_path).convert("L")
    original_w, original_h = img.size
    w, h = img.size
    was_resized = False
    processed_img = img
    processed_msk = msk
    if max(w, h) > max_size:
        was_resized = True
        scale_factor = max_size / max(w, h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        new_w += (new_w % 2)
        new_h += (new_h % 2)
        print(f"Resizing input image from {w}x{h} to {new_w}x{new_h} for processing.")
        processed_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        processed_msk = msk.resize((new_w, new_h), Image.Resampling.NEAREST)
    return processed_img, processed_msk, original_w, original_h, was_resized


def create_blend_mask(tile_size, overlap):
    if overlap >= tile_size or overlap <= 0 or overlap % 2 != 0:
        raise ValueError("Overlap must be positive, even, and less than tile_size.")
    blend_mask = np.ones((tile_size, tile_size, 1), dtype=np.float32)
    for i in range(overlap):
        weight = (i + 1) / (overlap + 1)
        blend_mask[i, :, 0] *= weight
        blend_mask[tile_size - 1 - i, :, 0] *= weight
        blend_mask[:, i, 0] *= weight
        blend_mask[:, tile_size - 1 - i, 0] *= weight
    return blend_mask


def is_mask_empty(mask_path: str) -> bool:
    msk = Image.open(mask_path).convert("L")
    msk_np = np.array(msk)
    return not np.any(msk_np > 127)


def run_inpainting_stage(
    pipeline, generator, device,
    image_input, mask_path, output_path,
    prompt, negative_prompt, stage_name,
    tile_size, overlap, num_steps, guidance, max_initial_size,
    blend_mask,
    target_original_wh
):
    stage_start_time = time.time()
    print("\n" + "=" * 42)
    print(f"=== STARTING STAGE: {stage_name} ===")
    print("=" * 42)

    try:
        processed_original, processed_mask, _, _, was_resized = load_and_resize_input(image_input, mask_path, max_size=max_initial_size)
        processed_w, processed_h = processed_original.size
    except Exception as e:
        print(f"Error during input loading for Stage '{stage_name}': {e}")
        return None

    print(f"Stage '{stage_name}': Processing image size: {processed_w}x{processed_h}")

    mask_np = np.array(processed_mask)
    mask_np = np.where(mask_np > 127, 255, 0).astype(np.uint8)

    if not np.any(mask_np):
        print(f"Warning (Stage '{stage_name}'): Mask ({os.path.basename(mask_path)}) contains no white areas. Skipping this stage.")
        final_skipped_image = processed_original
        target_w, target_h = target_original_wh

        if final_skipped_image.size != target_original_wh:
            print(f"Stage '{stage_name}': Resizing skipped image from {final_skipped_image.size} to {target_original_wh}")
            try:
                final_skipped_image = final_skipped_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"Error resizing skipped image for stage '{stage_name}': {e}")
                return None

        try:
            final_skipped_image.save(output_path)
            print(f"Stage '{stage_name}': Saved unchanged image to: {output_path}")
        except Exception as e:
            print(f"Error saving skipped stage '{stage_name}' output image: {e}")
        return final_skipped_image

    img_np = np.array(processed_original)
    final_img_np = img_np.copy()
    result_accumulator = np.zeros_like(img_np, dtype=np.float32)
    blend_count = np.zeros((processed_h, processed_w, 1), dtype=np.float32)

    step = tile_size - overlap
    tile_index, processed_tile_count = 0, 0
    total_possible_tiles = math.ceil(processed_w / step) * math.ceil(processed_h / step)
    print(f"Stage '{stage_name}': Starting tile processing ({total_possible_tiles} possible tiles)...")

    for y in range(0, processed_h, step):
        for x in range(0, processed_w, step):
            tile_index += 1
            x_start, y_start = x, y
            x_end = min(x_start + tile_size, processed_w)
            y_end = min(y_start + tile_size, processed_h)
            current_tile_w = x_end - x_start
            current_tile_h = y_end - y_start

            if not np.any(mask_np[y_start:y_end, x_start:x_end]):
                continue

            processed_tile_count += 1

            tile_img_np_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            tile_mask_np_full = np.zeros((tile_size, tile_size), dtype=np.uint8)

            tile_img_np_tile[:current_tile_h, :current_tile_w] = img_np[y_start:y_end, x_start:x_end]
            tile_mask_np_full[:current_tile_h, :current_tile_w] = mask_np[y_start:y_end, x_start:x_end]

            if current_tile_w < tile_size:
                tile_img_np_tile[:current_tile_h, current_tile_w:] = tile_img_np_tile[:current_tile_h, current_tile_w-1:current_tile_w]
                tile_mask_np_full[:current_tile_h, current_tile_w:] = tile_mask_np_full[:current_tile_h, current_tile_w-1:current_tile_w]
            if current_tile_h < tile_size:
                tile_img_np_tile[current_tile_h:, :] = tile_img_np_tile[current_tile_h-1:current_tile_h, :]
                tile_mask_np_full[current_tile_h:, :] = tile_mask_np_full[current_tile_h-1:current_tile_h, :]

            tile_img_pil = Image.fromarray(tile_img_np_tile)
            tile_mask_pil = Image.fromarray(tile_mask_np_full).convert("RGB")
            padded_tile_img, off_x, off_y = pad_to_multiple_of_8(tile_img_pil)
            padded_tile_mask, _, _ = pad_to_multiple_of_8(tile_mask_pil)

            if device == "cuda":
                import torch
                torch.cuda.empty_cache()

            result_tile_images = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=padded_tile_img,
                mask_image=padded_tile_mask,
                height=padded_tile_img.size[1],
                width=padded_tile_img.size[0],
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=generator
            ).images

            padded_result_tile = result_tile_images[0]
            unpadded_result_tile = unpad_image(padded_result_tile, off_x, off_y, tile_size, tile_size)
            result_tile_np = np.array(unpadded_result_tile).astype(np.float32)
            current_blend_mask = blend_mask[:current_tile_h, :current_tile_w]
            weighted_result_tile = result_tile_np[:current_tile_h, :current_tile_w] * current_blend_mask
            result_accumulator[y_start:y_end, x_start:x_end] += weighted_result_tile
            blend_count[y_start:y_end, x_start:x_end] += current_blend_mask

    print(f"Stage '{stage_name}': Compositing final image...")
    epsilon = 1e-6
    processed_areas_mask = blend_count > epsilon
    normalized_result = np.where(processed_areas_mask, result_accumulator / (blend_count + epsilon), 0)
    final_img_np = np.where(processed_areas_mask, np.clip(normalized_result, 0, 255).astype(np.uint8), final_img_np)
    final_processed_image = Image.fromarray(final_img_np)

    target_w, target_h = target_original_wh
    if was_resized or final_processed_image.size != target_original_wh:
        print(f"Stage '{stage_name}': Resizing output image from {final_processed_image.size} to target {target_original_wh}")
        try:
            final_output_image = final_processed_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error resizing final image for stage '{stage_name}': {e}")
            return None
    else:
        final_output_image = final_processed_image

    try:
        final_output_image.save(output_path)
        print(f"Stage '{stage_name}': Output image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving stage '{stage_name}' output image: {e}")

    stage_end_time = time.time()
    print(f"Stage '{stage_name}' finished.")
    if processed_tile_count > 0:
        print(f"Tiles processed (Stage '{stage_name}'): {processed_tile_count}")
    print(f"Time taken for Stage '{stage_name}': {stage_end_time - stage_start_time:.2f} seconds.")

    return final_output_image
