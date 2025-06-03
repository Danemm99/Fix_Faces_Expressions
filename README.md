# Face Expression Inpainting Pipeline

## Overview

This project implements an automated pipeline that takes an input image of a human face and fixes its expression without losing facial resemblance. The solution utilizes computer vision techniques to detect facial features and deep learning for inpainting. In particular, the pipeline generates masks for the eyes and mouth (smile) and then uses a Stable Diffusion inpainting model to adjust the expression (e.g., opening closed eyes and generating a natural smile).

## Technologies and Techniques

- **Python 3:** Primary programming language.
- **OpenCV:**  
  - Reading and processing images.
  - Face detection using two methods:
    - **OpenCV DNN face detector**: Uses `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` files.
    - **Haar‑cascade detectors**: Used in generating the mouth mask (via `haarcascade_frontalface_default.xml` and `haarcascade_smile.xml`).
  - Image operations (e.g., dilation using an elliptical kernel) to refine masks.
- **Mediapipe:**  
  - High-accuracy facial landmark detection for eyes and lips.
- **Stable Diffusion Inpainting (Diffusers Library):**  
  - Provides the inpainting pipeline to modify facial expressions.
  - Runs on PyTorch (with GPU support if available).
- **Tiled Processing and Blend Masks:**  
  - The solution divides large images into smaller overlapping tiles.
  - Each tile is processed individually and then blended seamlessly using a blend mask.
- **Command-line Interface:**  
  - The pipeline is executed via a CLI that accepts both an input image path and an output path for the final result.

## DNN Model and Haar‑cascade for Face Detection

For face detection, the pipeline employs two approaches:

- **OpenCV DNN Face Detector:**  
  The eyes processing uses an OpenCV DNN-based face detector which requires:
  - **`deploy.prototxt`**: Defines the network architecture.
  - **`res10_300x300_ssd_iter_140000.caffemodel`**: Contains the pre-trained weights.
  
- **Haar‑cascade Detectors:**  
  The mouth mask generation uses Haar‑cascade classifiers, specifically:
  - **`haarcascade_frontalface_default.xml`**
  - **`haarcascade_smile.xml`**

These files must be placed in the project root directory.

## Run the project


1. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   .\venv\Scripts\activate     # On Windows
   ```

2. **Install required packages:**

   Create a `requirements.txt` file with, the following dependencies:

   ```
   numpy
   mediapipe
   opencv-python
   Pillow
   diffusers
   transformers
   accelerate
   ```

   Than create a `requirements_torch.txt` file with, the following dependencies:

   ```
   torch
   torchvision
   torchaudio
   --index-url https://download.pytorch.org/whl/cu124     <------ go to https://pytorch.org/get-started/locally/ to find out which version you should install
   ```

   Then install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   ```bash
   pip install -r requirements_torch.txt
   ```
   
3. **Usage:**

   Run the pipeline from the command line by passing both the input image path and the desired output path. 
   For example:

   ```bash
   python main.py --input_path data/inputs/image_X.jpg --output_path data/outputs/final_image_X.png
   ```

   The script will:
   
    - Generate intermediate masks for eyes and mouth.
    - Process the image in tiles for inpainting.
    - Produce the final output image with corrected facial expression.

   To run **gradio** interface use the following command:
    ```bash
    python main.py --gradio
    ```
   After running, a URL will appear in the console (http://127.0.0.1:7860)

## Results Explanation

- **Input and Output Directories:**  
  The input images are located in the `data/inputs` folder, and the final output images are saved in the `data/outputs` folder. The output image corresponding to each input image carries the same numerical identifier as its respective input. For example, if your input image is named `image_1.jpg` in `data/inputs`, the final processed image will be saved as `final_image_1.png` (or a similar format) in `data/outputs`.

- **Overall Performance:**  
  Based on the results, the program generally performs very well in most cases. It effectively enhances the image by:
  - **Inpainting the Eyes:** If the eyes are closed, the pipeline is able to generate natural, bright, and detailed open eyes.
  - **Inpainting the Mouth (Smile):** If the face does not appear sufficiently expressive, the program adds a natural smile with a gentle, upward curve of the lips.

  This performance is achieved through the combination of precise facial landmark detection, accurate mask generation (using both DNN-based and Haar‑cascade methods), and the robust Stable Diffusion inpainting model.

- **Limitations:**  
  There are a few exceptions where the pipeline may not perform optimally:
  - **Low-Quality Images:** Poor image quality can lead to suboptimal results.
  - **Unclear Face Visibility:** If the face is not clearly visible, the corrections might not be as accurate.
  - **Small Faces:** Faces that are too small may not be detected correctly, leading to inaccurate or incomplete expression corrections.

**Conclusion:**  
Overall, as demonstrated by the output images, the pipeline works effectively in most cases. It reliably inpaints closed eyes and improves the smile when the expression is insufficient, yielding realistic and seamless modifications. However, for images of low quality, unclear face visibility, or very small faces, the corrections might be less successful.
