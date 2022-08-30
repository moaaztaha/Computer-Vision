# Facial Keypoints Detection
The repo includes multiple implementations to Facial Keypoints detection from plain pytorch with no augmentations to using FastAI.

- All notebooks used an `vgg16 with pretrained weights` except the FastAI used `resnet18`.

### Notebooks
-  `01`: contains the implementation from the book **Modern Computer Vision with Pytorch**. link to the book repo[here](https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch)
- `02`: implementation using cleaner code and recommended implementations for things like Datasets from PyTorch documentations.
- `03`: same as `02` but with custom Resize function that works for both the images and keypoints.
- `04`: implementation using FastAI for better augmentations and training.

### Deployment 
- Finally I deployed the FastAI model using [huggingface spaces](https://huggingface.co/spaces) and [gradio](https://gradio.app/docs).

#### [Demo url](https://huggingface.co/spaces/moaaztaha/facial_keypoints_detection)

![app demo](facial_keypoints_demo.gif 'Demo')
