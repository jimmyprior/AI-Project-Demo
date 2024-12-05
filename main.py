import librosa
import torch 
import numpy as np

from PIL import Image

import gradio as gr

import torchvision


from utils import mel_spectrogram, extract_audio_segment, rescale_minmax


SAMPLE_LENGTH = 5
MODEL_PATH = "70_percent_good.pth"


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),  # Resize images to 256x256
    torchvision.transforms.CenterCrop(224),  # Crop to 224x224
    torchvision.transforms.ToTensor(), # converts images to tensors for use with PyTorch
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalizes the images so for higher accuracy
])


# Load the model with trained weights
model = torchvision.models.resnet34() # Uses pretrained weights for the resnet34 model, hence default

# Modifies the fully connected layer of the pretained model to match the output size we need, which is based on the number of classese have (3)
num_classes = 3
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


def predict_from_spectogram(image : Image.Image):
        # Load and preprocess an example input
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    outputs = model(input_tensor)

    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(input_tensor)
        predictions = torch.argmax(outputs, dim=1)  # Get the predicted class
        print(predictions)
        print(f"Predicted class: {predictions.item()}")
    
    return predictions.item()


def generate_spectogram(nd_array, sample_rate) -> Image.Image:
    #sample at 0 seconds for length interval
    
    segment_y = extract_audio_segment(
        nd_array, 
        sample_rate, 
        0, 
        SAMPLE_LENGTH
    )
    #generate the spectograms and save them to the output folder in drive
    data = mel_spectrogram(segment_y, sample_rate)
    data = rescale_minmax(data) # rescale btwn 0,1
    scale_coef = (256 - 1e-06)
    data = (data*scale_coef).astype(np.uint8) # rescale to 255 pixel scale for image compatibility
    data = Image.fromarray(data)
    return data.convert("RGB") #need to convert to rgb because model expects data to have 3 color channels


def classify_audio(audio_file_path) -> str:
    
    #got from print idx in trainset 
    labels = {
        0 : 'Large',
        1 : 'Medium',
        2 : 'Small'
    }   
    #what is called by gradio when audio data is recieved
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
    #generate the spectogram 
    img = generate_spectogram(audio_data, sample_rate)
    #generate preciction from the spectogram 
    prediction = predict_from_spectogram(img)
    #return the prediction 
    return labels[int(prediction)]


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Ship Size Classification From Hydrophone Data")

    gr.Markdown(
        "Upload an audio file (download audio files from our test set [here](). Our model will attempt to determine the size of the ship (small, medium, large) based on the audio sample."
    )

    # Upload audio file
    audio_input = gr.Audio(label="Upload Audio File", type="filepath")

    submit_button = gr.Button("Generate Prediction")

    output = gr.Textbox(label="Size Prediction")

    submit_button.click(classify_audio, inputs=audio_input, outputs=output)

# Run the app
if __name__ == "__main__":
    demo.launch()
