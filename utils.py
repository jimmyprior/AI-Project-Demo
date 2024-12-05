import librosa 
import numpy as np 

def stft(filepath, duration : float = None):
    #modified version of code from presenter
    #using this over mel spectogram because applying mel scale should not matter
    # since human perception of the sound is not relevant in this case.
    y, sr = librosa.load(filepath, duration=duration)
    result = np.abs(librosa.stft(y, hop_length=256)) #what
    result_db = librosa.amplitude_to_db(result, ref=np.max)
    return result_db

def mel_spectrogram(y, sr):
    #experiment with these values
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512, hop_length=256)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def rescale_minmax(arr):
    #taken from the presenter vode
    """Rescales the values of a NumPy array between 0 and 1."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    if arr_max - arr_min == 0:  # Handle the case where all values are the same
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - arr_min) / (arr_max - arr_min)
    


def extract_audio_segment(audio, sample_rate, start_time, end_time):
    """
    Extracts a segment of audio from the given librosa audio sample.

    Parameters:
        audio (numpy.ndarray): The audio time series (1D or 2D array).
        sample_rate (int): The sample rate of the audio.
        start_time (float): The start time of the segment in seconds.
        end_time (float): The end time of the segment in seconds.

    Returns:
        numpy.ndarray: The extracted audio segment.
    """
    # Convert start and end times to sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Ensure indices are within bounds
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    # Extract and return the segment
    return audio[start_sample:end_sample]
