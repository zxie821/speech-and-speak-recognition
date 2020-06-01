# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------
import numpy as np
import scipy
from scipy.signal import lfilter
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from lab1_tools import *

import matplotlib.pyplot as plt
#example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

def plot_p_color_mesh(d2Matrix, caption):
    fig = plt.figure(figsize=(12,6))
    ax = plt.subplot(121)
    ax.set_title(caption)
    plt.pcolormesh(d2Matrix)
    plt.colorbar()
    plt.show()
    
    
def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, 
         nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)
def mfcc_unlifter(samples, winlen = 400, winshift = 200, 
                 preempcoeff=0.97, nfft=512, nceps=13, 
                 samplingrate=20000, liftercoeff=22):
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    return cepstrum(mspecs, nceps)

def test_mfcc():
    in_samples = example['samples']
    in_mfcc = mfcc(in_samples)
    in_mfcc_unlifter = mfcc_unlifter(in_samples)
    plot_p_color_mesh(in_mfcc, "mfcc() liftered")
    plot_p_color_mesh(in_mfcc_unlifter, "mfcc() unliftered")
    
    plot_p_color_mesh(example['mfcc'], "example mfcc")
    plot_p_color_mesh(example['lmfcc'], "example lmfcc")
# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    N = len(samples)//winshift - 1
    frames = np.zeros((N, winlen))
    for i in range(0, N):
        frames[i,:] = samples[i*winshift:i*winshift+winlen]
    return frames

def test_enframe():
    in_samples = example['samples']
    
    test_enframe = enframe(in_samples, 400, 200)
    plot_p_color_mesh(test_enframe, "enframe() output")
    example_enframe = example['frames']
    plot_p_color_mesh(example_enframe, "example frames")
    
#test_enframe()
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    a = np.array([1])
    b = np.array([1, -p])
    return lfilter(b, a, input, axis=-1)
def test_preemp():
    in_samples = example['samples']
    
    test_enframe = enframe(in_samples, 400, 200)
    preemph = preemp(test_enframe, 0.97)
    
    plot_p_color_mesh(preemph, "preemph() output")
    example_preemph = example['preemph']
    plot_p_color_mesh(example_preemph, "example preemph")
#test_preemp()
def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    hamming_win = scipy.signal.hamming(input.shape[1], sym=0)
    #plt.plot(hamming_win)
    #plt.ylabel('hamming window')
    #plt.show()
    result = np.zeros_like(input)
    for i in range(input.shape[0]):
        result[i] = input[i] * hamming_win
    return result
def test_windowing():
    in_samples = example['samples']
    
    test_enframe = enframe(in_samples, 400, 200)
    preemph = preemp(test_enframe, 0.97)
    windowed = windowing(preemph)
    plot_p_color_mesh(windowed, "windowing() output")
    example_windowed = example['windowed']
    plot_p_color_mesh(example_windowed, "example windowed result")
#test_windowing()
def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    fft_result = fft(input, nfft)
    return np.power(np.abs(fft_result),2)
def test_fft():
    in_samples = example['samples']
    test_enframe = enframe(in_samples, 400, 200)
    preemph = preemp(test_enframe, 0.97)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, 512)
    plot_p_color_mesh(spec, "powerSpectrum() output")
    example_spec = example['spec']
    plot_p_color_mesh(example_spec, "example spec result")
#test_fft()
def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    N, nfft = input.shape
    filter_bank = trfbank(samplingrate, nfft)
    nmelfilters = filter_bank.shape[0]
    result = np.zeros((N, nmelfilters))
    for n in range(N):
        for nf in range(nmelfilters):
            result[n, nf] = np.log(np.sum(input[n] * filter_bank[nf]))
    return result
def test_logMS():
    in_samples = example['samples']
    test_enframe = enframe(in_samples, 400, 200)
    preemph = preemp(test_enframe, 0.97)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, 512)
    logms = logMelSpectrum(spec, 20000)
    
    filter_bank = trfbank(20000, 512)
    plot_p_color_mesh(filter_bank, 'filter Bank')
    plot_p_color_mesh(logms, 'logMeSpectrum() output')
    plot_p_color_mesh(example['mspec'], 'example output')
    
#test_logMS()

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    return dct(input)[:,:nceps]

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    N, D = x.shape
    M, _ = y.shape        
    accumulated_dist = np.zeros((N, M))
    def get_acc(i_idx, j_idx):
        if i_idx<0 or j_idx<0:
            return 0
        return accumulated_dist[i_idx, j_idx]
    for i in range(N):
        for j in range(M):
            min_dp = np.min([get_acc(i-1, j), get_acc(i, j-1),
                            get_acc(i-1, j - 1)])
            accumulated_dist[i, j] = min_dp + dist(x[i], y[j])
    global_distance = accumulated_dist[-1, -1] /(M+ N)
    return global_distance, _, accumulated_dist,_
#test_mfcc()
