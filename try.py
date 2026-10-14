import numpy as np

def generate_bpsk_signals(n_samples):
    bits = np.random.randint(0, 2, n_samples)
    return 2 * bits - 1   

def add_noise(signal, snr_db):
    snr = 10**(snr_db/10)
    noise = np.random.normal(0, 1/np.sqrt(snr), size=signal.shape)
    return signal + noise

 
x_real = generate_bpsk_signals(1000)
x_real_noisy = add_noise(x_real, snr_db=10)

 