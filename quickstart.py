import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the pretrained autovc model:
autovc = torch.hub.load('RF5/simple-autovc', 'autovc').to(device)
autovc.eval()
# Load the pretrained hifigan model:
hifigan = torch.hub.load('RF5/simple-autovc', 'hifigan').to(device)
hifigan.eval()
# Load speaker embedding model:
sse = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder').to(device)
sse.eval()



# Get mel spectrogram (source)
mel = autovc.mspec_from_file('test_wav/iknewyouweretrouble.wav') 
# or autovc.mspec_from_numpy(numpy array, sampling rate) if you have a numpy array

# Get embedding for source speaker
sse_src_mel = sse.melspec_from_file('test_wav/iknewyouweretrouble.wav')
with torch.no_grad(): 
    src_embedding = sse(sse_src_mel[None].to(device))
# Get embedding for target speaker
sse_trg_mel = sse.melspec_from_file('test_wav/taylor_readyforit.wav')
with torch.no_grad(): 
    trg_embedding = sse(sse_trg_mel[None].to(device))

# Do the actual voice conversion!
with torch.no_grad():
    spec_padded, len_pad = autovc.pad_mspec(mel)
    x_src = spec_padded.to(device)[None]
    s_src = src_embedding.to(device)
    s_trg = trg_embedding.to(device)
    x_identic, x_identic_psnt, _ = autovc(x_src, s_src, s_trg)
    if len_pad == 0: x_trg = x_identic_psnt[0, 0, :, :]
    else: x_trg = x_identic_psnt[0, 0, :-len_pad, :]

# x_trg is now the converted spectrogram!



# Make a vocode function
@torch.no_grad()
def vocode(spec):
    # denormalize mel-spectrogram
    spec = autovc.denormalize_mel(spec)
    _m = spec.T[None]
    waveform = hifigan(_m.to(device))[0]
    return waveform.squeeze()

converted_waveform = vocode(x_trg) # output waveform 
# Save waveform as wav file
import soundfile as sf
sf.write('converted_uttr.wav', converted_waveform.cpu().numpy(), 16000)