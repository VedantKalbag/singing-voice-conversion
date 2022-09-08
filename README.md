# Singing Voice Conversion

This project aims at creating a zero shot system for singing voice conversion through disentanglement of content and style.

The inputs to the system are the source audio and target audio, from which mel-spectrograms are extracted along with the analysis output of the WORLD vocoder.
  
  The output of the WORLD vocoder include:
1. Spectral Envelope
2. Aperiodicity Envelope
3. Fundamental frequency (f0) map

- Here a pre-trained style encoder (input- mel spectrogram) is used to extract style embeddings, and a content encoder (input- spectral envelope) is trained to extract content embeddings. 
- These embeddings are concatenated in the latent space and a decoder is trained to reconstruct the morphed spectral envelope.
- The content encoder and the decoder are trained using a self-reconstruction loss
