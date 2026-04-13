# nlp-hw4
# Nithin Gundapu
# 700772575
README explanation (Q1)
This script trains a character-level LSTM language model on a toy text corpus.

Pipeline: text → character indices → embedding → LSTM → linear → softmax.

Training uses teacher forcing and cross-entropy loss with Adam.

train_losses and val_losses can be plotted to show learning curves.

sample() generates text with different temperatures:

Lower τ → more conservative, repetitive text.

Higher τ → more diverse, but also more errors.

You can change sequence length and hidden size in train_model() to observe:

Longer sequences → better long-range patterns but slower training.

Larger hidden size → more capacity but more overfitting risk.


README explanation (Q2)
This script builds a mini Transformer encoder for a batch of 10 toy sentences.

Steps:

Build a vocabulary, tokenize sentences, and pad to equal length.

Map tokens to embeddings, then add sinusoidal positional encodings.

Pass through a Transformer encoder block:

Multi-head self-attention

Add & Norm

Feed-forward

Add & Norm

The script prints:

Input token IDs.

Shape of final contextual embeddings.

An attention matrix (heatmap values) for sentence 0, head 0.

This shows how each word attends to others in the same sentence.

README explanation (Q3)
	This script implements scaled dot-product attention:
"Attention"(Q,K,V)="softmax" ((QK^⊤)/√(d_k ))V
	It:
	Creates random Q, K, V tensors.
	Prints:
	Raw attention scores QK^T.
	Scaled scores QK^T / sqrt(d_k).
	Softmax attention weights.
	Final output vectors.
	The scaling by sqrt(d_k) improves softmax stability by keeping scores in a reasonable range.

