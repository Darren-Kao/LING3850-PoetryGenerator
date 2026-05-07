# LING3850-PoetryGenerator
LING 3850 Final Project: comparing the effectiveness of an RNN and a Transformer model on poetry generation

## Model 1: Fine-tuned RWKV-7 World 0.4B
Instructions on use: Add data/quatrains_fixed.txt to the FineTuneRNN.ipynb Google Colab. Run all cells in order, changing the prompts in the final cell to whatever prompts are desired. Alternatively, load the fine-tuned model from the rnn/rnn-final-checkpoint folder, and evaluate on that model.

## Model 2: Fine-tuned GPT2-Medium
Instructions on use: Download and expand an already fine-tuned model from google drive (https://drive.google.com/file/d/14YJ74vHDQ5_hTV_oB1n8BeNbVLc0mOZ1/view?usp=sharing), then run generate.py to generate a sonnet (change "./model" to where the saved pre-trained model is).
Alternatively, retrain and run the model in google colab (make sure runtime type is set to GPU, and you have downloaded the .txt data file): https://colab.research.google.com/drive/15a3qbjEkBK8yHRuymgTNvg7SaO23JUyL?usp=sharing.
