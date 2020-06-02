# Sequence Classification with Transformers
Code for fine-tuning transformers (XLNet and Bert) on binary, multi-class and multi-label sequence classification tasks.
The code uses the [Hugging Face implementations](https://github.com/huggingface/transformers/).

## Requirements
- numpy == 1.16.4
- pandas == 0.24.2
- regex == 2019.11.1
- scikit_learn == 0.20.3
- torch == 1.3.1
- tqdm == 4.32.1
- transformers == 2.1.1

## Usage
`mlmc_class.py [-h] --train_file TRAIN_FILE --eval_file EVAL_FILE --model MODEL [--bert_model BERT_MODEL] [--xlnet_model XLNET_MODEL]
                     [--train_batch_size TRAIN_BATCH_SIZE] [--eval_batch_size EVAL_BATCH_SIZE] [--learning_rate LEARNING_RATE]
                     [--num_train_epochs NUM_TRAIN_EPOCHS] [--prob_threshold PROB_THRESHOLD] [--max_seq_length MAX_SEQ_LENGTH]`

Where:  
Required:  
-   `TRAIN_FILE` is the path to the training tsv file with headers data (sentences) and labels (comma separated in case of multi-label classification). See samples.
-   `EVAL_FILE` is the path to the evaluation tsv file with headers data (sentences) and labels (comma separated in case of multi-label classification). See samples.
-   `MODEL` specifies the pre-trained transformer model to be used.  Possible values: 
        `bert`
        `xlnet`

Not required:  
-   `BERT_MODEL` specifies the BERT pre-trained model to be used. Possible values:  

        `bert-base-uncased`  
        `bert-large-uncased`  
        `bert-base-cased`  
        `bert-base-multilingual`  
        `bert-base-chinese`  
        
    The default value is `bert-base-uncased`  
-   `XLNET_MODEL` specifies the XLNet pre-trained model to be used. Possible values:  

        `xlnet-base-cased`  
        `xlnet-large-cased`  
        
    The default value is `xlnet-base-cased`
-	`TRAIN_BATCH_SIZE` is the training batch size.
    The default value is `32`
-	`EVAL_BATCH_SIZE` is the evaluation batch size.
    The default value is `32`
- `LEARNING_RATE` is the initial learning rate for Adam.
    The default value is `2e-5`
- `NUM_TRAIN_EPOCHS` specifies the number of training epochs to perform.
    The default value is `4`
- `PROB_THRESHOLD` is the probabilty threshold for multiabel classification.
    The default value is `0.5`
- `MAX_SEQ_LENGTH` is the maximum total input sequence length after WordPiece tokenization.
    The default value is `128`  
