# MIAR


Dependencies
-
 *python >= 3.6  
 *Pytorch (1.0 <= version < 1.2)  
 *allennlp == 0.8.4  
`<pip install allennlp==0.8.4>`  

Parameters Configuration
-
MIAR/config.json is the config file. Some key json fields in config file are specified as follows：  
`<"train_data_path": train file path>`   
`<"validation_data_path": test file path>`   
`<"text_field_embedder": word embedding, including pre-trained file and dimension of embedding>`    
`<"pos_tag_embedding": pos-tag embedding>`   
`<"optimizer": optimizer, we use Adam here>`   
`<"num_epochs": training epochs>`   
`<"cuda_device": training with CPU or GPU>`   

Usage
-
  To train a new model, run the following command:  
  `<allennlp train <config file> -s <serialization path> -f --include-package MIAR>`  
  For example, with:  
  `<allennlp train MIAR/config.json -s MIAR/out -f --include-package MIAR>`  
  you can get the output folder at MIAR/out and log information showed on the console.  
