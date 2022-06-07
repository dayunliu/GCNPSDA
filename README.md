# GCNPSDA
GCNPSDA is a computational framework which is used to predict snoRNA and disease associations.It is described in detail in our paper "GCNPSDA: a graph neural network based method for predicting snoRNA-disease associations".

## Requirements
* TensorFlow 1.15
* python 3.7
* numpy 1.19
* pandas 1.1
* scikit-learn 1.0
* scipy 1.5


## Data
In this study, we extracted the dataset from MNDR v3.0 and ncRPheno. MNDR v3.0 contains 1,007,831 ncRNA disease associations involving five different RNAs, including 1,596 snoRNA-disease associations. We filtrated these 1,596 associations and finally obtained 1,441 non-redundant associations between 453 snoRNAs and 119 diseases; ncRPheno retrieved and integrated multiple widely used ncRNA-disease association databases. It contains as many as 482,751 non-redundant associations between 14,494
ncRNAs and 3,210 disease phenotypes, covering most disease subtypes. We downloaded the 584 snoRNA- disease associations from ncRPheno and removed duplicate data. Eventually, we obtained 362 non-redundant associations between 6 snoRNAs and 119 diseases from ncRPheno. In the end, we obtained 1538 snoRNA-disease associations based on 456 snoRNAs and 194 diseases as the dataset for our experiments. For convenience, we constructed these 1538 snoRNA-disease associations as a binary matrix SD,
which consists of 456 rows and 194 columns. If there is an association between a certain snoRNA and a certain disease, then the value of this element at the corresponding position of matrix SD will be set as 1. Otherwise, it will be 0.
## Run the demo

```bash
python main.py
```
