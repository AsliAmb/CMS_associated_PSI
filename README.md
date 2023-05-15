# Introduction:

The classifiers that use exon-level data can outperform classifiers that use gene-expression data, however, the feature space for alternative splicing events is significantly larger than the gene-level quantification of genes. To identify exon-skipping events that can distinguish CMS and are robust to variations in the input data, here we implemented the method described by Peter Bulhmann et al [1]. 

1. Meinshausen N, Buhlmann P. Stability selection. J R Stat Soc B. 2010;72:417-73


# Installation:  

Clone the CMS_associated_PSI repository and change current directory to CMS_associated_PSI/src  

In your terminal run the following command to build and install the package:  
```python setup.py install```

# Getting Started:  

Navigate to notebooks folder to run through Finding stable PSI events.ipynb that shows a typical workflow for finding stable PSI associated with CMS. 


