# Researchers Q&A

### What advancements does the project propose in the field of graph transformers for biomedical applications?

This project introduces the Temporal Attention Graph Transformer (TAGT) designed to predict SLE flares. It combines multi-modal genomic and clinical data, utilizes graph attention networks, and focuses on biological interpretability through embedding node interactions in a protein-protein interaction network context.

### How does the project address class imbalance in biomedical datasets?

The TAGT model employs Focal Loss with adaptive weighting and gradient accumulation to effectively address class imbalance, ensuring balanced predictive performance for all classes.

### What are the innovative training techniques employed?

Innovations include mixed precision training, gradient accumulation, cosine learning rate scheduling, and pathway-aware attention for biological interpretability. These techniques leverage advanced GPU capabilities, optimizing training speed and memory usage.

### How is biological interpretability achieved in the model?

Through Pathway-aware attention mechanisms, the model emphasizes gene clusters significance, treating them as biological pathways, thus enhancing interpretability and biological relevance in clinical data applications.
