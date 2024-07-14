# Neural-Network-and-Deep-Learning
## Files in the Repository
### Implement NN From Scratch
- Implemented the forward and backward function for ReLU, LinearMap, SoftmaxCrossEntropyLoss
- Used the functions to construct a single layer NN

### Predict Malaria with CNN
Implemented a convolution neural network-based machine learning model to predict whether the cell in the given picture is parasitized by the genus Plasmodium or not.
1. Transformed the image dataset from a singular folder to a hierarchical structure based on labels.
2. Identified and removed outliers using **_Isolation Forest detection_**
3. Used a pre-trained **_ResNet50 model_** for feature extraction to enhance training data preprocessing.
4. Normalized data by mean and standard deviation versus scaling by 1/255
5. Applied **_image trasformations_** to augment the data
6. Fine-tuned image properties such as contrast, saturation, and brightness
7. Built a CNN with layers including concolutional layers, max pooling layers, a flatten layer, and dense layers
8. Experimented with various optimizers (**_Adamax, SGD, Adadelta, Adagrad_**) and identified SGD as the most effective.
9. Added **_L2 regularization_** to prevent overfitting and improve generalization.
10. Increased the size of dense layers to enhance the model's capacity.
11. Evaluated model performance using validation accuracy and validation loss.
12. Predicted labels for the normalized test dataset and prepared final submission by exporting predictions to a CSV file.

### Predict Protein Functions using Transformer models (HF)
**Researched and Used Pretrained Models:**
- Established Protein BERT as the baseline model.
- Utilized other models from Hugging Face to surpass the baseline accuracy.

**Fine-tuned the Model:**
- Added a linear layer mapping from 1024 to 512 features.
- Applied a **_ReLU activation_** function for non-linear processing.
- Included a **_Dropout layer_** with a dropout rate of 0.5 to reduce overfitting.
- Experimented with **_learning rates_** within the range of 0.001 to 0.1.
- Tested multiple optimizers including **_SGD, Adam, and AdamW_**.

**Model Enhancement and Selection:**
- Selected 'facebook/esm-1b' from Hugging Face, a transformer protein language model trained on protein sequence data without label supervision.
- Utilized pretrained weights from **_Uniref50_** with an unsupervised masked language modeling (MLM) objective.

**Alternative Approaches:**
- Attempted to implement the **_ProtENN + HMMER model_** based on a paper from Kaggle, achieving competitive results.

### Molecular Property Prediction using Graph Neural Network
Implemented a Graph Neural Network (GNN) using PyTorch Geometric (PyG) to predict molecular properties from graph data structures. The goal is to train the model on a provided dataset and accurately predict a target property for new molecules.

- Utilized **_SMILES augmentation_** and molecular graph augmentation techniques to enhance the diversity and robustness of the training data.
- Experimented with various GNN architectures, including: **_Graph Convolutional Network (GCN)_**, **_Graph Attention Network (GAT)_**, **_GCN + MLP_**(Combined GCN with a Multi-Layer Perceptron (MLP) for improved learning capacity), **_LEC+GAT_** (Integrated Local Edge Convolution with GAT for leveraging edge features), and **_NET_**.
- Tained and Evaluated models using cross-validation and metrics such as **_Mean Absolute Error (MAE)_** and **_Root Mean Squared Error (RMSE)_**.
