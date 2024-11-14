# Node Classification in Citation Networks using Graph Neural Networks (GNNs)

This project applies Graph Neural Networks (GNNs) to classify academic papers within a citation network based on their subject areas. Using the Cora dataset, we leverage GNN architectures like Graph Convolutional Network (GCN) and GraphSAGE to make predictions by learning from the citation relationships between papers.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Extensions](#extensions)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project demonstrates the use of GNNs for node classification tasks, specifically focusing on predicting the subject area of academic papers in a citation network. Citation networks are modeled as graphs where nodes represent papers and edges represent citation relationships.

## Dataset

The project uses the **Cora dataset**, a popular dataset in graph-based research. It consists of:

- **Nodes**: Representing academic papers.
- **Edges**: Representing citation links between papers.
- **Features**: Sparse bag-of-words vectors for each paper.
- **Labels**: Subject areas such as Machine Learning, Data Mining, and Neural Networks.

The Cora dataset is available through libraries like PyTorch Geometric.

## Model Architecture

The project implements two main GNN architectures:

1. **Graph Convolutional Network (GCN)**
2. **GraphSAGE**

The GNN models follow a multi-layer structure:

- **Input Layer**: Initializes node features.
- **GNN Layers**: Performs message-passing operations to learn node embeddings.
- **Output Layer**: Classifies nodes into subject categories using a softmax function.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/gnn-node-classification.git
   cd gnn-node-classification
   ```

2. Install dependencies:
   - **PyTorch with CUDA** (for GPU support, replace `cu118` with your CUDA version):
     ```bash
     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
     ```
   - **PyTorch Geometric**:
     ```bash
     pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
     ```
   - Other required libraries:
     ```bash
     pip install scikit-learn matplotlib
     ```

## Usage

1. Run the training script:

   ```bash
   python main.py
   ```

2. Evaluation results, including accuracy, precision, recall, and F1-score, will be displayed in the console.

3. You can visualize training loss and node embeddings using the provided code.

## Results

The trained GNN model provides metrics like accuracy, precision, recall, and F1-score for node classification on the Cora dataset. Additionally, visualizations of node embeddings (e.g., using t-SNE or PCA) and training loss are included to assess model performance.

## Extensions

For further exploration, you can:

- Experiment with different GNN architectures, such as Graph Attention Networks (GAT).
- Tune hyperparameters like learning rate, number of layers, and hidden layer size.
- Use additional datasets like PubMed or CiteSeer.

## Project Structure

```
gnn-node-classification/
├── 00.Project-Description/
│   ├── Project.Description.docx          # Project overview document
├── 01.Dataset-And-Code/
│   ├── data/                             # Folder for storing the dataset
│   ├── main.py                           # Main script for training and evaluation
│   ├── models/                           # Folder containing model definitions
├── 02.Reports-And-Presentation/
│   ├── report.pdf                        # Detailed project report
│   ├── presentation.pdf                  # Project presentation slides
├── .gitignore                            # Git ignore file
├── README.md                             # Project README file
└── requirements.txt                      # List of dependencies
```

## Contributing

Contributions are welcome! Please submit issues or pull requests for any improvements or new features.

## License

This project is licensed under the MIT License.
