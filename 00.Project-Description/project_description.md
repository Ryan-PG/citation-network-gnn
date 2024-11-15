# Applying Graph Neural Networks (GNNs) for Node Classification in Citation Networks

## Project Title

**Node Classification in Citation Networks using Graph Neural Networks (GNNs)**

## Objective

The main goal of this project is to apply Graph Neural Networks (GNNs) to a citation network to classify academic papers based on their subject areas. Citation networks are represented as graphs where:

- **Nodes**: Represent academic papers.
- **Edges**: Represent citation relationships between papers.

Each paper (node) is linked to others that cite it or are cited by it. The objective is to use GNNs to predict the subject category of each paper based on the citation network structure and paper features.

---

## Datasets

The **Cora dataset** is commonly used for projects of this type and includes:

- **Nodes**: Papers.
- **Edges**: Citation links between papers.
- **Features**: A sparse bag-of-words representation for each paper.
- **Labels**: Subject areas (e.g., Machine Learning, Data Mining, Neural Networks, etc.).

The dataset can be sourced from libraries like **PyTorch Geometric** or **TensorFlow Graphs**.

---

## Project Steps

### 1. Introduction and Problem Definition

- Introduce the concept of citation networks and their significance in node classification.
- Define the goal of predicting paper subject areas based on connections in the citation graph.

### 2. Understanding the Dataset

- **Download and preprocess** the Cora dataset (or alternatives like PubMed or CiteSeer).
- **Perform exploratory data analysis (EDA)**:
  - Visualize the graph structure.
  - Analyze the distribution of nodes across categories.
  - Examine node features representing paper attributes.

### 3. Building the Graph Neural Network (GNN) Model

#### Model Architecture

- **GNN Type**: Use **Graph Convolutional Network (GCN)** or **GraphSAGE**.
- **Function**: The GNN performs message passing between nodes to learn node embeddings based on their neighbors.

#### Layers

- **Input Layer**: The feature vectors for each node (e.g., bag-of-words representation of papers).
- **GNN Layers**: Implement 2-3 layers of graph convolution or aggregation.
- **Output Layer**: A softmax layer for multi-class classification to predict each paper's category.

### 4. Training the Model

- **Dataset Split**: Divide data into training, validation, and test sets (e.g., 60% train, 20% validation, 20% test).
- **Training**: Use cross-entropy loss and **Adam optimizer**.
- **Validation**: Monitor performance on the validation set to prevent overfitting.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score.

### 5. Testing and Evaluation

- Evaluate the trained model on the test set to assess generalization.
- Compare the GNN model’s performance against baseline models (e.g., Logistic Regression or Random Forest using only node features).

### 6. Visualizing Results

- **Node Embeddings**: Use **t-SNE** or **PCA** to visualize how the GNN model clusters papers by subject.
- **Training Curves**: Plot accuracy and loss over epochs.

### 7. Extensions (into extensions folder in 01 directory)

- Experiment with other GNN architectures (e.g., **Graph Attention Networks** - GAT).
- Explore hyperparameters (e.g., number of layers, hidden layer sizes, learning rates).
- Test on other datasets, such as PubMed or CiteSeer.
- Add features like publication year or journal impact factor for enhanced modeling.

---

## Deliverables

1. **Code Implementation**:

   - A Python script or Jupyter Notebook implementing the GNN for node classification using libraries like **PyTorch Geometric** or **DGL**.

2. **Report**:

   - A comprehensive report detailing the dataset, methodology, GNN architecture, results, and additional experiments.

3. **Presentation**:
   - A concise presentation summarizing the project, including key results and visualizations.

## Tools and Libraries

- **Python**: Programming language.
- **PyTorch Geometric** or **DGL**: GNN implementation.
- **NetworkX**: For handling and visualizing graph structures.
- **Scikit-learn**: For preprocessing and evaluation metrics.
- **Matplotlib/Seaborn**: For visualizations and plotting.

---

## Outcome

By the end of the project, participants will:

- Understand how to apply GNNs to graph data for node classification.
- Gain experience with graph datasets and develop neural network models tailored to graph structures.
- Appreciate the strengths of GNNs in extracting insights from graph-structured data, a challenge for traditional machine learning models.
