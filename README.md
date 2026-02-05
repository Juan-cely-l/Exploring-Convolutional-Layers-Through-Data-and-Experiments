# -Exploring-Convolutional-Layers-Through-Data-and-Experiments
In this course, neural networks are not treated as black boxes but as architectural components whose design choices affect performance, scalability, and interpretability. This assignment focuses on convolutional layers as a concrete example of how inductive bias is introduced into learning systems.

## Dataset Selection (CIFAR-10)
CIFAR-10 is used as the target dataset for image classification. It is a canonical benchmark in computer vision, designed to evaluate models that learn to map small RGB images to one of ten object categories. The dataset is compact enough to fit in memory and is widely adopted for controlled experimentation with convolutional architectures.

Key characteristics:
- Image classification with 10 discrete classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- RGB images with spatial resolution 32x32, enabling convolutional feature learning while remaining computationally efficient.
- Well-established benchmark with standardized splits, supporting reproducibility and comparison across models.

## Exploratory Data Analysis (EDA)
The EDA was designed to validate data integrity and inform preprocessing decisions before model training.

Normalization:
- After normalization, pixel values are in the range [0.0, 1.0] and stored as float32, ensuring numerical stability during optimization.

Dataset size and structure:
- Training set shape is (50,000, 32, 32, 3), and labels are stored as integer class indices.

Class distribution:
- The training set is perfectly balanced, with 5,000 samples per class (10% each), reducing the risk of class-imbalance bias.

Qualitative inspection:
- Representative samples from each class are inspected to verify label consistency, intra-class variability, and the absence of obvious artifacts.
