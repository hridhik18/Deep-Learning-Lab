
# **Concepts Used in Deep Learning Assignment**

## **1. Convolutional Neural Networks (CNNs)**
Convolutional Neural Networks (CNNs) are a class of deep learning models designed specifically for processing structured grid-like data, such as images. Unlike fully connected neural networks, CNNs leverage spatial hierarchies in data by using specialized layers. They consist of three main layers:

- **Convolutional Layers**: Extract spatial features using filters/kernels.
- **Pooling Layers**: Reduce dimensionality while preserving important features.
- **Fully Connected Layers**: Map extracted features to output classes.

### **Applications of CNNs:**
- **Image Classification**: Recognizing objects in images (e.g., Cats vs. Dogs classification).
- **Object Detection**: Identifying multiple objects within an image.
- **Medical Imaging**: Detecting diseases in MRI and X-ray images.
- **Autonomous Vehicles**: Identifying pedestrians, roads, and traffic signs.

---

## **2. Weight Initialization Techniques**
Weight initialization plays a crucial role in training deep neural networks. Poor initialization can lead to problems such as **vanishing gradients**, **exploding gradients**, or **slow convergence**. A well-initialized network ensures stable gradient flow and faster training.

### **Common Weight Initialization Techniques:**
- **Zero Initialization**: All weights are set to zero.
- **Random Initialization (Uniform/Normal Distribution)**: Weights are initialized randomly from a normal or uniform distribution.
- **Xavier/Glorot Initialization**: Suggested for activation functions like `tanh` and `sigmoid` as it balances variance across layers by setting weights.
- **He Initialization**: Designed for `ReLU` activations and helps deeper networks avoid vanishing gradients.

### **Use Cases:**
- **Xavier Initialization**: Used in shallow networks with `tanh` or `sigmoid`.
- **He Initialization**: Preferred for deep networks with `ReLU`.

---

## **3. Activation Functions**
Activation functions introduce **non-linearity** to neural networks, enabling them to learn complex patterns. Common activation functions include:

- **ReLU (Rectified Linear Unit)**: Most commonly used in CNNs due to efficient gradient propagation.
- **Sigmoid**: Maps values between `0` and `1`, mainly used in binary classification.
- **Tanh**: Similar to `sigmoid` but ranges from `-1` to `1`, reducing bias shift.

### **Use Cases:**
- **Sigmoid/Tanh**: Used in the output layer of binary classification.
- **ReLU/Leaky ReLU**: Used in hidden layers for deep learning models.

---

## **4. Optimizers**
Optimizers adjust a model’s weights and biases to minimize the **loss function** by updating parameters in a way that reduces error with each iteration.

### **Common Optimizers:**
- **SGD (Stochastic Gradient Descent)**: Basic optimizer with momentum-based improvements.
- **Adam (Adaptive Moment Estimation)**: Combines momentum and adaptive learning rates for better convergence.
- **RMSprop**: Adapts learning rates based on gradient magnitude, suitable for recurrent networks.

---

## **5. ResNet (Residual Networks)**
ResNet-18 is a deep convolutional network designed to overcome **vanishing gradient problems** using **skip connections (residual learning)**. It enables training very deep networks without performance degradation, making it highly effective for image classification tasks.

### **Key Features of ResNet:**
- **Skip Connections**: Allow gradients to flow directly to earlier layers, preventing vanishing gradients.
- **Deep Network Training**: Enables models to be trained with hundreds of layers.
- **State-of-the-Art Performance**: Achieves high accuracy in image classification benchmarks like ImageNet.

---
