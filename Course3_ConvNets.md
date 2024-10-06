- Computer Vision(CV) → Image Classification, Object Detection, Neural Style Transer, Artistic Applications like Image Generation
- **Input Image Example:**
    - Image dimensions: 1000x1000 pixels (RGB image)
    - Total input vector: $1000 \times 1000 \times 3 = 3 \text{ million}$
- **Edge Detection:**
    - **Filter (Kernel)**: Small matrix applied to the image to detect features.
        - Example filter size: $3\times3$, applied to a region of the image.

- **Vertical Edge Detector (3x3 Matrix):**
    
    
    | -1 | 0 | 1 |
    | --- | --- | --- |
    | -1 | 0 | 1 |
    | -1 | 0 | 1 |
    - Detects vertical edges by highlighting differences between adjacent columns.

- **Horizontal Edge Detector (3x3 Matrix):**
    
    
    | 1 | 1 | 1 |
    | --- | --- | --- |
    | 0 | 0 | 0 |
    | -1 | -1 | -1 |
    - Detects horizontal edges by comparing differences between adjacent rows.
- **Filter Application:**
    - Apply a $3\times3$ filter over a region (e.g., $3\times3$) section) of the image for convolution.
- **Common Filters:**
    - **Sobel Filter** (used for edge detection in both x and y directions):
        
        
        | -1 | 0 | 1 |
        | --- | --- | --- |
        | -2 | 0 | 2 |
        | -1 | 0 | 1 |
    - **Scharr Filter** (more sensitive than Sobel filter):
        
        
        | -3 | 0 | 3 |
        | --- | --- | --- |
        | -10 | 0 | 10 |
        | -3 | 0 | 3 |
- **Learning Filters:**
    - Instead of manually choosing filters, CNNs learn the filter values (weights) during training. Example of a learned $3\times3$ filter matrix with weights **w_1** to **w_9**:
        
        
        | w₁ | w₂ | w₃ |
        | --- | --- | --- |
        | w₄ | w₅ | w₆ |
        | w₇ | w₈ | w₉ |
- **Convolution Process:**
    - Use **conv2D** operation to apply filters across the entire image.
    - In Python, libraries like TensorFlow/Keras provide functions such as `conv_forward` for convolutional layers.

---

### **Padding in Convolutional Neural Networks**

- **Convolution Example:**
    - Input matrix: $6\times6$
    - Filter (Kernel): $3\times3$
    - Output size equation: $(n - f + 1) \times (n - f + 1)$
        - Where `n` is the input size, `f` is the filter size.
        - Example: $6 - 3 + 1 = 4 \times 4$ output.
    - **Striking Output:**
        - Reducing the output size as the input shrinks after convolution (throwing away borders).
- **Padding the Image:**
    - Surround the image with additional layers (border) to preserve size after convolution.
        - Example:
            - Original image: $6\times6$
            - After padding: $8\times8$ (with (p = 1))
            - Padding amount (p = 1), meaning a 1-pixel border added around the image.
    - **Output Size with Padding: $(n + 2p - f + 1) \times (n + 2p - f + 1)$**
        - Where `p` is the padding amount.
- **Types of Convolutions:**
    - **Valid Convolution:** No padding, the output size reduces.
    - **Same Convolution:** Padding is applied so that the output size remains the same as the input size.
    - **Filter Sizes:** Filters `f` are usually odd, like 3x3, 5x5 or 7x7

---

### **Strided Convolution:**

- **Stride = 2:**
    - Moves the filter by 2 pixels (instead of the default 1) during convolution.
    - This reduces the output size but makes computations faster and the model more efficient.
- **Cross-Correlation:**
    - **Technical Note:** In theory, we do not flip the filter when applying it (which is cross-correlation, not convolution), but in **machine learning**, it's conventionally referred to as **convolution**.

---

### **Convolutions over Volumes (RGB Images)**

- **Convolution on RGB Images:**
    - `TODO: add drawings of volumes when sharing`
    - Input: $6\times6\times3$  $Height \ \times \ Width \ \times \ Channels$
        - RGB images have 3 channels (Red, Green, Blue).
    - Filter size: $3\times3$ (for each channel).
        - **Channels should match** between the input and the filter for convolution to work.
    - Output size: $4 \times 4$, computed as: $(n - f + 1) \times (n - f + 1)$
        - Example: $6 - 3 + 1 = 4 \times 4$.
- **Multiple Filters:**
    - Convolution with multiple filters can detect multiple features.
    - If using `N_c` filters, output size becomes: $(n - f + 1) \times (n - f + 1) \times N_c$
        - Example: $(6 \times 6 \times 3$ input convolved with \(3 \times 3\) filter results in \(4 \times 4 \times 2\) output if 2 filters are used.
- **Importance of Multiple Filters:**
    - Allows detection of various features in different parts of the image (e.g., edges, textures).

---

### **One Layer of Convolutional Neural Networks (CNNs)**

- **Process:**
    - The input volume (e.g., $6\times6\times3$ ) is convolved with a filter (e.g., $3\times3\times3$ ), producing a smaller output volume.
    - For each filter, after convolution, a bias term $b_1$ is added, and a non-linear activation function (e.g., ReLU) is applied: $(4 \times 4) + b_1 \quad \text{(activation)}$
    - Repeated for additional filters, e.g., another convolution producing: $(4 \times 4) + b_2 \quad \text{(activation)}$
    - After applying both filters, the output becomes a $4 \times 4 \times 2$ volume (2 feature maps).

---

### **Parameters in a CNN Layer:**

- **How Many Parameters in One Layer?**
    - For a filter of size $3\times3\times3$  (considering 3 channels for RGB):
    $3 \times 3 \times 3 = 27 \text{ (weights)} + 1 \text{ (bias)} = 28 \text{ parameters per filter.}$
    - If there are 10 filters in this layer:
    $28 \times 10 = 280 \text{ parameters in total.}$

---