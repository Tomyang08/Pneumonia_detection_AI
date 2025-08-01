**Pneumonia Detection with Deep Learning**  
*Automating pneumonia diagnosis using convolutional neural networks to support accessible healthcare.*  


### Project Overview  
Pneumonia causes over 700,000 annual deaths in children under 5, primarily in low-resource regions where diagnostic tools are scarce. This project develops an AI system to detect pneumonia from chest X-rays using deep learning, aiming to:  
- Reduce diagnostic delays
- Lower healthcare costs
- Improve accessibility in underserved communities 

**Key Details:**  
- Dataset: 2,400 grayscale X-rays from Mendeley (50% pneumonia, 50% healthy)  
- Models: CNN (83.5% accuracy), VGG16 (85% accuracy), benchmarked against KNN(75%)/Logistic Regression(67%)/Decision Trees(66%)
- Deployment: Streamlit web app

**Key Challenges & Technical Learnings**
1. Real-World Generalization Gap

Problem:

Model achieved 85% accuracy on lab data but struggled on real-world field data. This was likely due to rotated patient orientations, inconsistent brightness, and sensor noise in field X-rays (factors rarely present in controlled datasets).

### Technical Approach:
python
# Field data simulation in preprocessing
def get_field_data(flatten, all_data, metadata, shape):
    data, labels = get_data_split('field', flatten, all_data, metadata, shape)
    # Simulate real-world variability
    rand = random.uniform(-1, 1)
    for i in range(len(data)):
        if abs(rand) < 0.5:
            data[i] = rotate(data[i], rand * 40)  # Random rotation [-40°, 40°]
        else:
            data[i] = shear(data[i], rand * 40)   # Random shear
    return data, labels

### My key takeaways:

1.Augmentation limitations: Standard rotations (±15°) couldn't cover extreme field variations
2.Clinical variability: Field environments introduced unpredictable artifacts and orientations
3.What I did : Implemented dynamic field-data simulation during training


**Preprocessing Pipeline:**  
1. **Augmentation:**  
   - Rotation (±15°)  
   - Horizontal/Vertical flipping  
   - Shearing (0.2 rad)  
   - Scaling (0.8–1.2x)  
2. **Normalization:**  
   - Grayscale conversion  
   - Pixel value standardization  
3. **Train-Test Split:**  
   - 80% training (1,920 images)  
   - 20% testing (480 images)  

---

### Model Architecture  
python
# Custom CNN architecture
def CNNClassifier(n_layers, params):
    model = Sequential()
    model.add(Input(params['input_shape']))
    # Feature extraction blocks
    for _ in range(n_layers):
        model.add(Conv2D(64, 3, padding='same', kernel_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2))
    # Classification head
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=params['loss'], optimizer='adam', metrics=['accuracy'])
    return model
**CNN Architecture:**  
```  
Input (256×256 grayscale)  
↓  
Convolutional Layers (ReLU activation)  
↓  
Max Pooling  
↓  
Flatten → Dense Layers (Dropout regularization)  
↓  
Output (Sigmoid activation)  
```  

### My learnings:

Transfer learning superiority: VGG16 outperformed custom CNN by 1.5%
Regularization necessity: L2 regularization prevented overfitting
Spatial vs. flattened features: CNNs outperformed MLPs by 33% accuracy


### Deployment Pipeline

Technical Workflow:

python
# Streamlit deployment core
def process_image(image):
    image = image.convert('RGB').resize((64, 64))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

uploaded_file = st.file_uploader("Upload X-ray")
if uploaded_file:
    img = Image.open(uploaded_file)
    processed_img = process_image(img)
    pred = model.predict(processed_img)[0][0]
    diagnosis = "Pneumonia" if pred > 0.5 else "Normal"
    st.image(img, caption=f"Diagnosis: {diagnosis}")
 
 


### Future Work  
1. **Robustness Enhancements:**  
   - Develop preprocessing module for orientation/brightness normalization  
   - Train on diverse field data from partner clinics  
2. **Model Optimization:**  
   - Implement EfficientNet for parameter efficiency  
   - Integrate Grad-CAM for explainable diagnoses  
3. **Deployment:**  
   - Build lightweight iOS/Android app for rural health workers  
   - API integration with portable digital X-ray devices  



### Acknowledgments  
- Dataset: Mendeley Data & NIH dataset contributors
- Instructor: Imtisaal Mian  
- Team: Anishka, Arianna, Justin, Tom, Natalie, Iman
- AI, for not giving up when our accuracy dropped below 70%


### This project revealed the gap between controlled experiments and real-world medical applications. While we achieved 85% lab accuracy, field data challenges underscored the importance of adaptable, human-centered AI. Moving forward, I plan to collaborate with clinicians to bridge this gap—combining technical rigor with practical healthcare needs to build tools that truly save lives.*  

### For more details, refer to the presentation: https://docs.google.com/presentation/d/1OaJgCPJfzF4YikmzMSpG2Pfbn75k7XLSnv9HD_uAxpg/edit?usp=sharing
