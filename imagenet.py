import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained model
model = ResNet50(weights='imagenet')

# Category mapping (ImageNet classes to our categories)
category_map = {
    'person': ['person', 'man', 'woman', 'child', 'boy', 'girl',
                'face', 'head', 'suit', 'tie', 'hat', 'beard', 'bride', 'groom'],
    'animal': ['animal', 'dog', 'cat', 'bird', 'fish', 'bear', 'elephant', 
               'zebra', 'giraffe', 'horse', 'sheep', 'cow', 'wolf', 'fox'],
    'object': ['car', 'house', 'spoon', 'computer', 'phone', 'building', 
              'chair', 'table', 'knife', 'fork', 'cup', 'bottle', 'book']
}

def classify_image(img_path):
    # Load and preprocess image (same as before)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get predictions
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=5)[0]
    
    # Track best category based on probability
    best_category = None
    best_prob = 0.0
    specific_labels = []
    
    for _, label, prob in decoded:
        # Check person
        if any(p in label for p in category_map['person']):
            if prob > best_prob:
                best_category = 'person'
                best_prob = prob
                specific_labels.append(label)
                
        # Check animal (only if no stronger person prediction)
        if best_category != 'person' and any(a in label for a in category_map['animal']):
            if prob > best_prob:
                best_category = 'animal'
                best_prob = prob
                specific_labels.append(label)
                
        # Check object (only if no stronger person/animal prediction)
        if best_category not in ['person', 'animal'] and any(o in label for o in category_map['object']):
            if prob > best_prob:
                best_category = 'object'
                best_prob = prob
                specific_labels.append(label)
    
    # Default to object if no match
    if best_category is None:
        best_category = 'object'
        specific_labels = [decoded[0][1]]  # top prediction
    
    return {
        'primary_category': best_category,
        'specific_labels': specific_labels[:3],
        'all_predictions': decoded
    }

# Example usage
result = classify_image('imgs/man.jpg')
print(f"Primary Category: {result['primary_category']}")
print(f"Specific Labels: {', '.join(result['specific_labels'])}")
# print("All predictions:", result['all_predictions'])