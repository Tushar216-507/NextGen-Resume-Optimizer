"""
Custom Neural Network Model Trainer for Resume-Specific Typo Detection
This trains a model from scratch using TensorFlow/Keras for university project requirements.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import json
import re
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

print("🎓 Custom Resume Typo Detection Model Trainer")
print("=" * 60)
print("Training a Neural Network from scratch for university project")
print("=" * 60)

class ResumeTypoDataGenerator:
    """Generates training data for resume-specific typo detection"""
    
    def __init__(self):
        # Common IT/Resume words that are often misspelled
        self.correct_words = [
            # Programming Languages
            "python", "javascript", "java", "typescript", "csharp", "cplusplus",
            "html", "css", "php", "ruby", "go", "rust", "kotlin", "swift",
            
            # Frameworks & Libraries
            "react", "angular", "vue", "nodejs", "express", "django", "flask",
            "spring", "laravel", "bootstrap", "jquery", "tensorflow", "pytorch",
            
            # Technologies & Tools
            "docker", "kubernetes", "jenkins", "git", "github", "gitlab",
            "aws", "azure", "gcp", "terraform", "ansible", "nginx", "apache",
            
            # Databases
            "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "sqlite",
            
            # Concepts
            "algorithm", "database", "framework", "development", "programming",
            "experience", "project", "implementation", "optimization", "architecture",
            "microservices", "api", "frontend", "backend", "fullstack",
            
            # Resume Words
            "responsible", "managed", "developed", "implemented", "designed",
            "collaborated", "achieved", "improved", "created", "maintained"
        ]
        
        # Common typo patterns for each word
        self.typo_patterns = {
            "python": ["phyton", "pyhton", "pythn", "pythom"],
            "javascript": ["javascrip", "javasript", "javscript", "javascritp"],
            "react": ["reac", "reactt", "raect", "rect"],
            "angular": ["angualr", "anglar", "angulr", "anguar"],
            "docker": ["doker", "dokcer", "docekr", "dockr"],
            "kubernetes": ["kubernets", "kuberntes", "kuberentes", "kubrnetes"],
            "database": ["databse", "databas", "datbase", "dataabse"],
            "algorithm": ["algoritm", "algorith", "algorthm", "algorithem"],
            "development": ["developement", "developmnt", "devlopment", "developent"],
            "programming": ["programing", "programmin", "progamming", "programmng"],
            "experience": ["experence", "experince", "expereince", "experienc"],
            "implementation": ["implementaton", "implmentation", "implementaion", "implemntation"],
            "architecture": ["architecure", "architectur", "architecutre", "architeture"],
            "microservices": ["microservises", "microservice", "micro-services", "microservces"],
            "framework": ["framwork", "frameowrk", "framewrok", "framewok"],
            "frontend": ["front-end", "frontned", "frontent", "fronend"],
            "backend": ["back-end", "backned", "bakend", "bacend"],
            "responsible": ["responsable", "responible", "responsibile", "responisble"],
            "managed": ["managd", "mangaed", "manageed", "manged"],
            "developed": ["developd", "devloped", "developped", "develped"],
            "implemented": ["implementd", "implemeted", "implementted", "implemnted"],
            "collaborated": ["colaborated", "collabrated", "colaborted", "collabotated"],
            "optimization": ["optimizaton", "optimisation", "optimiztion", "optmization"],
            "mysql": ["mysq", "mysl", "mysqll", "mysqlq"],
            "postgresql": ["postgresq", "postgresl", "postgre", "postgresql"],
            "mongodb": ["mongod", "mongdb", "mongobd", "mongodbb"],
            "tensorflow": ["tensorflw", "tensrflow", "tensorflwo", "tensorfow"],
            "pytorch": ["pytorh", "pytroch", "pytoch", "pytorchh"]
        }
    
    def generate_training_data(self, samples_per_word: int = 50) -> Tuple[List[str], List[int]]:
        """Generate training data with correct and incorrect words"""
        texts = []
        labels = []  # 1 for correct, 0 for typo
        
        print(f"📊 Generating training data...")
        
        # Generate correct word samples
        for word in self.correct_words:
            for _ in range(samples_per_word):
                # Create context sentences
                context = self._create_context_sentence(word)
                texts.append(context)
                labels.append(1)  # Correct
        
        # Generate typo samples
        for word in self.correct_words:
            if word in self.typo_patterns:
                typos = self.typo_patterns[word]
                for typo in typos:
                    for _ in range(samples_per_word // len(typos)):
                        context = self._create_context_sentence(typo)
                        texts.append(context)
                        labels.append(0)  # Typo
        
        print(f"✅ Generated {len(texts)} training samples")
        print(f"   - Correct samples: {sum(labels)}")
        print(f"   - Typo samples: {len(labels) - sum(labels)}")
        
        return texts, labels
    
    def _create_context_sentence(self, word: str) -> str:
        """Create realistic resume context sentences"""
        templates = [
            f"I have experience with {word} development",
            f"Proficient in {word} programming",
            f"Used {word} for project implementation",
            f"Skilled in {word} and related technologies",
            f"Developed applications using {word}",
            f"Experience in {word} based solutions",
            f"Worked extensively with {word}",
            f"Implemented {word} in production environment",
            f"Strong background in {word}",
            f"Expertise in {word} development"
        ]
        
        import random
        return random.choice(templates)

class ResumeTypoNeuralNetwork:
    """Custom Neural Network for Resume Typo Detection"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 100, max_length: int = 50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
    def build_model(self) -> keras.Model:
        """Build custom neural network architecture from scratch"""
        print("🏗️ Building custom neural network architecture...")
        
        model = keras.Sequential([
            # Embedding layer - converts words to dense vectors
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length,
                name="word_embedding"
            ),
            
            # Bidirectional LSTM for sequence understanding
            layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, dropout=0.3),
                name="bidirectional_lstm_1"
            ),
            
            # Another LSTM layer for deeper understanding
            layers.Bidirectional(
                layers.LSTM(32, dropout=0.3),
                name="bidirectional_lstm_2"
            ),
            
            # Dense layers for classification
            layers.Dense(64, activation='relu', name="dense_1"),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu', name="dense_2"),
            layers.Dropout(0.3),
            
            # Output layer - binary classification (typo or not)
            layers.Dense(1, activation='sigmoid', name="output")
        ])
        
        # Compile with custom optimizer settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        print("✅ Neural network architecture built successfully")
        return model
    
    def prepare_data(self, texts: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and tokenize text data"""
        print("📝 Preparing and tokenizing data...")
        
        # Create and fit tokenizer
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size,
            oov_token="<OOV>"
        )
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences to same length
        X = keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        
        y = np.array(labels)
        
        print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} sequence length")
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, epochs: int = 20) -> Dict[str, Any]:
        """Train the neural network"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Validation set: {X_val.shape[0]} samples")
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_precision, val_recall = self.model.evaluate(X_val, y_val, verbose=0)
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        
        results = {
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_score': f1_score,
            'history': history.history
        }
        
        print(f"✅ Training completed!")
        print(f"📈 Validation Accuracy: {val_accuracy:.4f}")
        print(f"📈 Validation Precision: {val_precision:.4f}")
        print(f"📈 Validation Recall: {val_recall:.4f}")
        print(f"📈 Validation F1-Score: {f1_score:.4f}")
        
        return results
    
    def predict_typo(self, text: str) -> Tuple[float, bool]:
        """Predict if text contains typos"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not trained yet!")
        
        # Preprocess text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=self.max_length, padding='post', truncating='post'
        )
        
        # Predict
        prediction = self.model.predict(padded, verbose=0)[0][0]
        is_correct = prediction > 0.5
        
        return float(prediction), is_correct
    
    def save_model(self, model_path: str = "models/custom_resume_typo_model"):
        """Save the trained model and tokenizer"""
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        self.model.save(f"{model_path}/model.h5")
        
        # Save tokenizer
        with open(f"{model_path}/tokenizer.pickle", 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'created_at': datetime.now().isoformat()
        }
        
        with open(f"{model_path}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path: str = "models/custom_resume_typo_model"):
        """Load a trained model"""
        # Load config
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            max_length=config['max_length']
        )
        
        # Load model
        instance.model = keras.models.load_model(f"{model_path}/model.h5")
        
        # Load tokenizer
        with open(f"{model_path}/tokenizer.pickle", 'rb') as f:
            instance.tokenizer = pickle.load(f)
        
        print(f"✅ Model loaded from {model_path}")
        return instance

def plot_training_history(history: Dict[str, List[float]]):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Training')
    axes[0, 0].plot(history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Training')
    axes[0, 1].plot(history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history['precision'], label='Training')
    axes[1, 0].plot(history['val_precision'], label='Validation')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history['recall'], label='Training')
    axes[1, 1].plot(history['val_recall'], label='Validation')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Training plots saved to models/training_history.png")

def main():
    """Main training function"""
    print("🎓 Starting Custom Model Training for University Project")
    print("=" * 60)
    
    # Step 1: Generate training data
    data_generator = ResumeTypoDataGenerator()
    texts, labels = data_generator.generate_training_data(samples_per_word=100)
    
    # Step 2: Create and build model
    model = ResumeTypoNeuralNetwork(
        vocab_size=5000,
        embedding_dim=128,
        max_length=20
    )
    
    model.build_model()
    
    # Print model architecture
    print("\n🏗️ Model Architecture:")
    model.model.summary()
    
    # Step 3: Prepare data
    X, y = model.prepare_data(texts, labels)
    
    # Step 4: Train model
    results = model.train(X, y, epochs=25)
    
    # Step 5: Plot results
    try:
        plot_training_history(results['history'])
    except Exception as e:
        print(f"⚠️ Could not plot results: {e}")
    
    # Step 6: Save model
    model.save_model()
    
    # Step 7: Test the model
    print("\n🧪 Testing the trained model:")
    test_cases = [
        "I have experence in Python programming",  # Typo
        "I have experience in Python programming", # Correct
        "Proficient in Javascrip development",     # Typo
        "Proficient in JavaScript development",    # Correct
        "Used React framwork for frontend",        # Typo
        "Used React framework for frontend"        # Correct
    ]
    
    for test_text in test_cases:
        confidence, is_correct = model.predict_typo(test_text)
        status = "✅ CORRECT" if is_correct else "❌ TYPO DETECTED"
        print(f"{status} | Confidence: {confidence:.3f} | Text: {test_text}")
    
    print(f"\n🎉 CUSTOM MODEL TRAINING COMPLETED!")
    print(f"📊 Final Results:")
    print(f"   - Accuracy: {results['val_accuracy']:.4f}")
    print(f"   - F1-Score: {results['val_f1_score']:.4f}")
    print(f"   - Model saved to: models/custom_resume_typo_model/")
    print(f"\n✅ University requirement satisfied: Custom Neural Network trained from scratch!")

if __name__ == "__main__":
    main()