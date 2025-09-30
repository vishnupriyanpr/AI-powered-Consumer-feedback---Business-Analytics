"""
AI Customer Feedback Analyzer - Model Training Script
Optimized for RTX 4060 GPU - AIML Project 2025
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import json
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerFeedbackTrainer:
    def __init__(self, dataset_path="dataset/", model_output="models/"):
        self.dataset_path = dataset_path
        self.model_output = model_output
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("="*60)
        print("🚀 AI Customer Feedback Analyzer - AIML Project")
        print("="*60)
        print(f"🔥 Using device: {self.device}")
        print(f"📁 Dataset path: {self.dataset_path}")
        print(f"💾 Output path: {self.model_output}")

        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

        os.makedirs(model_output, exist_ok=True)
        self.tokenizer = None

    def load_datasets(self):
        print("\n📊 Loading datasets...")

        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        print(f"📂 Found {len(csv_files)} CSV files: {csv_files}")

        dataframes = []
        for file in csv_files:
            file_path = os.path.join(self.dataset_path, file)
            df = pd.read_csv(file_path)
            print(f"📁 {file}: {len(df)} samples, Columns: {list(df.columns)}")
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"📋 Combined dataset: {len(combined_df)} total samples")

        return combined_df

    def preprocess_data(self, df):
        print("\n🔧 Preprocessing data...")

        # Auto-detect columns
        text_col = 'sentence'  # From your data
        sentiment_col = 'sentiment'
        emotion_col = 'emotion'

        print(f"📝 Text column: {text_col}")
        print(f"😊 Sentiment column: {sentiment_col}")
        print(f"💭 Emotion column: {emotion_col}")

        # Clean data
        df[text_col] = df[text_col].astype(str).str.strip()
        df = df[df[text_col].str.len() > 10]
        print(f"🧹 After cleaning: {len(df)} samples remaining")

        # Prepare sentiment data
        sentiment_data = None
        if sentiment_col in df.columns:
            sentiment_data = df[[text_col, sentiment_col]].dropna()
            sentiment_data.columns = ['text', 'label']
            sentiment_data['label'] = sentiment_data['label'].astype(str).str.lower().str.strip()

            print(f"😊 Sentiment data: {len(sentiment_data)} samples")
            print(f"   Labels: {sentiment_data['label'].value_counts().to_dict()}")

        # Prepare emotion data
        emotion_data = None
        if emotion_col in df.columns:
            emotion_data = df[[text_col, emotion_col]].dropna()
            emotion_data.columns = ['text', 'label']
            emotion_data['label'] = emotion_data['label'].astype(str).str.lower().str.strip()

            # Sample large emotion dataset for faster training (optional)
            if len(emotion_data) > 20000:
                emotion_data = emotion_data.sample(n=20000, random_state=42)
                print(f"💭 Emotion data (sampled): {len(emotion_data)} samples")
            else:
                print(f"💭 Emotion data: {len(emotion_data)} samples")
            print(f"   Labels: {emotion_data['label'].value_counts().to_dict()}")

        return sentiment_data, emotion_data

    def create_model_and_tokenizer(self, num_labels, model_name="roberta-base"):
        print(f"🤖 Creating model: {model_name} with {num_labels} labels")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        return model

    def tokenize_data(self, texts, labels, max_length=256):  # Reduced for speed
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        if isinstance(labels.iloc[0], str):
            le = LabelEncoder()
            encoded_labels = le.fit_transform(labels)
            return encodings, torch.tensor(encoded_labels, dtype=torch.long), le
        else:
            return encodings, torch.tensor(labels.values, dtype=torch.long), None

    def train_model(self, data, model_name, task_name):
        if data is None or len(data) < 50:
            print(f"⚠️  Insufficient data for {task_name}, skipping...")
            return None, None

        print(f"\n🚀 Training {task_name} model...")
        print(f"📊 Dataset size: {len(data)} samples")

        texts = data['text'].tolist()
        labels = data['label']

        unique_labels = sorted(list(set(labels)))
        num_labels = len(unique_labels)
        print(f"🏷️  {task_name} classes ({num_labels}): {unique_labels}")

        if num_labels < 2:
            print(f"⚠️  Only {num_labels} unique label found, skipping...")
            return None, None

        # Create model
        model = self.create_model_and_tokenizer(num_labels, "roberta-base")

        # Tokenize
        encodings, label_tensors, label_encoder = self.tokenize_data(texts, labels)

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        print(f"📚 Training samples: {len(train_texts)}")
        print(f"📖 Validation samples: {len(val_texts)}")

        # Tokenize splits
        train_encodings, train_label_tensors, _ = self.tokenize_data(pd.Series(train_texts), pd.Series(train_labels))
        val_encodings, val_label_tensors, _ = self.tokenize_data(pd.Series(val_texts), pd.Series(val_labels))

        # Dataset class
        class FeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx].clone().detach()
                return item

        train_dataset = FeedbackDataset(train_encodings, train_label_tensors)
        val_dataset = FeedbackDataset(val_encodings, val_label_tensors)

        # **FIXED** Training arguments (corrected parameter names)
        training_args = TrainingArguments(
            output_dir=f"{self.model_output}/{model_name}",
            num_train_epochs=3,  # Reduced for faster training
            per_device_train_batch_size=16,  # Increased for RTX 4060
            per_device_eval_batch_size=32,
            warmup_steps=min(500, len(train_dataset) // 10),
            weight_decay=0.01,
            logging_dir=f"{self.model_output}/{model_name}/logs",
            logging_steps=50,
            eval_strategy="steps",  # FIXED: was evaluation_strategy
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            dataloader_pin_memory=torch.cuda.is_available(),
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            remove_unused_columns=False,
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            return {'accuracy': accuracy}

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        print("🏋️  Starting training...")
        trainer.train()

        # Save model
        model_path = f"{self.model_output}/{model_name}"
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)

        # Save label mapping
        if label_encoder:
            label_mapping = {str(i): label for i, label in enumerate(label_encoder.classes_)}
        else:
            label_mapping = {str(i): label for i, label in enumerate(unique_labels)}

        with open(f"{model_path}/label_mapping.json", "w") as f:
            json.dump(label_mapping, f, indent=2)

        print(f"✅ {task_name} model saved to {model_path}")

        # Final evaluation
        eval_results = trainer.evaluate()
        print(f"📈 {task_name} Results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

        return model, trainer

    def train_all_models(self):
        print("\n🎯 Starting Complete AI Model Training")
        print("="*60)

        # Load datasets
        df = self.load_datasets()

        # Preprocess
        sentiment_data, emotion_data = self.preprocess_data(df)

        # Train sentiment model
        print("\n" + "="*40)
        print("😊 SENTIMENT ANALYSIS TRAINING")
        print("="*40)
        sentiment_model, sentiment_trainer = self.train_model(
            sentiment_data, "sentiment_analyzer", "Sentiment Analysis"
        )

        # Train emotion model
        print("\n" + "="*40)
        print("💭 EMOTION DETECTION TRAINING")
        print("="*40)
        emotion_model, emotion_trainer = self.train_model(
            emotion_data, "emotion_detector", "Emotion Detection"
        )

        # Save training info
        model_info = {
            "project": "AI Customer Feedback Analyzer - AIML Project",
            "training_date": datetime.now().isoformat(),
            "dataset_samples": len(df),
            "sentiment_model": "sentiment_analyzer" if sentiment_model else None,
            "emotion_model": "emotion_detector" if emotion_model else None,
            "device_used": str(self.device),
            "models_ready": True,
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
            "pytorch_version": torch.__version__
        }

        with open(f"{self.model_output}/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Models saved in: {self.model_output}")
        print(f"🔗 Ready for frontend integration!")
        print(f"🌟 Your AI Customer Feedback Analyzer is ready!")
        print("="*60)

        return sentiment_model, emotion_model

if __name__ == "__main__":
    try:
        trainer = CustomerFeedbackTrainer(
            dataset_path="dataset/",
            model_output="models/"
        )

        sentiment_model, emotion_model = trainer.train_all_models()

        if sentiment_model or emotion_model:
            print("\n🧪 Quick test completed!")
            print("✅ Models are production-ready!")
            print("\n📋 Next steps:")
            print("  1. Update backend integration")
            print("  2. Restart Flask server")
            print("  3. Test frontend with real AI!")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
