"""
Sales Prediction Analysis 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'data_path': 'train.csv',
    'test_size': 0.2,
    'lag_periods': 5,
    'random_state': 42
}

class SalesPredictor:
    """Main class for sales prediction analysis"""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        self.model = None
        self.features = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            self.data = pd.read_csv(self.config['data_path'])
            self.data['Order Date'] = pd.to_datetime(
                self.data['Order Date'], format='%d/%m/%Y'
            )
            print(f"✓ Data loaded successfully! Shape: {self.data.shape}")
            return True
        except FileNotFoundError:
            print(f"✗ Error: File {self.config['data_path']} not found")
            return False
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def plot_sales_trend(self):
        """Visualize sales trend over time"""
        if self.data is None:
            print("✗ No data loaded. Please load data first.")
            return
            
        sales_by_date = self.data.groupby('Order Date')['Sales'].sum().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(sales_by_date['Order Date'], sales_by_date['Sales'], 
                linewidth=2, color='#2E86AB')
        plt.title('Sales Trend Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def create_features(self):
        """Create features for time series prediction"""
        if self.data is None:
            print("✗ No data loaded. Please load data first.")
            return None
            
        # Aggregate sales by date
        daily_sales = self.data.groupby('Order Date')['Sales'].sum().reset_index()
        daily_sales = daily_sales.sort_values('Order Date').reset_index(drop=True)
        
        # Create lagged features
        for i in range(1, self.config['lag_periods'] + 1):
            daily_sales[f'lag_{i}'] = daily_sales['Sales'].shift(i)
        
        # Add time-based features
        daily_sales['day_of_week'] = daily_sales['Order Date'].dt.dayofweek
        daily_sales['month'] = daily_sales['Order Date'].dt.month
        daily_sales['quarter'] = daily_sales['Order Date'].dt.quarter
        
        self.features = daily_sales.dropna()
        print(f"✓ Features created! Shape: {self.features.shape}")
        return self.features
    
    def prepare_model_data(self):
        """Prepare data for modeling"""
        if self.features is None:
            print("✗ No features created. Please create features first.")
            return None, None, None, None
            
        feature_columns = [col for col in self.features.columns 
                          if col not in ['Order Date', 'Sales']]
        
        X = self.features[feature_columns]
        y = self.features['Sales']
        
        # Split data (maintaining time order)
        split_idx = int(len(X) * (1 - self.config['test_size']))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the XGBoost model"""
        model_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': self.config['random_state']
        }
        
        self.model = xgb.XGBRegressor(**model_params)
        self.model.fit(X_train, y_train)
        print("✓ Model trained successfully!")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            print("✗ No model trained. Please train model first.")
            return None
            
        predictions = self.model.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'MAE': mean_absolute_error(y_test, predictions),
            'R2': r2_score(y_test, predictions)
        }
        
        print("\n📊 Model Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
            
        return predictions, metrics
    
    def plot_predictions(self, y_test, predictions):
        """Plot actual vs predicted sales"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(y_test.index, y_test, label='Actual Sales', 
                linewidth=2, color='#2E86AB')
        plt.plot(y_test.index, predictions, label='Predicted Sales', 
                linewidth=2, color='#A23B72', linestyle='--')
        
        plt.title('Sales Forecasting Results', fontsize=14, fontweight='bold')
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model is None:
            print("✗ No model trained. Please train model first.")
            return
            
        feature_columns = [col for col in self.features.columns 
                          if col not in ['Order Date', 'Sales']]
        
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Feature Importance in Sales Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting Sales Prediction Analysis...\n")
        
        # Load data
        if not self.load_data():
            return
        
        # Create features
        self.create_features()
        
        # Prepare data
        model_data = self.prepare_model_data()
        if model_data is None:
            return
        X_train, X_test, y_train, y_test = model_data
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate model
        predictions, metrics = self.evaluate_model(X_test, y_test)
        
        # Visualizations
        self.plot_sales_trend()
        self.plot_predictions(y_test, predictions)
        self.plot_feature_importance()
        
        print("\n✅ Analysis completed successfully!")

def main():
    """Main function to run the analysis"""
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create predictor instance
    predictor = SalesPredictor(CONFIG)
    
    # Run analysis
    predictor.run_analysis()

if __name__ == "__main__":
    main()
