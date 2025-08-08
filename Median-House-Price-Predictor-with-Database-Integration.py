# =============================================================================
# MEDIAN HOUSE PRICE PREDICTOR WITH DATABASE INTEGRATION
# =============================================================================

# 1. INSTALL AND IMPORT REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

# =============================================================================
# 2. DATABASE SETUP AND CONFIGURATION
# =============================================================================

class HousePriceDatabase:
    def __init__(self, db_name='house_price_predictions.db'):
        self.db_name = db_name
        self.setup_database()
    
    def setup_database(self):
        """Create database tables for storing predictions and model metrics"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            house_area REAL,
            house_age REAL,
            avg_rooms REAL,
            avg_bedrooms REAL,
            population REAL,
            avg_occupancy REAL,
            latitude REAL,
            longitude REAL,
            predicted_price REAL,
            prediction_date TIMESTAMP,
            model_used TEXT
        )
        ''')
        
        # Create model metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            r2_score REAL,
            mae REAL,
            rmse REAL,
            accuracy_percentage REAL,
            training_date TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database setup completed!")
    
    def store_prediction(self, features, prediction, model_name):
        """Store a single prediction in the database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO predictions (house_area, house_age, avg_rooms, avg_bedrooms, 
                               population, avg_occupancy, latitude, longitude,
                               predicted_price, prediction_date, model_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (*features, prediction, datetime.now(), model_name))
        
        conn.commit()
        conn.close()
    
    def store_model_metrics(self, model_name, r2, mae, rmse, accuracy):
        """Store model performance metrics"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO model_metrics (model_name, r2_score, mae, rmse, 
                                 accuracy_percentage, training_date)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_name, r2, mae, rmse, accuracy, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_predictions_by_area_range(self, min_area, max_area):
        """Get predictions within a specific area range"""
        conn = sqlite3.connect(self.db_name)
        query = '''
        SELECT * FROM predictions 
        WHERE house_area BETWEEN ? AND ?
        ORDER BY predicted_price
        '''
        df = pd.read_sql_query(query, conn, params=(min_area, max_area))
        conn.close()
        return df

# Initialize database
db = HousePriceDatabase()

# =============================================================================
# 3. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_prepare_data():
    """Load California Housing dataset and prepare for modeling"""
    print("📊 Loading California Housing Dataset...")
    
    # Load the dataset
    housing = fetch_california_housing()
    
    # Create DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target * 100000  # Convert to actual price range
    
    # Feature engineering - create total area approximation
    df['total_area'] = df['AveRooms'] * df['AveBedrms'] * df['Population'] / df['AveOccup']
    df['total_area'] = df['total_area'] / 100  # Scale down for realistic area
    
    print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"📈 Price range: ${df['target'].min():,.0f} - ${df['target'].max():,.0f}")
    
    return df

# Load data
housing_data = load_and_prepare_data()

# Display basic information
print("\n📋 Dataset Information:")
print(housing_data.head())
print(f"\n📊 Dataset Statistics:")
print(housing_data.describe())

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# =============================================================================

def create_visualizations(data):
    """Create visualizations for data understanding"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Price distribution
    axes[0,0].hist(data['target'], bins=50, alpha=0.7, color='skyblue')
    axes[0,0].set_title('Distribution of House Prices')
    axes[0,0].set_xlabel('Price ($)')
    axes[0,0].set_ylabel('Frequency')
    
    # Area vs Price scatter plot
    axes[0,1].scatter(data['total_area'], data['target'], alpha=0.6, color='coral')
    axes[0,1].set_title('House Area vs Price')
    axes[0,1].set_xlabel('Total Area (sq ft)')
    axes[0,1].set_ylabel('Price ($)')
    
    # Correlation heatmap
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[1,0], fmt='.2f')
    axes[1,0].set_title('Feature Correlation Matrix')
    
    # Geographic distribution
    scatter = axes[1,1].scatter(data['Longitude'], data['Latitude'], 
                               c=data['target'], cmap='viridis', alpha=0.6)
    axes[1,1].set_title('Geographic Price Distribution')
    axes[1,1].set_xlabel('Longitude')
    axes[1,1].set_ylabel('Latitude')
    plt.colorbar(scatter, ax=axes[1,1])
    
    plt.tight_layout()
    plt.show()

create_visualizations(housing_data)

# =============================================================================
# 5. MACHINE LEARNING MODEL DEVELOPMENT
# =============================================================================

class HousePricePredictor:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_features(self, data):
        """Prepare features for modeling"""
        # Select relevant features
        feature_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                       'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        X = data[feature_cols]
        y = data['target']
        
        self.feature_columns = feature_cols
        return X, y
    
    def train_and_evaluate_models(self, data):
        """Train multiple models and select the best one"""
        print("🚀 Training multiple models...")
        
        X, y = self.prepare_features(data)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_score = 0
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            accuracy = r2 * 100  # Convert R² to percentage
            
            results[name] = {
                'model': model,
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'accuracy': accuracy
            }
            
            # Store metrics in database
            db.store_model_metrics(name, r2, mae, rmse, accuracy)
            
            print(f"  ✅ {name} - Accuracy: {accuracy:.2f}%")
            
            # Keep track of best model
            if r2 > best_score:
                best_score = r2
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n🏆 Best Model: {self.best_model_name} with {results[self.best_model_name]['accuracy']:.2f}% accuracy")
        return results
    
    def predict_price(self, house_area, house_age=10, avg_rooms=6, avg_bedrooms=1.2, 
                     population=3000, avg_occupancy=3, latitude=34, longitude=-118):
        """Predict house price based on input features"""
        
        # Prepare input features
        features = np.array([[avg_rooms/avg_bedrooms, house_age, avg_rooms, avg_bedrooms,
                            population, avg_occupancy, latitude, longitude]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)[0]
        
        # Store prediction in database
        feature_list = [house_area, house_age, avg_rooms, avg_bedrooms, 
                       population, avg_occupancy, latitude, longitude]
        db.store_prediction(feature_list, prediction, self.best_model_name)
        
        return prediction

# Initialize and train predictor
predictor = HousePricePredictor()
model_results = predictor.train_and_evaluate_models(housing_data)

# =============================================================================
# 6. MODEL PERFORMANCE VISUALIZATION
# =============================================================================

def plot_model_comparison(results):
    """Plot model performance comparison"""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(12, 6))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, accuracies, color=['skyblue', 'coral', 'lightgreen'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Add horizontal line at 95%
    plt.axhline(y=95, color='red', linestyle='--', label='Target: 95%')
    plt.legend()
    
    # Error metrics comparison
    plt.subplot(1, 2, 2)
    mae_values = [results[model]['mae'] for model in models]
    rmse_values = [results[model]['rmse'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
    plt.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.title('Model Error Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_model_comparison(model_results)

# =============================================================================
# 7. PREDICTION SYSTEM WITH USER INTERFACE
# =============================================================================

def get_price_prediction_interface():
    """Interactive interface for price prediction"""
    print("\n" + "="*60)
    print("🏠 HOUSE PRICE PREDICTION SYSTEM")
    print("="*60)
    
    try:
        # Get user input
        house_area = float(input("Enter house area (sq ft): "))
        house_age = float(input("Enter house age (years) [default: 10]: ") or 10)
        latitude = float(input("Enter latitude [default: 34.0]: ") or 34.0)
        longitude = float(input("Enter longitude [default: -118.0]: ") or -118.0)
        
        # Make prediction
        predicted_price = predictor.predict_price(
            house_area=house_area,
            house_age=house_age,
            latitude=latitude,
            longitude=longitude
        )
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"📍 House Area: {house_area:,.0f} sq ft")
        print(f"📅 House Age: {house_age} years")
        print(f"💰 Predicted Price: ${predicted_price:,.2f}")
        print(f"🤖 Model Used: {predictor.best_model_name}")
        
        # Get similar houses from database
        area_range = house_area * 0.1  # 10% range
        similar_houses = db.get_predictions_by_area_range(
            house_area - area_range, house_area + area_range
        )
        
        if len(similar_houses) > 1:
            print(f"\n📊 Found {len(similar_houses)} similar houses in database:")
            print(f"   💵 Price range: ${similar_houses['predicted_price'].min():,.0f} - ${similar_houses['predicted_price'].max():,.0f}")
            print(f"   📈 Average price: ${similar_houses['predicted_price'].mean():,.0f}")
            
        return predicted_price
        
    except ValueError:
        print("❌ Invalid input! Please enter numeric values.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# =============================================================================
# 8. DATABASE ANALYSIS AND REPORTING
# =============================================================================

def analyze_database_predictions():
    """Analyze stored predictions in the database"""
    conn = sqlite3.connect(db.db_name)
    
    # Get all predictions
    predictions_df = pd.read_sql_query("SELECT * FROM predictions", conn)
    
    if len(predictions_df) > 0:
        print("\n📊 DATABASE ANALYSIS:")
        print(f"Total predictions stored: {len(predictions_df)}")
        print(f"Price range: ${predictions_df['predicted_price'].min():,.0f} - ${predictions_df['predicted_price'].max():,.0f}")
        print(f"Average predicted price: ${predictions_df['predicted_price'].mean():,.0f}")
        
        # Model performance from database
        metrics_df = pd.read_sql_query("SELECT * FROM model_metrics", conn)
        print(f"\n🎯 MODEL PERFORMANCE:")
        for _, row in metrics_df.iterrows():
            print(f"{row['model_name']}: {row['accuracy_percentage']:.2f}% accuracy")
    else:
        print("📝 No predictions in database yet.")
    
    conn.close()

# =============================================================================
# 9. BATCH PREDICTION SYSTEM
# =============================================================================

def batch_prediction_demo():
    """Demonstrate batch predictions for multiple houses"""
    print("\n🏘️ BATCH PREDICTION DEMO:")
    
    # Sample houses for prediction
    sample_houses = [
        {'area': 1200, 'age': 5, 'lat': 34.05, 'lon': -118.25},
        {'area': 1800, 'age': 15, 'lat': 37.77, 'lon': -122.42},
        {'area': 2200, 'age': 8, 'lat': 32.71, 'lon': -117.16},
        {'area': 1500, 'age': 20, 'lat': 34.42, 'lon': -119.70},
        {'area': 2800, 'age': 3, 'lat': 37.39, 'lon': -121.89}
    ]
    
    predictions = []
    for i, house in enumerate(sample_houses, 1):
        price = predictor.predict_price(
            house_area=house['area'],
            house_age=house['age'],
            latitude=house['lat'],
            longitude=house['lon']
        )
        predictions.append(price)
        print(f"House {i}: {house['area']} sq ft, Age {house['age']} → ${price:,.0f}")
    
    return predictions

# =============================================================================
# 10. MAIN EXECUTION AND DEMONSTRATION
# =============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("🚀 HOUSE PRICE PREDICTOR - SYSTEM READY!")
    print("="*60)
    
    # Check if we achieved 95% accuracy
    best_accuracy = max([results['accuracy'] for results in model_results.values()])
    
    if best_accuracy >= 95:
        print(f"✅ SUCCESS! Achieved {best_accuracy:.2f}% accuracy (Target: 95%+)")
    else:
        print(f"⚠️ Current best accuracy: {best_accuracy:.2f}% (Target: 95%+)")
        print("💡 Consider feature engineering or trying advanced models for higher accuracy")
    
    # Run demonstrations
    batch_prediction_demo()
    analyze_database_predictions()
    
    print(f"\n🎯 SYSTEM FEATURES:")
    print(f"✅ Machine Learning Models: {len(predictor.models)} trained")
    print(f"✅ Database Integration: SQLite with prediction storage")
    print(f"✅ Batch Prediction: Multiple houses at once")
    print(f"✅ Performance Tracking: Model metrics stored")
    print(f"✅ Data Analysis: Historical prediction analysis")
    
    print(f"\n📋 TO USE THE PREDICTION SYSTEM:")
    print(f"   Run: get_price_prediction_interface()")
    
# Run the main system
main()

# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

def get_model_performance():
    """Get detailed model performance metrics"""
    conn = sqlite3.connect(db.db_name)
    metrics_df = pd.read_sql_query("""
        SELECT model_name, accuracy_percentage, r2_score, mae, rmse, training_date 
        FROM model_metrics 
        ORDER BY accuracy_percentage DESC
    """, conn)
    conn.close()
    return metrics_df

def export_predictions_to_csv():
    """Export all predictions to CSV file"""
    conn = sqlite3.connect(db.db_name)
    predictions_df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    
    predictions_df.to_csv('house_price_predictions.csv', index=False)
    print("✅ Predictions exported to 'house_price_predictions.csv'")
    return predictions_df

print("\n🎉 Project setup completed successfully!")
print("Run get_price_prediction_interface() to start predicting house prices!")
