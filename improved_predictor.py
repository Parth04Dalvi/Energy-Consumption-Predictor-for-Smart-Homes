# Energy Consumption Predictor for Smart Homes
# This script demonstrates a machine learning model to predict residential energy
# consumption based on various factors. It uses a synthetic dataset for
# demonstration purposes and includes features for saving, loading, and
# visualizing the model.

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib # Library for model persistence

def generate_synthetic_data(num_samples=2000):
    """
    Generates a synthetic dataset for smart home energy consumption.

    Features now include:
    - Temperature (random float, °C)
    - Humidity (random float, %)
    - Wind Speed (random float, m/s)
    - Time of Day (integer, 0-23)
    - Day of Week (integer, 0-6, Monday=0)
    - Number of Occupants (integer, 1-5)
    - is_hvac_on (boolean)
    - is_major_appliance_on (boolean)
    
    The target variable is 'Energy Consumption' (kWh).
    """
    np.random.seed(42)  # for reproducibility

    # Generate feature data
    temperature = np.random.uniform(15, 30, num_samples)
    humidity = np.random.uniform(40, 80, num_samples)
    wind_speed = np.random.uniform(0, 10, num_samples)
    time_of_day = np.random.randint(0, 24, num_samples)
    day_of_week = np.random.randint(0, 7, num_samples)
    num_occupants = np.random.randint(1, 6, num_samples)
    is_hvac_on = (temperature > 25) | (temperature < 18)
    is_major_appliance_on = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

    # Generate target variable (Energy Consumption) with noise and new feature influences
    base_consumption = 5 + 0.5 * num_occupants
    temp_consumption = np.maximum(0, (temperature - 20) * 0.8)
    time_consumption = np.abs(12 - time_of_day) * 0.2
    hvac_consumption = is_hvac_on * np.random.uniform(2, 5, num_samples)
    appliance_consumption = is_major_appliance_on * np.random.uniform(3, 6, num_samples)
    day_consumption = np.where(day_of_week >= 5, 1.5, 0) # Higher consumption on weekends (day 5 and 6)
    
    noise = np.random.normal(0, 1, num_samples)
    
    energy_consumption = base_consumption + temp_consumption + time_consumption + \
                         hvac_consumption + appliance_consumption + day_consumption + noise

    # Create a Pandas DataFrame
    data = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'time_of_day': time_of_day,
        'day_of_week': day_of_week,
        'num_occupants': num_occupants,
        'is_hvac_on': is_hvac_on,
        'is_major_appliance_on': is_major_appliance_on,
        'energy_consumption': energy_consumption
    })
    
    return data

def train_and_evaluate_model(data):
    """
    Trains a RandomForestRegressor model and evaluates its performance.
    """
    # Separate features (X) and target (y)
    X = data.drop('energy_consumption', axis=1)
    y = data['energy_consumption']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForestRegressor model
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Training complete.")
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Performance ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kWh")
    print(f"R-squared (R²): {r2:.2f}")

    return model, X_test, y_test, predictions

def save_model(model, filename="energy_predictor_model.joblib"):
    """Saves the trained model to a file."""
    try:
        joblib.dump(model, filename)
        print(f"\nModel successfully saved as '{filename}'")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filename="energy_predictor_model.joblib"):
    """Loads a trained model from a file."""
    try:
        model = joblib.load(filename)
        print(f"\nModel '{filename}' successfully loaded.")
        return model
    except FileNotFoundError:
        print(f"\nError: Model file '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def visualize_results(X_test, y_test, predictions, model):
    """
    Creates visualizations of the model's performance and feature importance.
    """
    # Scatter plot of Actual vs. Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Energy Consumption (kWh)")
    plt.ylabel("Predicted Energy Consumption (kWh)")
    plt.title("Actual vs. Predicted Energy Consumption")
    plt.grid(True)
    plt.show()

    # Feature importance bar chart
    feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sorted_importances.plot(kind='bar')
    plt.title("Feature Importances")
    plt.ylabel("Importance Score")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Step 1: Generate synthetic data
    print("Generating synthetic dataset...")
    energy_data = generate_synthetic_data()
    print("Dataset generated successfully.")
    
    # Step 2: Train and evaluate the model
    trained_model, X_test, y_test, predictions = train_and_evaluate_model(energy_data)
    
    # Step 3: Save the trained model for future use
    save_model(trained_model)

    # Step 4: Visualize the results
    print("\nCreating visualizations...")
    visualize_results(X_test, y_test, predictions, trained_model)
    
    # Step 5: Demonstrate prediction on new, unseen data using the trained model
    # To demonstrate loading, you could uncomment the line below and comment out
    # the training step above, assuming the model has already been saved.
    # trained_model = load_model()
    
    if trained_model:
        print("\n--- Demonstrating Prediction on New Data ---")
        
        # Example: a moderately warm, humid day on a weekend with a major appliance on
        new_example = pd.DataFrame([{
            'temperature': 24.5,
            'humidity': 65.0,
            'wind_speed': 5.2,
            'time_of_day': 17,
            'day_of_week': 6,
            'num_occupants': 3,
            'is_hvac_on': False,
            'is_major_appliance_on': True
        }])

        prediction = trained_model.predict(new_example)
        print(f"Predicted energy consumption for the new example: {prediction[0]:.2f} kWh")
