# Energy Consumption Predictor for Smart Homes
# This script demonstrates a simple machine learning model to predict
# residential energy consumption based on various factors. It uses a
# synthetic dataset for demonstration purposes.

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def generate_synthetic_data(num_samples=1000):
    """
    Generates a synthetic dataset for smart home energy consumption.

    Features include:
    - Temperature (random float, °C)
    - Time of Day (integer, 0-23)
    - Number of Occupants (integer, 1-5)
    - Appliances Used (e.g., HVAC on/off, boolean)
    
    The target variable is 'Energy Consumption' (kWh).
    """
    np.random.seed(42) # for reproducibility

    # Generate feature data
    temperature = np.random.uniform(15, 30, num_samples) # Ambient temperature in °C
    time_of_day = np.random.randint(0, 24, num_samples) # Hour of the day
    num_occupants = np.random.randint(1, 6, num_samples) # Number of people in the house
    is_hvac_on = (temperature > 25) | (temperature < 18) # HVAC on if it's too hot or too cold
    is_major_appliance_on = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3]) # 30% chance a major appliance is on

    # Generate target variable (Energy Consumption) with some noise
    # The consumption model is based on simple rules to create a realistic pattern
    # Base consumption + temp influence + time influence + occupants + appliance
    base_consumption = 5 + 0.5 * num_occupants
    temp_consumption = np.maximum(0, (temperature - 20) * 0.8)
    time_consumption = np.abs(12 - time_of_day) * 0.2
    hvac_consumption = is_hvac_on * np.random.uniform(2, 5, num_samples)
    appliance_consumption = is_major_appliance_on * np.random.uniform(3, 6, num_samples)
    
    noise = np.random.normal(0, 1, num_samples)
    
    energy_consumption = base_consumption + temp_consumption + time_consumption + hvac_consumption + appliance_consumption + noise

    # Create a Pandas DataFrame
    data = pd.DataFrame({
        'temperature': temperature,
        'time_of_day': time_of_day,
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
    X = data[['temperature', 'time_of_day', 'num_occupants', 'is_hvac_on', 'is_major_appliance_on']]
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

    return model

def predict_new_data(model, new_data):
    """
    Uses the trained model to predict energy consumption for new data.
    """
    # Ensure new data has the same columns and order as the training data
    new_df = pd.DataFrame(new_data)
    prediction = model.predict(new_df)
    
    return prediction

if __name__ == "__main__":
    # Step 1: Generate synthetic data
    print("Generating synthetic dataset...")
    energy_data = generate_synthetic_data(num_samples=2000)
    print("Dataset generated successfully.")
    
    # Step 2: Train and evaluate the model
    trained_model = train_and_evaluate_model(energy_data)
    
    # Step 3: Demonstrate prediction on new, unseen data
    print("\n--- Demonstrating Prediction on New Data ---")
    
    # Example 1: Cool day, midday, 2 occupants, no major appliance
    new_example_1 = pd.DataFrame([{
        'temperature': 20.0,
        'time_of_day': 12,
        'num_occupants': 2,
        'is_hvac_on': False,
        'is_major_appliance_on': False
    }])
    
    # Example 2: Hot day, evening, 4 occupants, major appliance on
    new_example_2 = pd.DataFrame([{
        'temperature': 28.5,
        'time_of_day': 19,
        'num_occupants': 4,
        'is_hvac_on': True,
        'is_major_appliance_on': True
    }])

    prediction_1 = predict_new_data(trained_model, new_example_1)
    prediction_2 = predict_new_data(trained_model, new_example_2)
    
    print(f"Predicted energy consumption for example 1: {prediction_1[0]:.2f} kWh")
    print(f"Predicted energy consumption for example 2: {prediction_2[0]:.2f} kWh")

    # You can visualize the model's performance by uncommenting the following lines
    # import matplotlib.pyplot as plt
    # from sklearn.model_selection import learning_curve
    #
    # plt.figure(figsize=(10, 6))
    # plt.scatter(energy_data['temperature'], energy_data['energy_consumption'], c=energy_data['time_of_day'], cmap='viridis')
    # plt.title('Synthetic Energy Consumption Data')
    # plt.xlabel('Temperature (°C)')
    # plt.ylabel('Energy Consumption (kWh)')
    # plt.colorbar(label='Time of Day (hour)')
    # plt.grid(True)
    # plt.show()
