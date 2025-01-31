import numpy as np
import pandas as pd
# from rdflib import Graph, Namespace  # COMMENTED OUT - not needed for pure Statistical AI
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# def load_thresholds(g):
#     """
#     Load threshold values for each category from the Knowledge Graph.
#     """
#     query_str = """
#     PREFIX ex: <http://example.org/>
#     SELECT ?category ?thresholdValue
#     WHERE {
#         ?category a ex:Category ;
#                   ex:thresholdValue ?thresholdValue .
#     }
#     """
#     results = g.query(query_str)
    
#     thresholds = {}
#     for row in results:
#         category_uri = row[0]
#         threshold_val = row[1]
#         category_name = category_uri.split('/')[-1]
#         thresholds[category_name] = float(threshold_val)
#     return thresholds

def generate_synthetic_data(num_samples, thresholds):
    """
    Generate synthetic historical data using user-provided thresholds.
    (No KG usage here for pure ML approach; you can pass a custom dictionary.)
    """
    categories = list(thresholds.keys())
    np.random.seed(42)  # for reproducible synthetic data
    
    data = []
    for i in range(num_samples):
        # Pick a category randomly
        cat_index = np.random.randint(0, len(categories))
        cat = categories[cat_index]
        
        # Random spending within plausible ranges
        if cat == "Groceries":
            amount = np.random.randint(100, 501)
        elif cat == "Entertainment":
            amount = np.random.randint(20, 201)
        elif cat == "Clothes":
            amount = np.random.randint(20, 301)
        elif cat == "Rent":
            amount = np.random.randint(800, 1501)
        elif cat == "Insurance":
            amount = np.random.randint(100, 301)
        elif cat == "Car":
            amount = np.random.randint(100, 1001)
        else:
            # fallback if new categories appear
            amount = np.random.randint(50, 1001)

        # Check overspending relative to threshold
        threshold = thresholds[cat]
        overspending = 0
        if amount > threshold:
            overspending = 1
        
        data.append([cat, amount, overspending])
    
    df = pd.DataFrame(data, columns=["category", "amount", "overspending"])
    return df

def train_model(df):
    """
    Train a Decision Tree Classifier on historical data.
    """
    # Encode category as numeric
    categories = df['category'].unique()
    cat_to_num = {}
    idx = 0
    for c in categories:
        cat_to_num[c] = idx
        idx += 1

    # Create a new column with numeric encodings
    category_nums = []
    for i in range(len(df)):
        cat = df.iloc[i]['category']
        category_nums.append(cat_to_num[cat])
    df['category_num'] = category_nums

    X = df[['category_num', 'amount']]
    y = df['overspending']

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, cat_to_num

# def apply_rules(g, current_spending, category, amount):
#     """
#     Apply both threshold and percentage-based rules for a given category and spending amount.
#     (COMMENTED OUT: not used in pure Statistical AI version)
#     """
#     return False

if __name__ == "__main__":
    # For the pure Statistical AI approach, define thresholds manually:
    custom_thresholds = {
        "Groceries": 400,
        "Entertainment": 150,
        "Clothes": 200,
        "Rent": 1200,
        "Insurance": 200,
        "Car": 700
    }

    # Generate data and train the model (no KG usage)
    df_history = generate_synthetic_data(num_samples=200, thresholds=custom_thresholds)
    model, cat_to_num = train_model(df_history)

    # Current monthly spending
    current_month_spending = {
        "Groceries": 200,
        "Entertainment": 80,
        "Clothes": 120,
        "Rent": 1100,
        "Insurance": 160,
        "Car": 100
    }

    # --- Symbolic rule checks are disabled in this version ---
    any_rule_violated = False  # Always False because we’re not applying any rules

    # Predict with the ML model
    test_category = "Entertainment"
    test_amount = current_month_spending[test_category]
    
    test_input_data = []
    # Convert category string to numeric encoding
    test_input_data.append([cat_to_num[test_category], test_amount])
    test_input = pd.DataFrame(test_input_data, columns=['category_num', 'amount'])
    
    model_prediction = model.predict(test_input)[0]

    # Final decision: ML prediction only
    final_decision = model_prediction

    print("\n--- Purely Statistical AI ---")
    print("Current Month Spending:", current_month_spending)
    print("Model Prediction for {0} (0=Not Overspending, 1=Overspending): {1}".format(test_category, model_prediction))
    print("Any Rule Violations Detected:", any_rule_violated)
    if final_decision == 1:
        print("Final Decision: Overspending is likely.")
    else:
        print("Final Decision: Overspending is not likely.")
