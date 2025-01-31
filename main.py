import numpy as np
import pandas as pd
from rdflib import Graph, Namespace
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_thresholds(g):
    """
    Load threshold values for each category from the Knowledge Graph.
    """
    query_str = """
    PREFIX ex: <http://example.org/>
    SELECT ?category ?thresholdValue
    WHERE {
        ?category a ex:Category ;
                  ex:thresholdValue ?thresholdValue .
    }
    """
    results = g.query(query_str)
    
    thresholds = {}
    for row in results:
        category_uri = row[0]
        threshold_val = row[1]
        category_name = category_uri.split('/')[-1]
        thresholds[category_name] = float(threshold_val)
    return thresholds

def generate_synthetic_data(num_samples, thresholds):
    """
    Generate synthetic historical data using thresholds loaded from the KG.
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

        # Check if overspending relative to the threshold
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

def apply_rules(g, current_spending, category, amount):
    """
    Apply both threshold and percentage-based rules for a given category and spending amount.
    """
    threshold_query = """
    PREFIX ex: <http://example.org/>
    SELECT ?thresholdValue
    WHERE {{
        ex:{cat} ex:thresholdValue ?thresholdValue .
    }}
    """.format(cat=category)
    
    threshold_result = g.query(threshold_query)
    threshold_value = None
    for row in threshold_result:
        threshold_value = float(row[0])

    if amount > threshold_value:
        print("Threshold violated for {0}: Spending {1} exceeds {2}".format(category, amount, threshold_value))
        return True

    rule_query = """
    PREFIX ex: <http://example.org/>
    SELECT ?relation ?percentageOf ?percentageValue
    WHERE {{
        ?rule a ex:BudgetRule ;
              ex:appliesTo ex:{cat} ;
              ex:relation ?relation ;
              ex:percentageOf ?percentageOf ;
              ex:percentageValue ?percentageValue .
    }}
    """.format(cat=category)

    rule_results = g.query(rule_query)
    
    for row in rule_results:
        relation = row[0]
        percentage_of = row[1]
        percentage_value = row[2]

        related_category = percentage_of.split('/')[-1]
        if related_category in current_spending:
            related_amount = current_spending[related_category]
        else:
            related_amount = 0.0

        limit = (float(percentage_value) / 100.0) * related_amount
        rel_name = relation.split('/')[-1]

        if rel_name == "LessThanOrEqualTo":
            if amount > limit:
                print("Rule violated for {0}: Spending {1} exceeds {2} ({3}% of {4})".format(
                    category, amount, limit, percentage_value, related_category))
                return True

    return False  

if __name__ == "__main__":
    g = Graph()
    g.parse("budget_ontology.ttl", format="ttl")

    thresholds = load_thresholds(g)

    # Generate data and train the model
    df_history = generate_synthetic_data(num_samples=200, thresholds=thresholds)
    model, cat_to_num = train_model(df_history)

    # Current monthly spending
    current_month_spending = {
        "Groceries": 200,       # Below threshold (400 in KG example)
        "Entertainment": 80,   # Below threshold (150) and less than 50% of Groceries
        "Clothes": 120,         # Below threshold (200) and less than 20% of Rent
        "Rent": 1100,           # At threshold (1200)
        "Insurance": 160,       # At threshold (200)
        "Car": 100              # Below threshold (700) and less than 80% of Insurance
    }

    # Check spending for all categories
    any_rule_violated = False
    for cat in current_month_spending:
        amt = current_month_spending[cat]
        rule_result = apply_rules(g, current_month_spending, cat, amt)
        if rule_result:
            any_rule_violated = True

    # Predict with the ML model
    test_category = "Entertainment"
    test_amount = current_month_spending[test_category]
    
    test_input_data = []
    # Convert category string to nume