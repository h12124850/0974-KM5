import numpy as np
import pandas as pd
from rdflib import Graph, Namespace
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

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

# def generate_synthetic_data(num_samples, thresholds):
#     """
#     Generate synthetic historical data using thresholds loaded from the KG.
#     (COMMENTED OUT - not needed for pure Symbolic AI)
#     """
#     return pd.DataFrame()

# def train_model(df):
#     """
#     Train a Decision Tree Classifier on historical data.
#     (COMMENTED OUT - not used in pure Symbolic AI)
#     """
#     return None, None

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

    # In pure Symbolic AI approach, we rely entirely on KG thresholds & rules
    thresholds = load_thresholds(g)

    # # ML generation + training is commented out
    # df_history = generate_synthetic_data(num_samples=200, thresholds=thresholds)
    # model, cat_to_num = train_model(df_history)

    current_month_spending = {
        "Groceries": 200,
        "Entertainment": 80,
        "Clothes": 120,
        "Rent": 1100,
        "Insurance": 160,
        "Car": 100
    }

    any_rule_violated = False
    for cat in current_month_spending:
        amt = current_month_spending[cat]
        rule_result = apply_rules(g, current_month_spending, cat, amt)
        if rule_result:
            any_rule_violated = True

    # # For symbolic-only, no ML predictions:
    # model_prediction = 0

    # Final decision based solely on rules
    if any_rule_violated:
        final_decision = 1
    else:
        final_decision = 0

    print("\n--- Purely Symbolic AI ---")
    print("Current Month Spending:", current_month_spending)
    # print("Model Prediction for Entertainment:", model_prediction, " (COMMENTED OUT - no ML in symbolic version)")
    print("Any Rule Violations Detected:", any_rule_violated)
    if final_decision == 1:
        print("Final Decision: Overspending is likely.")
    else:
        print("Final Decision: Overspending is not likely.")
