"""
Association Rule Learning on the Groceries Dataset
Assignment: Unsupervised Learning Method - Association Rule Learning
Software: Python (pandas, mlxtend, matplotlib)

This script:
1. Loads and explores the Groceries dataset (Q1)
2. Mines frequent itemsets using the Apriori algorithm (Q2)
3. Generates association rules (Q3)
4. Produces summary tables and figures used in the report (Q4-Q8 draw on this output)
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ---------------------------------------------------------------------------
# 0. LOAD DATA
# ---------------------------------------------------------------------------
# The Groceries.csv file is a "basket format" file: each row is one
# transaction (shopping basket), and each column holds one purchased item.
# Rows are ragged (different numbers of items), so we read it as raw text
# and split each line into a list of items rather than using pandas' normal
# rectangular CSV reader.

RAW_PATH = "groceries.csv"

transactions = []
with open(RAW_PATH, "r", encoding="utf-8") as f:
    for line in f:
        items = [item.strip() for item in line.strip().split(",") if item.strip() != ""]
        if items:
            transactions.append(items)

print(f"Number of transactions loaded: {len(transactions)}")

# ---------------------------------------------------------------------------
# 1. UNDERSTAND THE DATA (Question 1)
# ---------------------------------------------------------------------------
n_transactions = len(transactions)

all_items = [item for basket in transactions for item in basket]
unique_items = sorted(set(all_items))
n_items = len(unique_items)

item_counts = pd.Series(all_items).value_counts()
top10_items = item_counts.head(10)

print("\n--- Question 1: Understand the Data ---")
print(f"a. Number of transactions: {n_transactions}")
print(f"b. Number of different items: {n_items}")
print("c. Top 10 most frequent items:")
print(top10_items)

top10_items.to_frame("frequency").to_csv("top10_items.csv")

# Figure 1: Top 10 most frequent items (bar chart)
plt.figure(figsize=(9, 5))
top10_items.sort_values().plot(kind="barh", color="#4C72B0")
plt.xlabel("Number of transactions")
plt.title("Top 10 Most Frequently Purchased Items")
plt.tight_layout()
plt.savefig("figures/fig1_top10_items.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 2. FREQUENT ITEMSETS (Question 2)
# ---------------------------------------------------------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_array, columns=te.columns_)

# Minimum support chosen low enough to surface a healthy number of
# multi-item itemsets in a sparse basket dataset (9,835 transactions,
# 169 items).
MIN_SUPPORT = 0.01

frequent_itemsets = apriori(df_onehot, min_support=MIN_SUPPORT, use_colnames=True)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets["itemsets_str"] = frequent_itemsets["itemsets"].apply(
    lambda x: ", ".join(sorted(x))
)

# Itemsets with 2+ products, sorted by support
multi_itemsets = (
    frequent_itemsets[frequent_itemsets["length"] >= 2]
    .sort_values("support", ascending=False)
    .reset_index(drop=True)
)

print("\n--- Question 2: Frequent Itemsets (>=2 items) ---")
print(f"Total frequent itemsets found (support >= {MIN_SUPPORT}): {len(frequent_itemsets)}")
print(f"Multi-item (2+) frequent itemsets found: {len(multi_itemsets)}")
print(multi_itemsets[["itemsets_str", "support"]].head(15).to_string(index=False))

multi_itemsets[["itemsets_str", "support"]].head(15).to_csv(
    "top15_frequent_itemsets.csv", index=False
)

# Figure 2: Top 10 frequent itemsets (2+ items) by support
top10_itemsets = multi_itemsets.head(10)
plt.figure(figsize=(9, 5))
plt.barh(top10_itemsets["itemsets_str"][::-1], top10_itemsets["support"][::-1], color="#55A868")
plt.xlabel("Support")
plt.title("Top 10 Frequent Itemsets (2+ items) by Support")
plt.tight_layout()
plt.savefig("figures/fig2_top10_itemsets.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. ASSOCIATION RULES (Question 3)
# ---------------------------------------------------------------------------
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
rules = rules[rules["lift"] > 1]  # keep only genuinely useful (positively correlated) rules

rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

rules_sorted = rules.sort_values("lift", ascending=False).reset_index(drop=True)

print(f"\n--- Question 3: Association Rules ---")
print(f"Total rules generated (confidence >= 0.2, lift > 1): {len(rules_sorted)}")

report_cols = [
    "antecedents_str",
    "consequents_str",
    "support",
    "confidence",
    "lift",
]
top10_rules = rules_sorted.head(10)
print(top10_rules[report_cols].to_string(index=False))

rules_sorted[report_cols].head(15).to_csv("top15_rules.csv", index=False)

# ---------------------------------------------------------------------------
# 4. FIND INTERESTING RULES (Question 5)
# ---------------------------------------------------------------------------
rule_highest_conf = rules_sorted.loc[rules_sorted["confidence"].idxmax()]
rule_highest_lift = rules_sorted.loc[rules_sorted["lift"].idxmax()]
rule_highest_support = rules_sorted.loc[rules_sorted["support"].idxmax()]

print("\n--- Question 5: Interesting Rules ---")
print("Highest confidence rule:")
print(rule_highest_conf[report_cols])
print("\nHighest lift rule:")
print(rule_highest_lift[report_cols])
print("\nHighest support rule:")
print(rule_highest_support[report_cols])

summary_rules = pd.DataFrame(
    [rule_highest_conf[report_cols], rule_highest_lift[report_cols], rule_highest_support[report_cols]],
    index=["Highest confidence", "Highest lift", "Highest support"],
)
summary_rules.to_csv("q5_summary_rules.csv")

# Figure 3: scatter plot of support vs confidence, sized/colored by lift
plt.figure(figsize=(7, 6))
sc = plt.scatter(
    rules_sorted["support"],
    rules_sorted["confidence"],
    c=rules_sorted["lift"],
    cmap="viridis",
    s=60,
    edgecolor="k",
    alpha=0.8,
)
plt.colorbar(sc, label="Lift")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association Rules: Support vs Confidence (colored by Lift)")
plt.tight_layout()
plt.savefig("figures/fig3_rules_scatter.png", dpi=150)
plt.close()

# Figure 4: Top 10 rules by lift (bar chart)
top10_lift = rules_sorted.head(10).copy()
top10_lift["rule_label"] = top10_lift["antecedents_str"] + " -> " + top10_lift["consequents_str"]
plt.figure(figsize=(10, 5.5))
plt.barh(top10_lift["rule_label"][::-1], top10_lift["lift"][::-1], color="#C44E52")
plt.xlabel("Lift")
plt.title("Top 10 Association Rules by Lift")
plt.tight_layout()
plt.savefig("figures/fig4_top10_rules_lift.png", dpi=150)
plt.close()

print("\nAll tables (CSV) and figures (PNG) have been saved to the working directory.")
