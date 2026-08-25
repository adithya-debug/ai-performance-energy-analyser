import time
import psutil
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ==========================================================
# AI PERFORMANCE AND ENERGY ANALYZER
# ==========================================================

print("=" * 60)
print("AI PERFORMANCE AND ENERGY ANALYZER")
print("=" * 60)


# ----------------------------------------------------------
# 1. Load dataset
# ----------------------------------------------------------

print("\nLoading dataset...")

iris = load_iris()

X = iris.data
y = iris.target


# ----------------------------------------------------------
# 2. Split dataset
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ----------------------------------------------------------
# 3. Define machine-learning models
# ----------------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "K-Nearest Neighbors":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Support Vector Machine":
        SVC()
}


# ----------------------------------------------------------
# 4. Store results
# ----------------------------------------------------------

results = []


# ----------------------------------------------------------
# 5. Test every model
# ----------------------------------------------------------

for model_name, model in models.items():

    print("\n" + "-" * 50)
    print("Testing:", model_name)
    print("-" * 50)

    # CPU before execution
    psutil.cpu_percent(interval=None)

    # Start timer
    start_time = time.perf_counter()

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # End timer
    end_time = time.perf_counter()

    # CPU measurement
    cpu_usage = psutil.cpu_percent(interval=0.1)

    # Memory measurement
    memory_usage = psutil.virtual_memory().percent

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Execution time
    execution_time = end_time - start_time

    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    print(
        "Execution Time:",
        round(execution_time, 6),
        "seconds"
    )

    print(
        "CPU Usage:",
        cpu_usage,
        "%"
    )

    print(
        "Memory Usage:",
        memory_usage,
        "%"
    )


    # Store result
    results.append({

        "Model": model_name,

        "Accuracy (%)":
            round(accuracy * 100, 2),

        "Execution Time (seconds)":
            round(execution_time, 6),

        "CPU Usage (%)":
            cpu_usage,

        "Memory Usage (%)":
            memory_usage
    })


# ----------------------------------------------------------
# 6. Create results table
# ----------------------------------------------------------

results_df = pd.DataFrame(results)


# ----------------------------------------------------------
# 7. Display final comparison
# ----------------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL AI PERFORMANCE COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ----------------------------------------------------------
# 8. Save results
# ----------------------------------------------------------

output_file = "results/model_performance.csv"

results_df.to_csv(
    output_file,
    index=False
)


print("\n")
print("=" * 60)
print("RESULTS SAVED SUCCESSFULLY")
print("=" * 60)

print(
    "File:",
    output_file
)

print("\nProject completed successfully!")