import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from imblearn.over_sampling import SMOTE
import pickle


training_data = pd.read_csv("/home/Oviemena/Learning_Path_Generator/AI/training_data.csv")
X = training_data["text"]
y = training_data["label"]


# Vectorize the text
vectorizer = TfidfVectorizer(
    max_features=150,
    min_df=1,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)
X_vec = vectorizer.fit_transform(X)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# Balance dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Define classifier parameters
param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.01],
    'max_depth': [4, 5],
    'min_samples_split': [2, 3],
    'subsample': [0.8, 0.9]
}



# Initialize classifier with class weights
clf = GradientBoostingClassifier(
    random_state=42,
    validation_fraction=0.2,
    n_iter_no_change=10,
    verbose=1
)


# Grid search with stratified k-fold
grid_search = GridSearchCV(
    clf,
    param_grid,
    cv=5,
    scoring=['accuracy', 'f1_macro'],
    refit='f1_macro',  # Optimize for F1 score
    n_jobs=-1,
    verbose=2
)
# Add early stopping callback
print("\nTraining Progress:")
grid_search.fit(X_train_balanced, y_train_balanced)

# Get best model
best_clf = grid_search.best_estimator_

# Get best model
best_clf = grid_search.best_estimator_


# Add performance monitoring
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)


# Add feature importance analysis
feature_importance = pd.DataFrame({
    'feature': vectorizer.get_feature_names_out(),
    'importance': best_clf.feature_importances_
})
feature_importance['importance_pct'] = feature_importance['importance'] * 100
print("\nTop 10 Most Important Features (%):")
print(feature_importance.nlargest(10, 'importance')[['feature', 'importance_pct']])



# Evaluate on test set
y_pred = best_clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))




# Save model and vectorizer
with open("/home/Oviemena/Learning_Path_Generator/AI/classifier.pkl", "wb") as f:
    pickle.dump({'classifier': best_clf, 'vectorizer': vectorizer}, f)