from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_dataset(csv_path: Path) -> pd.DataFrame:
	"""Load Iris dataset from CSV file."""
	if not csv_path.exists():
		raise FileNotFoundError(f"Dataset not found: {csv_path}")
	return pd.read_csv(csv_path)


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	csv_path = base_dir / "Iris.csv"

	iris_df = load_dataset(csv_path)

	feature_columns = [
		"SepalLengthCm",
		"SepalWidthCm",
		"PetalLengthCm",
		"PetalWidthCm",
	]
	target_column = "Species"

	X = iris_df[feature_columns]
	y = iris_df[target_column]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	model = DecisionTreeClassifier(random_state=42)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	accuracy = accuracy_score(y_test, y_pred)
	matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
	report = classification_report(y_test, y_pred)

	print("=== Iris Flower Classification ===")
	print(f"Samples: {len(iris_df)}")
	print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
	print(f"Features: {', '.join(feature_columns)}")
	print(f"Target: {target_column}\n")

	print("Model: DecisionTreeClassifier")
	print(f"Accuracy on test data: {accuracy:.4f}\n")

	print("Confusion Matrix (rows=true, cols=pred):")
	print("Labels:", list(model.classes_))
	print(matrix)
	print()

	print("Classification Report:")
	print(report)

	print("=== Basic Classification Concepts ===")
	print("1) Features are input measurements (sepal/petal length & width).")
	print("2) The target is the species label (setosa, versicolor, virginica).")
	print("3) Train/Test split checks performance on unseen data.")
	print("4) Accuracy = correctly predicted samples / total test samples.")
	print("5) Precision/Recall/F1 show class-wise performance quality.")


if __name__ == "__main__":
	main()
