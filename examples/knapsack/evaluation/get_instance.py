import numpy as np
import pickle

def generate_knapsack_dataset(num_instances, num_items):
    dataset = {}
    for i in range(num_instances):
        weights = np.random.randint(1, 51, num_items).tolist()
        values = np.random.randint(10, 101, num_items).tolist()
        capacity = int(0.5 * sum(weights))
        instance_data = {
            "capacity": capacity,
            "num_items": num_items,
            "weights": weights,
            "values": values
        }
        dataset[f"test_{i+1}"] = instance_data
    return dataset

if __name__ == "__main__":
    ds = generate_knapsack_dataset(5, 100)
    with open("knapsack_dataset.pkl", "wb") as f:
        pickle.dump(ds, f)
    print("Knapsack dataset generated and saved to knapsack_dataset.pkl.")
