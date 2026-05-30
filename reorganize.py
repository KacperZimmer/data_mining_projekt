import os
import shutil

# Define the logical directories and the files/folders they should contain
project_structure = {
    "data": [
        "f1_data_raw",
        "f1_data_cleaned"
    ],
    "notebooks": [
        "eda_f1.ipynb",
        "datamining_NN.ipynb",
        "Model XGBoost.ipynb",
        "Model Random Forest.ipynb",
        "f1_decision_trees_cart.ipynb"
    ],
    "reports": [
        "EDA_Raport.md",
        "raport_XGBoost.md",
        "raport_random_forest.md",
        "Model Drzewa Decyzyjnego.md",
        "raport_random_forest_a_XGBoost.md",
        "Model Wielowarstwowej Sieci Neuronowej.md"
    ],
    "src": [
        "data_prep.py"
    ]
}

def reorganize_project():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for target_dir, items in project_structure.items():
        # Create target directory if it doesn't exist
        dir_path = os.path.join(base_dir, target_dir)
        os.makedirs(dir_path, exist_ok=True)
        
        for item in items:
            src_path = os.path.join(base_dir, item)
            dst_path = os.path.join(dir_path, item)
            
            # Check if the file/folder exists in the root directory before moving
            if os.path.exists(src_path):
                print(f"Moving '{item}' -> '{target_dir}/'")
                shutil.move(src_path, dst_path)
            else:
                print(f"Skipped: '{item}' (not found in root)")

    print("\nReorganization complete! Don't forget to update the file paths in your notebooks/scripts.")

if __name__ == "__main__":
    reorganize_project()
