import os
import json

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(base_dir, "..", "huggingface_models/ner-model")

best_ckpt = None
best_score = 0

for folder in os.listdir(checkpoint_dir):
    if folder.startswith("checkpoint-"):
        state_file = os.path.join(checkpoint_dir, folder, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file) as f:
                data = json.load(f)
                for log in data["log_history"]:
                    if "eval_f1" in log:
                        if log["eval_f1"] > best_score:
                            best_score = log["eval_f1"]
                            best_ckpt = folder

print(f"✅ Best checkpoint: {best_ckpt} with F1: {best_score}")
