# Generation

The `generation_types` module in `generation_types.py` handles all prompt creation, and the prompts are inputted to the generation model in `main.py`.
Thus, if the prompt formats are the same, one can just modify `main.py` to change the generation model.
However, if the changes to prompt formats are required, one needs to change corresponding module in `generation_types.py`.