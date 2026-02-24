"""
Meta-Recipe Expansion Module

This module processes YAML CI pipeline recipes to allow for step inheritance and intelligent overrides,
reducing boilerplate when defining sequential training or processing steps.

Supported Meta-Tokens:
----------------------
1. <repeat_last>
   - As a string: `- <repeat_last>`
     Creates an exact duplicate of the immediately preceding step.
   - As a dictionary key: `- <repeat_last>: { overrides... }`
     Copies the preceding step and applies the specified overrides.

   Merge Behavior:
   - Dictionaries are deep-merged recursively.
   - Lists consisting exclusively of CLI arguments (e.g., strings starting with `--`) are
     merged intelligently via prefix matching (e.g., `--lr=0.01` replaces `--lr=0.001`).
   - All other lists are completely overwritten by the override list, unless a meta-token is used.

2. <replace>
   - Aborts the recursive merge for a specific node.
   - Overwrites the inherited base object entirely with the contents provided.
   - Usage: `key: { <replace>: [new, list] }` or `key: { <replace>: {new: dict} }`

3. <extend>
   - Explicitly adds items to an inherited list instead of overwriting it.
   - Will automatically convert a single item into a list if necessary.
   - <extend> can also take optional slice arguments to specify which portion of the old list to add.
   - Suffixes can be added to overcome .yaml's restriction on duplicate keys.
   - Usage: `binpacks: { <extend>: [new_data.binpack] }`
            `binpacks: { <extend(1,3)_foo>: [data1.binpack, data2.binpack] }`
            `binpacks: { <extend(1,3)(4,)_foo>: }`
   - The first example would extend the inherited `binpacks` list with `new_data.binpack`
     The second example takes the slice `[1 : 3]` of the previous list and adds `data1.binpack`, `data2.binpack`.
     The third example takes the slices `[1 : 3]` and `[4 :]` of the previous list -- i.e. removes elements 3 and 4.

4. <remove>
   - Assigning `"<remove>"` to a key deletes that key entirely from the inherited dictionary.
   - Cannot be used to remove specific string elements from inside a list; use `<replace>` instead.

Constraints & Edge Cases:
-------------------------
- The first step in a sequence cannot use `<repeat_last>`.
- The script enforces that the first step has `resume: none`. Subsequent steps default to `previous_checkpoint`.
- A dictionary containing `<append>` or `<replace>` evaluates immediately. Do not mix other keys into
  the same dictionary level alongside these tokens, as they will be ignored.
- Attempting to `<append>` to an existing non-list primitive will raise a ValueError.
"""

import copy
import re

def intelligent_merge(base, override):
    """
    Recursively deep-merges two dictionaries.
    Applies intelligent argument matching for lists of CLI flags.
    Handles grouped <replace...>, <extend...>, and <remove> meta-tokens with optional suffixes.
    Supports multiple slices for <extend>, e.g., <extend(0,2)(4,-1)_suffix>.
    """
    if isinstance(override, dict):
        merged = copy.deepcopy(base) if isinstance(base, dict) else {}

        for k, v in override.items():
            # 1. Handle Explicit Replacements
            if isinstance(k, str) and k.startswith("<replace"):
                if not isinstance(v, dict):
                    raise ValueError(f"{k} must contain a dictionary of fields to replace.")
                for rep_k, rep_v in v.items():
                    merged[rep_k] = copy.deepcopy(rep_v)

            # 2. Handle Extensions & Multiple Slicing of the BASE list
            elif isinstance(k, str) and k.startswith("<extend"):
                if not isinstance(v, dict):
                    raise ValueError(f"{k} must contain a dictionary of fields to extend.")

                # Extract the full block of parentheses, ignoring the suffix
                match = re.match(r"^<extend((?:\([^)]*\))*)(?:_.*?)?>$", k)
                if not match:
                    raise ValueError(f"Invalid syntax for extend directive: {k}")

                slices_str = match.group(1)
                # Find all individual slice arguments like '1,3' or ',-1'
                slice_args = re.findall(r'\(([^)]*)\)', slices_str) if slices_str else []

                for ext_k, ext_v in v.items():
                    if not isinstance(ext_v, list):
                        ext_v = [ext_v]

                    if ext_k not in merged or merged[ext_k] is None:
                        # If there is no base list, just take the new items
                        merged[ext_k] = copy.deepcopy(ext_v)
                    elif not isinstance(merged[ext_k], list):
                        raise ValueError(f"Cannot extend non-list field '{ext_k}'")
                    else:
                        base_list = merged[ext_k]
                        new_base = []

                        if not slice_args:
                            # No slices specified, keep the whole base list
                            new_base = copy.deepcopy(base_list)
                        else:
                            # Apply each slice to the base list and concatenate
                            for arg in slice_args:
                                parts = [p.strip() for p in arg.split(",")]
                                start, end = None, None
                                if len(parts) >= 1 and parts[0]:
                                    start = int(parts[0])
                                if len(parts) >= 2 and parts[1]:
                                    end = int(parts[1])

                                new_base.extend(copy.deepcopy(base_list[start:end]))

                        # Extend the cleanly sliced base with the new items
                        new_base.extend(copy.deepcopy(ext_v))
                        merged[ext_k] = new_base

            # 3. Handle Explicit Removals
            elif v == "<remove>":
                merged.pop(k, None)

            # 4. Standard Recursive Merge
            elif k in merged:
                merged[k] = intelligent_merge(merged[k], v)

            # 5. Populate New Keys
            else:
                merged[k] = intelligent_merge(None, v)

        return merged

    elif isinstance(override, list):
        if isinstance(base, list):
            # Heuristic: Intelligent CLI args replacement vs full array overwrite
            is_arg_list = len(override) > 0 and all(isinstance(x, str) and x.startswith("--") for x in override)
            if is_arg_list:
                merged = copy.deepcopy(base)
                for item in override:
                    if "=" in item:
                        prefix = item.split("=", 1)[0] + "="
                        replaced = False
                        for i, existing in enumerate(merged):
                            if isinstance(existing, str) and existing.startswith(prefix):
                                merged[i] = item
                                replaced = True
                                break
                        if not replaced:
                            merged.append(item)
                    else:
                        if item not in merged:
                            merged.append(item)
                return merged
        # Standard list without meta-directives overwrites the base entirely
        return copy.deepcopy(override)

    else:
        # Primitives or mismatched types (override replaces base entirely)
        return copy.deepcopy(override)

def enforce_resume_constraint(step, is_first_step, step_index):
    """
    Validates and enforces the 'resume' constraint, modifying the step in-place.
    """
    if "run" not in step:
        step["run"] = {}

    current_resume = step["run"].get("resume")

    if is_first_step:
        if current_resume != "none":
            print(f"WARNING (Step {step_index}): First step must have 'resume: none'. Overriding '{current_resume}' to 'none'.")
            step["run"]["resume"] = "none"
    else:
        valid_resumes = ["previous_model", "previous_checkpoint"]
        if current_resume not in valid_resumes:
            print(f"WARNING (Step {step_index}): 'resume: {current_resume}' is invalid for subsequent steps. Overriding to 'previous_checkpoint'.")
            step["run"]["resume"] = "previous_checkpoint"

def expand_meta_recipe(recipe):
    """
    Expands '<repeat_last>' directives and applies overrides in the training steps sequentially.
    Modifies the recipe dictionary in-place.
    """
    if "training" not in recipe or "steps" not in recipe["training"]:
        return recipe

    original_steps = recipe["training"]["steps"]
    if not original_steps:
        return recipe

    expanded_steps = []
    previous_step = None

    for i, step in enumerate(original_steps):
        is_first = (i == 0)
        step_index = i + 1

        if is_first:
            if step == "<repeat_last>" or (isinstance(step, dict) and "<repeat_last>" in step):
                raise ValueError(f"Step {step_index} cannot use '<repeat_last>'. The base step must be fully defined.")

            new_step = copy.deepcopy(step)
            enforce_resume_constraint(new_step, is_first_step=True, step_index=step_index)
            expanded_steps.append(new_step)
            previous_step = new_step
            continue

        # Handle exact copy
        if step == "<repeat_last>":
            new_step = copy.deepcopy(previous_step)
            enforce_resume_constraint(new_step, is_first_step=False, step_index=step_index)
            expanded_steps.append(new_step)
            previous_step = new_step

        # Handle modified copy
        elif isinstance(step, dict) and "<repeat_last>" in step:
            overrides = step["<repeat_last>"]
            new_step = intelligent_merge(previous_step, overrides)
            enforce_resume_constraint(new_step, is_first_step=False, step_index=step_index)
            expanded_steps.append(new_step)
            previous_step = new_step

        # Handle completely new standalone step
        else:
            new_step = copy.deepcopy(step)
            enforce_resume_constraint(new_step, is_first_step=False, step_index=step_index)
            expanded_steps.append(new_step)
            previous_step = new_step

    recipe["training"]["steps"] = expanded_steps
    return recipe

if __name__ == "__main__":
    import argparse
    import sys
    import yaml

    class IndentSafeDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(IndentSafeDumper, self).increase_indent(flow, False)

    DumperClass = IndentSafeDumper

    parser = argparse.ArgumentParser(
        description="Expand a meta-recipe YAML file containing <repeat_last> directives.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="Path to the input meta-recipe YAML file."
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Optional path to the output YAML file. If omitted, prints to stdout."
    )

    args = parser.parse_args()

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            recipe = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}", file=sys.stderr)
        sys.exit(1)

    # Expand the meta-recipe
    try:
        expanded_recipe = expand_meta_recipe(recipe)
    except ValueError as e:
        print(f"Expansion Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Output handling
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            yaml.dump(expanded_recipe, f, Dumper=DumperClass, default_flow_style=False, sort_keys=False, width=300)
        print(f"Expanded recipe successfully written to {args.output_file}")
    else:
        print(yaml.dump(expanded_recipe, Dumper=DumperClass, default_flow_style=False, sort_keys=False, width=300))
