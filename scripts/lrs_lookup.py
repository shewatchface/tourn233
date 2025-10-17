import json 
import os 
import hashlib
current_dir = os.path.dirname(os.path.abspath(__file__))



def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 


def get_dpo_lr(model: str):
    scale_factor = 1.0
    hashed_model = hash_model(model)
    print(f"model_name: {model}", flush=True)

    config_file = f"{os.path.join(current_dir, 'lrs')}/dpo_{model.split('/', 1)[1]}.json"
    print(f"config_dpo1: {config_file}")
    if os.path.exists(config_file):
        print(f"Config: {config_file}")
        with open(config_file, "r") as f:
            dpo_lrs = json.load(f)
    else:
        config_file = f"{os.path.join(current_dir, 'lrs')}/dpo_{model.split('/', 1)[0]}.json"
        print(f"config_dpo0: {config_file}")
        if os.path.exists(config_file):
            print(f"Config: {config_file}")
            with open(config_file, "r") as f:
                dpo_lrs = json.load(f)
        else:
            config_file = f"{os.path.join(current_dir, 'lrs')}/dpo.json"
            print(f"config_dpo_default: {config_file}")
            if os.path.exists(config_file):
                print(f"Config: {config_file}")
                with open(config_file, "r") as f:
                    dpo_lrs = json.load(f)

    ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_dpo_{model.split('/', 1)[1]}.json"
    print(f"ratio_dpo1: {ratio_file}")
    if os.path.exists(ratio_file):
        print(f"Ratio: {ratio_file}")
        with open(ratio_file, "r") as f:
            ratio_lrs = json.load(f)
            scale_factor = ratio_lrs["ratio"]
    else:
        ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_dpo_{model.split('/', 1)[0]}.json"
        print(f"ratio_dpo0: {ratio_file}")
        if os.path.exists(ratio_file):
            print(f"Ratio: {ratio_file}")
            with open(ratio_file, "r") as f:
                ratio_lrs = json.load(f)
                scale_factor = ratio_lrs["ratio"]
        else:
            ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_dpo.json"
            print(f"ratio_dpo_default: {ratio_file}")
            if os.path.exists(ratio_file):
                print(f"Ratio: {ratio_file}")
                with open(ratio_file, "r") as f:
                    ratio_lrs = json.load(f)
                    scale_factor = ratio_lrs["ratio"]

    for lr in dpo_lrs:
        if lr["h"] == hashed_model:
            lr_return = lr["lr"] * scale_factor
            print(f"scale: {scale_factor}")
            print(f"lr: {lr_return}")
            return lr_return

    return None


def get_grpo_lr(model: str):
    scale_factor = 1.0
    hashed_model = hash_model(model)
    print(f"model_name: {model}", flush=True)

    config_file = f"{os.path.join(current_dir, 'lrs')}/grpo_{model.split('/', 1)[1]}.json"
    print(f"config_grpo1: {config_file}")
    if os.path.exists(config_file):
        print(f"Config: {config_file}")
        with open(config_file, "r") as f:
            grpo_lrs = json.load(f)
    else:
        config_file = f"{os.path.join(current_dir, 'lrs')}/grpo_{model.split('/', 1)[0]}.json"
        print(f"config_grpo0: {config_file}")
        if os.path.exists(config_file):
            print(f"Config: {config_file}")
            with open(config_file, "r") as f:
                grpo_lrs = json.load(f)
        else:
            config_file = f"{os.path.join(current_dir, 'lrs')}/grpo.json"
            print(f"config_grpo_default: {config_file}")
            if os.path.exists(config_file):
                print(f"Config: {config_file}")
                with open(config_file, "r") as f:
                    grpo_lrs = json.load(f)

    ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_grpo_{model.split('/', 1)[1]}.json"
    print(f"ratio_grpo1: {ratio_file}")
    if os.path.exists(ratio_file):
        print(f"Ratio: {ratio_file}")
        with open(ratio_file, "r") as f:
            ratio_lrs = json.load(f)
            scale_factor = ratio_lrs["ratio"]
    else:
        ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_grpo_{model.split('/', 1)[0]}.json"
        print(f"ratio_grpo0: {ratio_file}")
        if os.path.exists(ratio_file):
            print(f"Ratio: {ratio_file}")
            with open(ratio_file, "r") as f:
                ratio_lrs = json.load(f)
                scale_factor = ratio_lrs["ratio"]
        else:
            ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_grpo.json"
            print(f"ratio_grpo_default: {ratio_file}")
            if os.path.exists(ratio_file):
                print(f"Ratio: {ratio_file}")
                with open(ratio_file, "r") as f:
                    ratio_lrs = json.load(f)
                    scale_factor = ratio_lrs["ratio"]

    for lr in grpo_lrs:
        if lr["h"] == hashed_model:
            lr_return = lr["lr"] * scale_factor
            print(f"scale: {scale_factor}")
            print(f"lr: {lr_return}")
            return lr_return

    return None


def get_instruct_lr(model: str):
    scale_factor = 1.0
    hashed_model = hash_model(model)
    print(f"model_name: {model}", flush=True)

    config_file = f"{os.path.join(current_dir, 'lrs')}/instruct_{model.split('/', 1)[1]}.json"
    print(f"config_instruct1: {config_file}")
    if os.path.exists(config_file):
        print(f"Config: {config_file}")
        with open(config_file, "r") as f:
            instruct_lrs = json.load(f)
    else:
        config_file = f"{os.path.join(current_dir, 'lrs')}/instruct_{model.split('/', 1)[0]}.json"
        print(f"config_instruct0: {config_file}")
        if os.path.exists(config_file):
            print(f"Config: {config_file}")
            with open(config_file, "r") as f:
                instruct_lrs = json.load(f)
        else:
            config_file = f"{os.path.join(current_dir, 'lrs')}/instruct.json"
            print(f"config_instruct_default: {config_file}")
            if os.path.exists(config_file):
                print(f"Config: {config_file}")
                with open(config_file, "r") as f:
                    instruct_lrs = json.load(f)

    ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_instruct_{model.split('/', 1)[1]}.json"
    print(f"ratio_instruct1: {ratio_file}")
    if os.path.exists(ratio_file):
        print(f"Ratio: {ratio_file}")
        with open(ratio_file, "r") as f:
            ratio_lrs = json.load(f)
            scale_factor = ratio_lrs["ratio"]
    else:
        ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_instruct_{model.split('/', 1)[0]}.json"
        print(f"ratio_instruct0: {ratio_file}")
        if os.path.exists(ratio_file):
            print(f"Ratio: {ratio_file}")
            with open(ratio_file, "r") as f:
                ratio_lrs = json.load(f)
                scale_factor = ratio_lrs["ratio"]
        else:
            ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_instruct.json"
            print(f"ratio_instruct_default: {ratio_file}")
            if os.path.exists(ratio_file):
                print(f"Ratio: {ratio_file}")
                with open(ratio_file, "r") as f:
                    ratio_lrs = json.load(f)
                    scale_factor = ratio_lrs["ratio"]

    for lr in instruct_lrs:
        if lr["h"] == hashed_model:
            lr_return = lr["lr"] * scale_factor
            print(f"scale: {scale_factor}")
            print(f"lr: {lr_return}")
            return lr_return

    return None


def get_grpo_python_lr(model: str):
    scale_factor = 1.0
    hashed_model = hash_model(model)
    print(f"model_name: {model}", flush=True)

    config_file = f"{os.path.join(current_dir, 'lrs')}/grpo_python_{model.split('/', 1)[1]}.json"
    print(f"config_grpo_python1: {config_file}")
    if os.path.exists(config_file):
        print(f"Config: {config_file}")
        with open(config_file, "r") as f:
            grpo_python_lrs = json.load(f)
    else:
        config_file = f"{os.path.join(current_dir, 'lrs')}/grpo_python_{model.split('/', 1)[0]}.json"
        print(f"config_grpo_python0: {config_file}")
        if os.path.exists(config_file):
            print(f"Config: {config_file}")
            with open(config_file, "r") as f:
                grpo_python_lrs = json.load(f)
        else:
            config_file = f"{os.path.join(current_dir, 'lrs')}/grpo_python.json"
            print(f"config_grpo_python_default: {config_file}")
            if os.path.exists(config_file):
                print(f"Config: {config_file}")
                with open(config_file, "r") as f:
                    grpo_python_lrs = json.load(f)

    ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_grpo_python_{model.split('/', 1)[1]}.json"
    print(f"ratio_grpo_python1: {ratio_file}")
    if os.path.exists(ratio_file):
        print(f"Ratio: {ratio_file}")
        with open(ratio_file, "r") as f:
            ratio_lrs = json.load(f)
            scale_factor = ratio_lrs["ratio"]
    else:
        ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_grpo_python_{model.split('/', 1)[0]}.json"
        print(f"ratio_grpo_python0: {ratio_file}")
        if os.path.exists(ratio_file):
            print(f"Ratio: {ratio_file}")
            with open(ratio_file, "r") as f:
                ratio_lrs = json.load(f)
                scale_factor = ratio_lrs["ratio"]
        else:
            ratio_file = f"{os.path.join(current_dir, 'lrs')}/ratio_grpo_python.json"
            print(f"ratio_grpo_python_default: {ratio_file}")
            if os.path.exists(ratio_file):
                print(f"Ratio: {ratio_file}")
                with open(ratio_file, "r") as f:
                    ratio_lrs = json.load(f)
                    scale_factor = ratio_lrs["ratio"]

    for lr in grpo_python_lrs:
        if lr["h"] == hashed_model:
            lr_return = lr["lr"] * scale_factor
            print(f"scale: {scale_factor}")
            print(f"lr: {lr_return}")
            return lr_return

    return None
