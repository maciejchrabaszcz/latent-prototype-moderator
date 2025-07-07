import glob
import json
import os
import random
import re
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd

from datasets import Dataset, load_dataset




def load_aegis_dataset():
    dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0")["test"]
    text = dataset["text"]
    text_type = dataset["text_type"]
    labels_0 = dataset["labels_0"]
    labels_1 = dataset["labels_1"]
    labels_2 = dataset["labels_2"]

    output = []
    for i in range(len(text)):
        if text_type[i] != "user_message":
            continue  # Skip non-user messages.

        prompt = text[i]
        label_0 = int(labels_0[i] == "Safe")
        label_1 = int(labels_1[i] == "Safe")
        label_2 = int(labels_2[i] == "Safe")
        prompt_harmfulness = True if label_0 + label_1 + label_2 <= 1 else False

        output.append(
            {
                "prompt": prompt.strip(),
                "response": "",  # Pass empty response in case classifier requires it
                "prompt_harmfulness": prompt_harmfulness,
                "annotations": [labels_0[i], labels_1[i], labels_2[i]],
            }
        )
    return output



def load_beavertails() -> list[dict]:
    dataset = load_dataset("PKU-Alignment/BeaverTails")
    test_dataset = dataset["30k_test"]
    prompts = test_dataset["prompt"]
    responses = test_dataset["response"]
    safety = test_dataset["is_safe"]
    category = test_dataset["category"]
    formatted_dataset = [
        {
            "prompt": prompt,
            "response": responses,
            "response_harmfulness": True if not safe else False,
            "category": category,
        }
        for prompt, responses, safe, category in zip(
            prompts, responses, safety, category
        )
    ]
    return formatted_dataset



HARMBENCH_PATH = "safety-eval/evaluation/tasks/classification/harmbench/"


def load_harmbench_prompt(folder_path=HARMBENCH_PATH) -> list[dict]:
    datapath = os.path.join(folder_path, "harmbench_behaviors_text_test.csv")
    df = pd.read_csv(datapath, sep=",")

    # Filter to only "standard" or "copyright" FunctionalCategory
    # (remove "contextual" and "multimodal")
    df = df[df["FunctionalCategory"].isin(["standard", "copyright"])]
    return [
        {
            "prompt": row["Behavior"],
            "response": "",  # Pass empty response in case classifier requires it
            "prompt_harmfulness": True,
            "FunctionalCategory": row["FunctionalCategory"],
        }
        for i, row in df.iterrows()
    ]


def load_harmbench_response(folder_path=HARMBENCH_PATH) -> list[dict]:
    datapath = os.path.join(folder_path, "harmbench_text_behaviors_val_set.json")

    with open(datapath, "r") as f:
        raw_data = json.load(f)
    keys = list(raw_data.keys())
    all_data = []

    def preprocess_data(x):
        # x includes human_0, human_1, human_2.
        # if more than two have 1, then it is harmful.
        # if more than two have 0, then it is non-harmful.
        human_0 = x["human_0"]
        human_1 = x["human_1"]
        human_2 = x["human_2"]

        harmfulness = True if sum([human_0, human_1, human_2]) > 1 else False
        is_adversarial = True if x["experiment"] != "default" else False
        return {
            "prompt": x["test_case"],
            "response": x["generation"],
            "response_harmfulness": harmfulness,
            "prompt_type": "adversarial" if is_adversarial else "vanilla",
        }

    for k in keys:
        all_data.extend([preprocess_data(x) for x in raw_data[k]])
    return all_data




def load_openai(current_path = "safety-eval/evaluation/tasks/classification/openai_mod/") -> list[dict]:

    datapath = os.path.join(current_path, "openai_moderation_evaluation_dataset.jsonl")
    df = pd.read_json(datapath, orient="records", lines=True)
    harmful_class_names = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]

    return [
        {
            "prompt": row["prompt"],
            "response": "",  # Pass empty response in case classifier requires it
            "prompt_harmfulness": True
            if any(row[class_name] == 1 for class_name in harmful_class_names)
            else False,
            "moderation_categories": [
                class_name for class_name in harmful_class_names if row[class_name]
            ],
        }
        for i, row in df.iterrows()
    ]




def load_saferlhf() -> list[dict]:
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", revision="v0")
    test_dataset = dataset["test"]

    prompts = test_dataset["prompt"]
    response_0 = test_dataset["response_0"]
    response_1 = test_dataset["response_1"]
    is_response_0_safe = test_dataset["is_response_0_safe"]
    is_response_1_safe = test_dataset["is_response_1_safe"]

    safe_responses, harmful_responses = [], []
    for p, r0, r1, r0s, r1s in zip(
        prompts, response_0, response_1, is_response_0_safe, is_response_1_safe
    ):
        # pick only when r0s != r1s
        if r0s == r1s:
            continue
        if r0s:
            safe_responses.append(
                {
                    "prompt": p,
                    "response": r0,
                    "response_harmfulness": False,
                }
            )
        else:
            harmful_responses.append(
                {
                    "prompt": p,
                    "response": r0,
                    "response_harmfulness": True,
                }
            )

        if r1s:
            safe_responses.append(
                {
                    "prompt": p,
                    "response": r1,
                    "response_harmfulness": False,
                }
            )
        else:
            harmful_responses.append(
                {
                    "prompt": p,
                    "response": r1,
                    "response_harmfulness": True,
                }
            )

    # pick 1000 from each category
    random.seed(42)
    indices = random.sample(range(len(safe_responses)), 1000)
    return [safe_responses[i] for i in indices] + [
        harmful_responses[i] for i in indices
    ]




def load_simplesafety_tests() -> list[dict]:
    dataset = load_dataset("Bertievidgen/SimpleSafetyTests")["test"]
    prompts = dataset["prompt"]
    categories = dataset["category"]
    harm_areas = dataset["harm_area"]

    output = []
    for prompt, category, harm_area in zip(prompts, categories, harm_areas):
        output.append(
            {
                "prompt": prompt,
                "response": "",  # Pass empty response in case classifier requires it
                "prompt_harmfulness": True,  # always harmful
                "category": category,
                "harm_area": harm_area,
            }
        )
    return output




def load_toxicchat_all():
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    test_dataset = dataset["test"]
    prompts = test_dataset["user_input"]
    toxicity = test_dataset["toxicity"]
    formatted_dataset = [
        {
            "prompt": prompt,
            "response": "",  # Pass empty response in case classifier requires it
            "prompt_harmfulness": True if toxic else False,
        }
        for prompt, toxic in zip(prompts, toxicity)
    ]
    return formatted_dataset


def load_toxicchat_humanannotated():
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    test_dataset = dataset["test"]
    prompts = test_dataset["user_input"]
    human_annotation = test_dataset["human_annotation"]
    toxicity = test_dataset["toxicity"]
    jailbreaking = test_dataset["jailbreaking"]
    openai_moderation = test_dataset["openai_moderation"]

    human_annotated_dataset = []
    for prompt, annotated, toxic, jb, openai_mod in zip(
        prompts, human_annotation, toxicity, jailbreaking, openai_moderation
    ):
        if annotated:
            openai_mod = eval(openai_mod)  # Convert string to list[tuple[str, float]]
            # pick the str which have highest score
            openai_mod, score = max(openai_mod, key=lambda x: x[1])
            if score < 0.5:
                openai_mod = "low_confidence"

            human_annotated_dataset.append(
                {
                    "prompt": prompt,
                    "response": "",  # Pass empty response in case classifier requires it
                    "prompt_harmfulness": True if toxic else False,
                    "category_adversarial": "adversarial" if jb else "vanilla",
                    "category_openai_moderation": openai_mod,
                }
            )
    return human_annotated_dataset







def load_wildguardtest_prompt(include_response=True) -> list[dict]:
    df = load_dataset("allenai/wildguardmix", "wildguardtest")["test"]
    df = df.to_pandas()
    # drop if prompt_harm_label is None
    df = df.dropna(subset=["prompt_harm_label"])
    # setup prompt_type based on "adversarial"
    df["prompt_type"] = df.apply(
        lambda row: "adversarial" if row["adversarial"] else "vanilla", axis=1
    )
    df["gt_label"] = df["prompt_harm_label"]

    return [
        {
            "prompt": row["prompt"],
            "response": row["response"] if include_response else "",
            # Pass empty response in case classifier requires it
            "prompt_harmfulness": True if row["gt_label"] == "harmful" else False,
            "prompt_type": row["prompt_type"],
            "prompt_harm_category": row["subcategory"],
        }
        for i, row in df.iterrows()
    ]




def load_wildguardtest_response():
    df = load_dataset("allenai/wildguardmix", "wildguardtest")["test"]
    df = df.to_pandas()
    # drop if prompt_harm_label is None
    df = df.dropna(subset=["response_harm_label"])
    # setup prompt_type based on "adversarial"
    df["prompt_type"] = df.apply(
        lambda row: "adversarial" if row["adversarial"] else "vanilla", axis=1
    )
    df["gt_label"] = df["response_harm_label"]
    df["prompt_harm_category"] = df["subcategory"]

    return [
        {
            "prompt": row["prompt"],
            "response": row["response"],
            "response_harmfulness": True if row["gt_label"] == "harmful" else False,
            "prompt_type": row["prompt_type"],
            "prompt_harmfulness": row["prompt_harm_label"] == "harmful",
            "prompt_harm_category": row["subcategory"],
        }
        for i, row in df.iterrows()
    ]



def load_xstest_response():
    dataset = load_dataset("allenai/xstest-response", split="response_harmfulness")
    return [
        {
            "prompt": row["prompt"],
            "response": row["response"],
            "response_harmfulness": row["label"] == "harmful",
            "prompt_type": row["prompt_type"],
            "prompt_harmfulness": row["prompt_type"] == "prompt_harmful",
            "prompt_harm_category": row["prompt_harm_category"],
        }
        for row in dataset
    ]








def load_alpaca_eval():
    """
    Load the task data. It will load data to self.data.

    Returns:
        List of task data.
    """
    alpaca_eval_data = load_dataset(
        "tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True
    )["eval"]
    instructions = [{"instruction": row["instruction"]} for row in alpaca_eval_data]
    return instructions



def load_questions(
    question_file: str, extract_first_turns_as_instructions: bool = True
) -> list[dict]:
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                cur_examples = json.loads(line)
                if extract_first_turns_as_instructions:
                    cur_examples["instruction"] = cur_examples["turns"][0].strip()
                questions.append(cur_examples)
    return questions


def load_mtbench(mtbench_path="safety-eval/evaluation/tasks/generation/mtbench/"):
    question_file = os.path.join(mtbench_path, "question.jsonl")
    questions = load_questions(question_file)
    return questions




def download_gsm8k(save_dir):
    current_dir = os.path.abspath(save_dir)
    os.system(
        f"wget -P {current_dir} https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/test.jsonl"
    )


# These examplars are from the Table 20 of CoT paper (https://arxiv.org/pdf/2201.11903.pdf).
GSM_EXEMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot_answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        "short_answer": "6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot_answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
        "short_answer": "5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "cot_answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
        "short_answer": "39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "cot_answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
        "short_answer": "8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "cot_answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
        "short_answer": "9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "cot_answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
        "short_answer": "29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "cot_answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
        "short_answer": "33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot_answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
        "short_answer": "8",
    },
]


def load_gsm8k(
    save_dir: str = "safety-eval/evaluation/tasks/generation/gsm8k/",
    MAX_NUM_EXAMPLES_PER_TASK=0,
    n_shot=0,
    no_cot=False,
):
    if not os.path.exists(os.path.join(save_dir, "test.jsonl")):
        download_gsm8k(save_dir)

    test_data = []
    with open(os.path.join(save_dir, "test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append(
                {
                    "question": example["question"],
                    "answer": example["answer"].split("####")[1].strip(),
                }
            )

    # some numbers are in the `x,xxx` format, and we want to remove the comma
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(
            example["answer"]
        ), f"answer is not a valid number: {example['answer']}"

    random.seed(42)
    if MAX_NUM_EXAMPLES_PER_TASK and len(test_data) > MAX_NUM_EXAMPLES_PER_TASK:
        test_data = random.sample(test_data, MAX_NUM_EXAMPLES_PER_TASK)

    # get the exemplars
    if n_shot > 0:
        if len(GSM_EXEMPLARS) > n_shot:
            gsm_exemplers = random.sample(GSM_EXEMPLARS, n_shot)
        else:
            gsm_exemplers = GSM_EXEMPLARS
        demonstrations = []
        for example in gsm_exemplers:
            if no_cot:
                demonstrations.append(
                    "Question: "
                    + example["question"]
                    + "\n"
                    + "Answer: "
                    + example["short_answer"]
                )
            else:
                demonstrations.append(
                    "Question: "
                    + example["question"]
                    + "\n"
                    + "Answer: "
                    + example["cot_answer"]
                )
        prompt_prefix = (
            "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
        )
    else:
        prompt_prefix = "Answer the following question.\n\n"

    loaded_data = []
    for data in test_data:
        question = data["question"]
        loaded_data.append(
            {
                "instruction": prompt_prefix + "Question: " + question.strip(),
                "answer": data["answer"],
            }
        )

    return loaded_data




def download_bbh(current_dir):
    # use the following url_format for the Big-Bench-Hard dataset
    url = "https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip"
    os.makedirs(f"{current_dir}/bbh_download", exist_ok=True)
    os.system(f"wget {url} -O {current_dir}/bbh_data.zip")
    os.system(f"unzip {current_dir}/bbh_data.zip -d {current_dir}/bbh_download")
    os.system(
        f"mv {current_dir}/bbh_download/BIG-Bench-Hard-main/* {current_dir} && "
        f"rm -r {current_dir}/bbh_download {current_dir}/bbh_data.zip"
    )


def load_bbh(
    save_dir="safety-eval/evaluation/tasks/generation/bbh",
    no_cot=False,
    MAX_NUM_EXAMPLES_PER_TASK=40,
):
    current_dir = os.path.abspath(save_dir)
    if not os.path.exists(os.path.join(current_dir, "bbh")):
        download_bbh(current_dir)

    task_files = glob.glob(os.path.join(current_dir, "bbh", "*.json"))
    print(task_files)
    all_tasks = {}
    random.seed(42)
    for task_file in task_files:
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]
            all_tasks[task_name] = random.sample(
                all_tasks[task_name], MAX_NUM_EXAMPLES_PER_TASK
            )

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(current_dir, "cot-prompts", "*.txt"))
    for cot_prompt_file in cot_prompt_files:
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])

        if no_cot:
            prompt_fields = task_prompt.split("\n\n")
            new_prompt_fields = []
            for prompt_field in prompt_fields:
                if prompt_field.startswith("Q:"):
                    assert (
                        "So the answer is" in prompt_field
                    ), f"`So the answer is` not found in prompt field of {task_name}.txt."
                    assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                    answer = prompt_field.split("So the answer is")[-1].strip()
                    question = prompt_field.split("\nA:")[0].strip()
                    new_prompt_fields.append(question + "\nA: " + answer)
                else:
                    new_prompt_fields.append(prompt_field)
            task_prompt = "\n\n".join(new_prompt_fields)
        all_prompts[task_name] = task_prompt

    assert set(all_tasks.keys()) == set(
        all_prompts.keys()
    ), "task names in task data and task prompts are not the same."

    task_data = [
        {
            "task_name": task_name,
            "task_inputs": all_tasks[task_name],
            "cot_prompt": all_prompts[task_name],
        }
        for task_name in all_tasks.keys()
    ]

    loaded_data = []
    # input_prompts, targets, task_names = [], [], []
    for data in task_data:
        cot_prompt = data["cot_prompt"]
        task_inputs = data["task_inputs"]
        for task_input in task_inputs:
            loaded_data.append(
                {
                    "instruction": cot_prompt.strip() + "\n\nQ: " + task_input["input"],
                    "target": task_input["target"],
                    "task_name": data["task_name"],
                }
            )

    return loaded_data



import gzip


def download_codex(current_dir):
    os.system(
        f"wget -P {current_dir} https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
    )
    os.system(
        f"wget -P {current_dir} "
        f"https://huggingface.co/datasets/bigcode/humanevalpack/raw/main/data/python/data/humanevalpack.jsonl"
    )


def stream_jsonl(filename: str):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def read_problems(evalset_file: str) -> dict[str, dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def load_codex(save_dir="safety-eval/evaluation/tasks/generation/codex_eval"):
    current_dir = os.path.abspath(save_dir)
    if not os.path.exists(os.path.join(current_dir, "HumanEval.jsonl.gz")):
        download_codex(current_dir)

    test_data = list(
        read_problems(os.path.join(current_dir, "HumanEval.jsonl.gz")).values()
    )

    # these stop sequences are those mentioned in the codex paper.
    stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]

    data_file_hep_path = os.path.join(current_dir, "humanevalpack.jsonl")
    with open(data_file_hep_path) as fin:
        instructions = [json.loads(l) for l in fin]
        instructions_dict = {
            x["task_id"].replace("Python", "HumanEval"): x["instruction"]
            for x in instructions
        }
        stop_sequences.append("\n```")

    loaded_data = []
    for example in test_data:
        instruction = instructions_dict[example["task_id"]]
        loaded_data.append({"instruction": instruction, **example})

    return loaded_data



BEST_COL = "Best Answer"
ANSWER_COL = "Correct Answers"
INCORRECT_COL = "Incorrect Answers"


def download_truthfulqa(current_dir):
    os.system(
        f"wget -P {current_dir}/data/eval/truthfulqa https://github.com/sylinrl/TruthfulQA/raw/main/TruthfulQA.csv"
    )


def load_truthfulqa(
    save_dir="safety-eval/evaluation/tasks/generation/truthfulqa",
) -> list[dict]:
    current_dir = os.path.abspath(save_dir)
    if not os.path.exists(os.path.join(current_dir, "data/eval/truthfulqa")):
        download_truthfulqa(current_dir=current_dir)

    questions = pd.read_csv(
        os.path.join(current_dir, "data/eval/truthfulqa/TruthfulQA.csv")
    )

    return questions
from datasets import Dataset, get_dataset_config_names, load_dataset
from tqdm import tqdm

DATA_NAME = "edinburgh-dawg/mmlu-redux"

def load_mmlu_r(data_name: str = DATA_NAME, split: str = "test"):
    # list all configs for this dataset
    configs = get_dataset_config_names(data_name)

    reformatted_data = []
    POSSIBLE_ANSWERS = ["A", "B", "C", "D"]
    for subset in tqdm(configs):
        try:
            subset_data = load_dataset(data_name, subset, split=split)
        except Exception as e:
            print(f"Error loading subset {subset}: {e}")
            continue

        for index, item in enumerate(subset_data):
            correct_answer = None
            if item["error_type"] == "ok":
                correct_answer = POSSIBLE_ANSWERS[int(item["answer"])]
            elif item["error_type"] == "wrong_groundtruth":
                if item["correct_answer"] in list("ABCDEF"):
                    correct_answer = POSSIBLE_ANSWERS[
                        "ABCDEF".index(item["correct_answer"])
                    ]
                if str(item["correct_answer"]) in list("0123"):
                    correct_answer = POSSIBLE_ANSWERS[int(item["correct_answer"])]
                else:
                    continue
            else:
                # multiple answers, bad questions, etc.
                continue
            reformatted_data.append(
                {
                    "id": f"{data_name.replace('/', '-')}-{subset}-#{index}",
                    "question": item["question"],
                    "choices": item["choices"],
                    "correct_answer": correct_answer,
                    "source": data_name,
                    "config": subset,
                    "task_type": "multiple_choice",
                }
            )
    return reformatted_data

if __name__ == "__main__":
    SAVE_FOLDER = Path("data/processed_benchmarks")

    parser = ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=SAVE_FOLDER,
        help="Directory to save the processed benchmarks.",
    )
    args = parser.parse_args()

    save_folder = args.save_dir

    aegis_dataset = Dataset.from_list(load_aegis_dataset())
    aegis_dataset.save_to_disk(save_folder / "harmfulness/prompt/aegis")

    beavertails_dataset = Dataset.from_list(load_beavertails())
    beavertails_dataset.save_to_disk(save_folder / "harmfulness/response/beavertails")

    harmbench_prompt_dataset = Dataset.from_list(load_harmbench_prompt())
    print(harmbench_prompt_dataset)
    harmbench_prompt_dataset.save_to_disk(save_folder / "harmfulness/prompt/harmbench")
    harmbench_response_dataset = Dataset.from_list(load_harmbench_response())
    print(harmbench_response_dataset)
    harmbench_response_dataset.save_to_disk(save_folder / "harmfulness/response/harmbench")

    openai_dataset = Dataset.from_list(load_openai())
    print(openai_dataset)
    openai_dataset.save_to_disk(save_folder / "harmfulness/prompt/openai_mod")

    safe_rlhf_dataset = Dataset.from_list(load_saferlhf())
    print(safe_rlhf_dataset)
    safe_rlhf_dataset.save_to_disk(save_folder / "harmfulness/response/safe_rlhf")
    simplesafety_tests_dataset = Dataset.from_list(load_simplesafety_tests())
    print(simplesafety_tests_dataset)

    simplesafety_tests_dataset.save_to_disk(
        save_folder / "harmfulness/prompt/simplesafety_tests"
    )

    toxicchat_dataset = Dataset.from_list(load_toxicchat_all())
    print(toxicchat_dataset)
    toxicchat_dataset.save_to_disk(save_folder / "harmfulness/prompt/toxicchat")
    toxicchat_humanannotated = Dataset.from_list(load_toxicchat_humanannotated())
    print(toxicchat_humanannotated)
    toxicchat_humanannotated.save_to_disk(
        save_folder / "harmfulness/prompt/toxicchat_humanannotated"
    )

    wildguardtest_prompt_dataset = Dataset.from_list(load_wildguardtest_prompt())
    print(wildguardtest_prompt_dataset)
    wildguardtest_prompt_dataset.save_to_disk(
        save_folder / "harmfulness/prompt/wildguardtest"
    )

    wildguardtest_response_dataset = Dataset.from_list(load_wildguardtest_response())
    print(wildguardtest_response_dataset)
    wildguardtest_response_dataset.save_to_disk(
        save_folder / "harmfulness/response/wildguardtest"
    )

    xs_test_response_dataset = Dataset.from_list(load_xstest_response())
    print(xs_test_response_dataset)
    xs_test_response_dataset.save_to_disk(save_folder / "harmfulness/prompt/xstest")

    save_folder = save_folder / "general_capabilities"
    alpaca_eval_instructions = Dataset.from_list(load_alpaca_eval())
    print(alpaca_eval_instructions)
    alpaca_eval_instructions.save_to_disk(save_folder / "alpaca_eval")

    mtbench_questions = Dataset.from_list(load_mtbench())
    print(mtbench_questions)
    mtbench_questions.save_to_disk(save_folder / "mtbench")

    gsm8k_dataset = Dataset.from_list(load_gsm8k())
    print(gsm8k_dataset)
    gsm8k_dataset.save_to_disk(save_folder / "gsm8k")

    bbh_dataset = Dataset.from_list(load_bbh())
    print(bbh_dataset)
    bbh_dataset.save_to_disk(save_folder / "bbh")

    codex_dataset = Dataset.from_list(load_codex())
    print(codex_dataset)
    codex_dataset.save_to_disk(save_folder / "codex")

    truthfulqa_dataset = Dataset.from_pandas(load_truthfulqa()).rename_column(
        "Question", "instruction"
    )
    print(truthfulqa_dataset)
    truthfulqa_dataset.save_to_disk(save_folder / "truthfulqa")

    mmlu_r_dataset = Dataset.from_list(load_mmlu_r())
    print(mmlu_r_dataset)
    mmlu_r_dataset.save_to_disk(save_folder / "mmlu_r")
