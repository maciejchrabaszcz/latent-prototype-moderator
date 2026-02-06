import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm


def generate_hidden_states(
    model,
    dataloader,
    save_folder: Path,
    remove_files_if_exists: bool = False,
    calculate_mean_hidden_states: bool = False,
):
    pqwriters = None
    if calculate_mean_hidden_states:
        mean_pqwriters = None
    schema = pa.schema(
        [("labels", pa.float32()), ("hidden_state", pa.list_(pa.float32()))]
    )

    for batch in tqdm(dataloader, desc="Processing", leave=False):
        batch = batch.to(model.device)
        labels = batch.pop("labels", None)
        with torch.no_grad():
            hidden_states = model.forward(**batch, output_hidden_states=True)[
                "hidden_states"
            ]
        if pqwriters is None:
            if remove_files_if_exists:
                for layer_idx in range(len(hidden_states)):
                    cur_file = save_folder / f"layer_{layer_idx}_hidden_states.parquet"
                    if cur_file.exists():
                        shutil.rmtree(
                            cur_file,
                            ignore_errors=True,
                        )
                    if calculate_mean_hidden_states:
                        cur_file = (
                            save_folder
                            / f"layer_{layer_idx}_mean_hidden_states.parquet"
                        )
                        if cur_file.exists():
                            shutil.rmtree(
                                cur_file,
                                ignore_errors=True,
                            )
            pqwriters = {
                layer_idx: pq.ParquetWriter(
                    save_folder / f"layer_{layer_idx}_hidden_states.parquet",
                    schema=schema,
                )
                for layer_idx in range(len(hidden_states))
            }
            if calculate_mean_hidden_states:
                mean_pqwriters = {
                    layer_idx: pq.ParquetWriter(
                        save_folder / f"layer_{layer_idx}_mean_hidden_states.parquet",
                        schema=schema,
                    )
                    for layer_idx in range(len(hidden_states))
                }
        for layer_idx, layer_hidden_states in enumerate(hidden_states):
            data = {
                "labels": labels.detach()
                .to(torch.float32)
                .cpu()
                .numpy()
                .flatten()
                .tolist(),
                "hidden_state": [],
            }
            for prompt_hidden_state in layer_hidden_states:
                data["hidden_state"].append(
                    prompt_hidden_state[-1]
                    .detach()
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                    .flatten()
                )
            if calculate_mean_hidden_states:
                mean_data = {
                    "labels": labels.detach()
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                    .flatten()
                    .tolist(),
                    "hidden_state": [],
                }
                mean_layer_hidden_states = layer_hidden_states.clone()
                attention_mask = batch["attention_mask"]
                # Calculate mean hidden states only for valid tokens
                valid_tokens = attention_mask.sum(dim=1)
                mean_layer_hidden_states = (
                    mean_layer_hidden_states * attention_mask.unsqueeze(-1)
                ).sum(dim=1) / valid_tokens.unsqueeze(-1)
                for mean_hidden_state in mean_layer_hidden_states:
                    mean_data["hidden_state"].append(
                        mean_hidden_state.detach()
                        .to(torch.float32)
                        .cpu()
                        .numpy()
                        .flatten()
                    )

            array_labels = pa.array(data["labels"], type=pa.float32())
            hidden_states_arrays = pa.array(
                data["hidden_state"], type=pa.list_(pa.float32())
            )

            table = pa.Table.from_arrays(
                [array_labels, hidden_states_arrays], schema=schema
            )
            pqwriters[layer_idx].write_table(table)
            if calculate_mean_hidden_states:
                mean_array_labels = pa.array(mean_data["labels"], type=pa.float32())
                mean_hidden_states_arrays = pa.array(
                    mean_data["hidden_state"], type=pa.list_(pa.float32())
                )
                mean_table = pa.Table.from_arrays(
                    [mean_array_labels, mean_hidden_states_arrays], schema=schema
                )
                # Write the mean table to the corresponding layer's file
                mean_pqwriters[layer_idx].write_table(mean_table)
            # Write the table to the corresponding layer's file
    for writer in pqwriters.values():
        writer.close()
