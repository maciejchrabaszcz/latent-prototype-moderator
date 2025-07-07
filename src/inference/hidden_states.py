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
):
    pqwriters = None
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
            pqwriters = {
                layer_idx: pq.ParquetWriter(
                    save_folder / f"layer_{layer_idx}_hidden_states.parquet",
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

            array_labels = pa.array(data["labels"], type=pa.float32())
            hidden_states_arrays = pa.array(
                data["hidden_state"], type=pa.list_(pa.float32())
            )

            table = pa.Table.from_arrays(
                [array_labels, hidden_states_arrays], schema=schema
            )

            # Write the table to the corresponding layer's file
            pqwriters[layer_idx].write_table(table)
    for writer in pqwriters.values():
        writer.close()
