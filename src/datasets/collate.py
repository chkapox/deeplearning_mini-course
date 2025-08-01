import torch


def collate_fn(dataset_items: list[dict]):
    assert len(dataset_items) > 0
    F = dataset_items[0]["data_object"].shape[-2]
    max_T = max(x["data_object"].shape[-1] for x in dataset_items)

    B = len(dataset_items)
    batch_data = torch.zeros((B, 1, F, max_T), dtype=dataset_items[0]["data_object"].dtype)
    labels = torch.empty((B,), dtype=torch.long)

    for i, item in enumerate(dataset_items):
        x = item["data_object"]
        T_i = x.shape[-1]
        batch_data[i, :, :, :T_i] = x
        labels[i] = int(item["labels"])

    return {"data_object": batch_data, "labels": labels}
