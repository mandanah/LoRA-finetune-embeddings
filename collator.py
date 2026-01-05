from schema import PairBatch


def collator_function(batch, tokenizer, max_length):
    anchors, positives = zip(*batch)

    anchor_encodings = tokenizer(
        list(anchors),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    positive_encodings = tokenizer(
        list(positives),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return PairBatch(
        anchor_input_ids=anchor_encodings["input_ids"],
        anchor_attention_mask=anchor_encodings["attention_mask"],
        positive_input_ids=positive_encodings["input_ids"],
        positive_attention_mask=positive_encodings["attention_mask"],
    )
