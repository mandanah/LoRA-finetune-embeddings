import torch
from loss_function import multiple_negative_ranking_loss
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_NAME, SAVE_DIR, MAX_LENGTH
from data.dataset import PairDataset
from model import EncoderModel, build_lora_encoder
from torch.utils.data import DataLoader
import torch.optim as optim
from data.load_toy_data import load_toy_data
from collator import collator_function
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_toy_data()
    ds = PairDataset(pairs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: collator_function(
            x, tokenizer=tokenizer, max_length=MAX_LENGTH
        ),
    )

    model = build_lora_encoder(MODEL_NAME)
    model = EncoderModel(model).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=LEARNING_RATE)

    total_steps = EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            a_input_ids = batch.anchor_input_ids.to(device)
            p_input_ids = batch.positive_input_ids.to(device)
            a_attention_mask = batch.anchor_attention_mask.to(device)
            p_attention_mask = batch.positive_attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            a_embeddings = model(a_input_ids, a_attention_mask)
            p_embeddings = model(p_input_ids, p_attention_mask)

            loss = multiple_negative_ranking_loss(a_embeddings, p_embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"lora_finetuned_epoch{epoch + 1}.pth")
    model.eval()
    model.base_model.save_pretrained(SAVE_DIR)  # saves LoRA adapter + config
    tokenizer.save_pretrained(SAVE_DIR)  # saves tokenizer files


if __name__ == "__main__":
    train()
