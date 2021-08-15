import torch
from transformers import BertForNextSentencePrediction, CanineForSequenceClassification, AdamW
from Data import NextLineData, RhymingData
import os
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import softmax
import time
import numpy as np


def train(model, train_dataloader, test_dataloader, optimizer, num_epochs, batch_size, device, model_type, model_filepath, stats_filepath):

    best_accuracy = 0
    t = time.time()

    for epoch in range(1, num_epochs+1):

        print("Epoch:", epoch)

        total_loss = 0
        model.train()

        for i, batch in enumerate(train_dataloader):
            if i % 100 == 99:
                print("\r", end="")
                print(f"{round((time.time() - t) / 60)}m {round((time.time() - t) % 60)}s Epoch {epoch}: Training batch {i + 1} of {len(train_dataloader)}", end="")

            model.zero_grad()

            loss, _ = model(batch[0].to(device), token_type_ids=batch[1].to(device), attention_mask=batch[2].to(device), labels=batch[3].to(device)).values()

            total_loss += loss.item()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        avg_loss = total_loss / (len(train_dataloader) * batch_size)

        print("Average training loss:", avg_loss, "Time:", time.time() - t, end="\r")
        with open(stats_filepath, "a") as stats_file:
            stats_file.write(f"{model_type.upper()} Epoch {epoch}: Average training loss {avg_loss}, Time {time.time()-t}\n")

        # validation

        total_loss = 0
        total_acc = 0

        model.eval()

        for i, batch in enumerate(test_dataloader):
            if i % 100 == 99:
                print("\r", end="")
                print(f"{round((time.time() - t) / 60)}m {round((time.time() - t) % 60)}s Epoch {epoch}: Training {i + 1} of {len(test_dataloader)}", end="")

            with torch.no_grad():
                loss, logits = model(batch[0].to(device), token_type_ids=batch[1].to(device), attention_mask=batch[2].to(device), labels=batch[3].to(device)).values()

            total_loss += loss.item()

            predictions = np.argmax(softmax(logits, dim=1).cpu().numpy(), axis=1)
            total_acc += sum(predictions == batch[3].numpy())

        avg_loss = total_loss / (len(test_dataloader) * batch_size)
        avg_acc = total_acc / (len(test_dataloader) * batch_size)

        print("Average validation loss:", avg_loss, ", Average validation accuracy:", avg_acc, "Time:", time.time() - t, end="\r")
        with open(stats_filepath, "a") as stats_file:
            stats_file.write(f"{model_type.upper()} Epoch {epoch}: Average validation loss {avg_loss}, Average validation accuracy {avg_acc}, Time {time.time()-t}\n")

        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            torch.save(model, model_filepath)

    print(f"Training Finished: Best accuracy {best_accuracy}, Time {time.time() - t}")
    with open(stats_filepath, "a") as stats_file:
        stats_file.write(f"{model_type.upper()} Best accuracy {best_accuracy}, Time {time.time() - t}\n")


if __name__ == "__main__":

    prefix = "rhyming_poetry_10000ex_3epochs."
    stats_filepath = prefix + "stats"

    for model_type in ["Canine", "Bert"]:
        model_filepath = prefix + model_type.lower()
        num_examples = 100000
        batch_size = 1
        learning_rate = 2e-5
        num_epochs = 3

        directory = r'rhymedata\english_raw'
        data = RhymingData([os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".txt")])
        train_dataloader, validation_dataloader = data.preprocess(model_type, batch_size, num_examples)

        if model_type == "Bert":
            model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
        elif model_type == "Canine":
            model = CanineForSequenceClassification.from_pretrained('google/canine-s')
        else:
            raise Exception("model_type must be Bert or Canine")

        model.cuda()

        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train(model, train_dataloader, validation_dataloader, optimizer, num_epochs, batch_size, device, model_type, model_filepath, stats_filepath)


