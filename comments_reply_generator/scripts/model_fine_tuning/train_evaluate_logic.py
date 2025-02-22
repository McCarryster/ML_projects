from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from datetime import datetime
import torch
import warnings
from config import *

warnings.filterwarnings("ignore", category=UserWarning)
rouge = Rouge()

def evaluate(model, tokenizer, writer, epoch, global_step, dataloader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    generated_texts = []
    actual_texts = []
    val_start_time = datetime.now()
    print('#'*100)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(train_args['device'])
            attention_mask = batch['attention_mask'].to(train_args['device'])
            labels = batch['labels'].to(train_args['device'])

            # Get model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Generate output
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # max_length=evaluation_args['max_length'],
                max_new_tokens=evaluation_args['max_new_tokens'],
                num_beams=evaluation_args['num_beams'],
                no_repeat_ngram_size=evaluation_args['no_repeat_ngram_size'],
                temperature=evaluation_args['temperature'],
                top_p=evaluation_args['top_p'],
                top_k=evaluation_args['top_k']
            )

            # Convert generated IDs to text
            generated_text = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            actual_text = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
            generated_texts.extend(generated_text)
            actual_texts.extend(actual_text)

            # Calculate accuracy
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=-1)
            total_correct += (predicted == labels).sum().item()
            total_tokens += attention_mask.sum().item()

            # Calculate perplexity
            batch_perplexity = torch.exp(torch.tensor(loss.item())).item()

            if (batch_idx+1) % train_args['logging_steps'] == 0:
                print(f"VALIDATION: Epoch [{epoch+1}/{train_args['num_epochs']}], "
                    f"Batch [{batch_idx+1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Accuracy: {(predicted == labels).sum().item() / attention_mask.sum().item():.4f}, "
                    f"Perplexity: {batch_perplexity:.4f}, "
                    f"Curr VAL time [{batch_idx+1}/{len(dataloader)}]: {datetime.now() - val_start_time}")
                writer.add_scalar('Val/loss', loss.item(), epoch * len(dataloader) + batch_idx)
                writer.add_scalar("Val/accuracy", (predicted == labels).sum().item() / attention_mask.sum().item(), epoch * len(dataloader) + batch_idx)
                writer.add_scalar("Val/perplexity", batch_perplexity, epoch * len(dataloader) + batch_idx)
                writer.add_text("Generated Text Progress", generated_text[0], epoch * len(dataloader) + batch_idx)

        # Calculate average loss, accuracy, perplexity
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_correct / total_tokens
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Calculate BLEU score
        bleu_scores = []
        for gen_com, act_com in zip(generated_texts, actual_texts):
            bleu_scores.append(sentence_bleu(gen_com, act_com))
        avg_bleu = sum(bleu_scores) / len(bleu_scores)

        # Calculate ROUGE score
        rouge_scores = {
            'rouge-1': {'r': 0, 'p': 0, 'f': 0},
            'rouge-2': {'r': 0, 'p': 0, 'f': 0},
            'rouge-l': {'r': 0, 'p': 0, 'f': 0}
        }
        for get_text, ref_text in zip(generated_texts, actual_texts):
            scores = rouge.get_scores(get_text, ref_text)
            for rouge_type in rouge_scores:
                for metric in ['r', 'p', 'f']:
                    rouge_scores[rouge_type][metric] += scores[0][rouge_type][metric]
        num_texts = len(generated_texts)
        avg_rouge = {
            rouge_type: {metric: score / num_texts for metric, score in scores.items()}
            for rouge_type, scores in rouge_scores.items()
        }
        
        # Calculate METEOR score
        meteor = 0
        for gen_com, act_com in zip(generated_texts, actual_texts):
            reference = act_com.split()
            candidate = gen_com.split()
            meteor += meteor_score([reference], candidate)
        avg_meteor = meteor / len(generated_texts)

        print(f"DONE VAL: Epoch [{epoch+1}/{len(dataloader)}], "
              f"avg_loss: {avg_loss:.4f}, "
              f"avg_accuracy: {avg_accuracy:.4f}, "
              f"avg_perplexity: {avg_perplexity:.4f}, "
              f"avg_bleu: {avg_bleu:.4f}, "
              f"avg_rouge: {avg_rouge}, "
              f"avg_meteor: {avg_meteor:.4f}")
        writer.add_scalar("Val/avg_loss", avg_loss, epoch)
        writer.add_scalar("Val/avg_accuracy", avg_accuracy, epoch)
        writer.add_scalar("Val/avg_perplexity", avg_perplexity, epoch)
        writer.add_scalar("Val/avg_bleu", avg_bleu, epoch)
        writer.add_scalar("Val/avg_rouge_1_r", avg_rouge['rouge-1']['r'], epoch)
        writer.add_scalar("Val/avg_rouge_1_p", avg_rouge['rouge-1']['p'], epoch)
        writer.add_scalar("Val/avg_rouge_1_f", avg_rouge['rouge-1']['f'], epoch)
        writer.add_scalar("Val/avg_rouge_2_r", avg_rouge['rouge-2']['r'], epoch)
        writer.add_scalar("Val/avg_rouge_2_p", avg_rouge['rouge-2']['p'], epoch)
        writer.add_scalar("Val/avg_rouge_2_f", avg_rouge['rouge-2']['f'], epoch)
        writer.add_scalar("Val/avg_rouge_l_r", avg_rouge['rouge-l']['r'], epoch)
        writer.add_scalar("Val/avg_rouge_l_p", avg_rouge['rouge-l']['p'], epoch)
        writer.add_scalar("Val/avg_rouge_l_f", avg_rouge['rouge-l']['f'], epoch)
        writer.add_scalar("Val/avg_meteor", avg_meteor, epoch)
        print(f"VAL time: {datetime.now() - val_start_time}")
        print('#'*100)
        return avg_loss

def train(model, tokenizer, optimizer, scaler, scheduler, writer, train_dataloader, val_dataloader, checkpoints_path, model_save_dir):
    model.train()
    best_val_loss = float('inf')
    early_stopping_counter = 0
    global_step = 0
    for epoch in range(train_args['num_epochs']):
        epoch_start_time = datetime.now()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(train_args['device'])
            attention_mask = batch['attention_mask'].to(train_args['device'])
            labels = batch['labels'].to(train_args['device'])

            # Forward pass using mixed precision
            if train_args['mixed_precision']:
                with torch.amp.autocast(train_args['device']):
                # with torch.amp.autocast(train_args['device'], dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                    # Backward pass and optimize with GradScaler
                    optimizer.zero_grad()
                    loss.backward()

                    # Unscale the gradients before clipping and stepping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_args['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.zero_grad(optimizer)
                    scheduler.step()
            else:
                # Forward pass without mixed precision
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Calculate total loss and tokens
            total_loss += loss.item()
            total_tokens += attention_mask.sum().item()

            # Calculate accuracy
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=-1)
            total_correct += (predicted == labels).sum().item()

            # Calculate perplexity
            batch_perplexity = torch.exp(torch.tensor(loss.item())).item()
            
            # Increment global step for saving model
            global_step += 1

            # Print metrics and write them to tensorboard after every n batch iteration
            if (batch_idx+1) % train_args['logging_steps'] == 0:
                print(f"TRAINING: Epoch [{epoch+1}/{train_args['num_epochs']}], "
                      f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Accuracy: {(predicted == labels).sum().item() / attention_mask.sum().item():.4f}, "
                      f"Perplexity: {batch_perplexity:.4f}, "
                      f"Time: {datetime.now() - epoch_start_time}")
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_dataloader) + batch_idx)
                writer.add_scalar("Train/Accuracy", (predicted == labels).sum().item() / attention_mask.sum().item(), epoch * len(train_dataloader) + batch_idx)
                writer.add_scalar("Train/Perplexity", batch_perplexity, epoch * len(train_dataloader) + batch_idx)

            # Save model every n steps
            if global_step % train_args['save_steps'] == 0:
                # os.makedirs(checkpoints_path, exist_ok=True) # Create checkpoints folder
                # checkpoint_dir = os.path.join(checkpoints_path, f'checkpoint_{global_step}') # Path for saving checkpoint
                # os.makedirs(checkpoint_dir, exist_ok=True)   # Create single checkpoint folder
                # torch.save(model.state_dict(), f"{checkpoint_dir}/model_state_dict.pth")
                # torch.save(optimizer.state_dict(), f"{checkpoint_dir}/optimizer_state_dict.pth")
                # torch.save(scheduler.state_dict(), f"{checkpoint_dir}/scheduler_state_dict.pth")
                # torch.save({
                #     'global_step': global_step,
                #     'epoch': epoch,
                #     'best_val_loss': best_val_loss,
                #     'early_stopping_counter': early_stopping_counter,
                # }, f"{checkpoint_dir}/metadata.pth")
                print(f"CHECKPOINT: Model saved at Epoch [{epoch+1}/{train_args['num_epochs']}], Batch [{batch_idx+1}/{len(train_dataloader)}], Step {global_step}")

            if global_step % train_args['validation_steps'] == 0:
                avg_val_loss = evaluate(model, tokenizer, writer, epoch, global_step, val_dataloader)

            if train_args['early_stopping']:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                if early_stopping_counter >= train_args['early_stopping_patience']:
                    print(f"Early stopping at Epoch [{epoch+1}/{train_args['num_epochs']}], Batch [{batch_idx}/{len(train_dataloader)}], Step {global_step}")
                    break

        # Calculate average loss, accuracy and perplexity
        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_correct / total_tokens
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Print metrics and write them to tensorboard
        print(f"EPOCH DONE {epoch+1}/{train_args['num_epochs']}: "
              f"Loss: {avg_loss:.4f}, "
              f"Accuracy: {avg_accuracy:.4f}, "
              f"Perplexity: {avg_perplexity:.4f}, "
              f"Per Epoch time: {datetime.now() - epoch_start_time}")
        writer.add_scalar("Train/avg_loss", avg_loss, epoch)
        writer.add_scalar("Train/avg_accuracy", avg_accuracy, epoch)
        writer.add_scalar("Train/avg_perplexity", avg_perplexity, epoch)
    writer.close()

    # model.save_pretrained(model_save_dir)
    # tokenizer.save_pretrained(model_save_dir)