from datetime import datetime
import random
from torch.utils.data import Subset
from itertools import islice
import torch
from config import *
import os
from logger_setup import logger, log_separator
from huggingface_hub import HfApi, ModelCard, ModelCardData, create_repo


def hf_push_checkpoint(dir, path_in_repo, commit_message):
    # Create repo if it doesn't exist
    create_repo(stuff_and_paths['hf_repo_id'], exist_ok=True)
    api = HfApi()
    try:
        # Push to HF Hub
        api.upload_folder(
            folder_path=dir,
            path_in_repo=path_in_repo,  # Creates folder in repo
            repo_id=stuff_and_paths['hf_repo_id'],
            commit_message=commit_message,
        )
    except Exception as e:
        print(f"Failed to upload checkpoint: {str(e)}")
        logger.info(f"Failed to upload checkpoint: {str(e)}")


def get_random_subset(dataloader, subset_size):
    dataset_size = len(dataloader.dataset)
    indices = list(range(dataset_size))
    random_indices = random.sample(indices, subset_size)
    subset = Subset(dataloader.dataset, random_indices)
    subset_dataloader = torch.utils.data.DataLoader(subset, batch_size=dataloader.batch_size)
    return subset_dataloader


# 7. Evalation loop
def evaluate(model, tokenizer, writer, epoch, val_dataloader):
    model.eval()
    total_loss = 0
    val_start_time = datetime.now()

    print('#'*100)
    log_separator()

    if train_args['full_validation']:
        eval_dataloader = val_dataloader
    else:
        eval_dataloader = get_random_subset(val_dataloader, train_args['val_set_size'])

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            input_ids = batch['input_ids'].to(train_args['device'])
            attention_mask = batch['attention_mask'].to(train_args['device'])
            labels = batch['labels'].to(train_args['device'])

            # Get model outputs and calculate loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Accumulate loss and perplexity
            total_loss += loss.item()

            # Test generation, print metrics and write them to tensorboard after every n batch iteration
            if (batch_idx+1) % train_args['logging_steps'] == 0:
                avg_loss = total_loss / (batch_idx+1)
                avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

                print(f"VALIDATION: Epoch [{epoch+1}/{train_args['num_epochs']}], "
                    f"Batch [{batch_idx+1}/{len(eval_dataloader)}], "
                    f"Loss: {avg_loss:.4f}, "
                    f"Perplexity: {avg_perplexity:.4f}, "
                    f"Curr VAL time: {datetime.now() - val_start_time}")
                logger.info(f"VALIDATION: Epoch [{epoch+1}/{train_args['num_epochs']}], "
                            f"Batch [{batch_idx+1}/{len(eval_dataloader)}], "
                            f"Loss: {avg_loss:.4f}, "
                            f"Perplexity: {avg_perplexity:.4f}, "
                            f"Curr VAL time: {datetime.now() - val_start_time}")

                # Process first example in batch using tensor operations
                example_labels = labels[0]
                example_input_ids = input_ids[0]
                example_attention_mask = attention_mask[0]

                # Find start of target sequence using labels tensor
                non_pad_mask = (example_labels != -100)
                if non_pad_mask.any():
                    start_of_target = non_pad_mask.nonzero(as_tuple=True)[0].min().item()
                else:
                    start_of_target = 0  # Fallback if all labels are -100

                # Create actual input by truncating before target sequence
                actual_input_ids = example_input_ids[:start_of_target].unsqueeze(0)
                actual_attention_mask = example_attention_mask[:start_of_target].unsqueeze(0)


                generated_ids = model.generate(
                    input_ids=actual_input_ids,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=actual_attention_mask,
                    max_new_tokens=generation_args['max_new_tokens'],
                    num_beams=generation_args['num_beams'],
                    temperature=generation_args['temperature'],
                    repetition_penalty=generation_args['repetition_penalty'],
                    no_repeat_ngram_size=generation_args['no_repeat_ngram_size'],
                    early_stopping=generation_args['early_stopping'],
                    length_penalty=generation_args['length_penalty'],
                    do_sample=generation_args['do_sample'],
                    top_k=generation_args['top_k'],
                    top_p=generation_args['top_p'],
                )
                
                # Decode results
                input_len = actual_input_ids.shape[1]
                generated_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                decoded_input = tokenizer.decode(actual_input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                
                print(f"[INPUT]: {decoded_input}")
                print(f'[GENERATED COMMENT]: {generated_text}')
                logger.info(f"[INPUT]: {decoded_input}")
                logger.info(f'[GENERATED COMMENT]: {generated_text}')

                writer.add_scalar('Val/loss', avg_loss, epoch * len(eval_dataloader) + batch_idx)
                writer.add_scalar("Val/perplexity", avg_perplexity, epoch * len(eval_dataloader) + batch_idx)
                writer.add_text("Generated Text Progress", generated_text, epoch * len(eval_dataloader) + batch_idx)

                print('#'*100)
                log_separator()

    # Calculate Perplexity and avg_loss
    avg_loss = total_loss / len(eval_dataloader)
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # Print results
    print(f"*** DONE VAL: Epoch [{epoch+1}/{train_args['num_epochs']}], "
            f"avg_loss: {avg_loss:.4f}, "
            f"avg_perplexity: {avg_perplexity:.4f}, "
            f"Validation took: {datetime.now() - val_start_time}")
    logger.info(f"DONE VAL: Epoch [{epoch+1}/{train_args['num_epochs']}], "
                f"avg_loss: {avg_loss:.4f}, "
                f"avg_perplexity: {avg_perplexity:.4f}, "
                f"Validation took: {datetime.now() - val_start_time}")
    writer.add_scalar("Val/avg_loss", avg_loss, epoch)
    writer.add_scalar("Val/avg_perplexity", avg_perplexity, epoch)

    model.train()

    log_separator()
    print('#'*100)
    
    return avg_loss


def perform_optimization_step(model, loss, optimizer, scheduler, scaler, accumulation_steps, batch_idx):
    if train_args['mixed_precision']:
        # Mixed Precision Only
        if not train_args['gradient_accumulation']:
            scaled_loss = scaler.scale(loss)
            optimizer.zero_grad()
            scaled_loss.backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_args['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        # Mixed Precision + Gradient Accumulation
        else:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_args['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
    # Gradient Accumulation Only
    elif train_args['gradient_accumulation']:
        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_args['gradient_clip'])
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    # Regular Training
    else:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_args['gradient_clip'])
        optimizer.step()
        scheduler.step()


def train(model, tokenizer, optimizer, scaler, scheduler, writer, train_dataloader, val_dataloader, checkpoints_path, model_save_dir, checkpoint_data=None):
    if train_args['grad_checkpoint']:
        model.enable_input_require_grads()    # Should be included when using LoRA
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(train_args['device'])
    model.train()

    start_epoch = 0
    global_step = 0
    best_val_loss = None
    early_stopping_counter = 0
    avg_val_loss = None
    early_stop = False
    completed_accumulation_steps = 0

    if train_args['resume_from_checkpoint']:
        start_epoch = checkpoint_data['epoch']
        global_step = checkpoint_data['global_step']
        best_val_loss = checkpoint_data['best_val_loss']
        early_stopping_counter = checkpoint_data['early_stopping_counter']
        completed_accumulation_steps = checkpoint_data['completed_accumulation_steps']

    for epoch in range(start_epoch, train_args['num_epochs']):

        epoch_start_time = datetime.now()
        total_loss = 0
        batches_processed = 0

        # Skip from train_dataloader batches that the model has seen before
        if epoch == start_epoch and train_args['resume_from_checkpoint']:
            batches_processed = checkpoint_data['batches_processed']
            dataloader = islice(train_dataloader, batches_processed, None)
            start_batch_idx = batches_processed
            total_loss = checkpoint_data['total_loss']
        else:
            dataloader = train_dataloader
            start_batch_idx = 0


        for batch_idx, batch in enumerate(dataloader, start=start_batch_idx):
            input_ids = batch['input_ids'].to(train_args['device'])
            attention_mask = batch['attention_mask'].to(train_args['device'])
            labels = batch['labels'].to(train_args['device'])

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if train_args['gradient_accumulation']:
                loss = loss / train_args['accumulation_steps']
            total_loss += loss.item()

            perform_optimization_step(
                model=model,
                loss=loss,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                accumulation_steps=train_args['accumulation_steps'],
                batch_idx=batch_idx)

            if (batch_idx + 1) % train_args['accumulation_steps'] == 0:
                completed_accumulation_steps += 1
            batches_processed += 1
            
            global_step += 1

            # Print metrics and write them to tensorboard after every n batches
            if (batch_idx+1) % train_args['logging_steps'] == 0:
                if train_args['gradient_accumulation']:
                    avg_loss = total_loss / completed_accumulation_steps
                else:
                    avg_loss = total_loss / (batch_idx + 1)
                avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

                print(f"TRAINING: Epoch [{epoch+1}/{train_args['num_epochs']}], "
                      f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                      f"Loss: {avg_loss:.4f}, "
                      f"Perplexity: {avg_perplexity:.4f}, "
                      f"Time: {datetime.now() - epoch_start_time}")
                logger.info(f"TRAINING: Epoch [{epoch+1}/{train_args['num_epochs']}], "
                            f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                            f"Loss: {avg_loss:.4f}, "
                            f"Perplexity: {avg_perplexity:.4f}, "
                            f"Time: {datetime.now() - epoch_start_time}")
                writer.add_scalar('Train/Loss', avg_loss, epoch * len(train_dataloader) + batch_idx)
                writer.add_scalar("Train/Perplexity", avg_perplexity, epoch * len(train_dataloader) + batch_idx)

            # 8. Model saving. Save locally OR save locally and to HF. Save model every n steps
            if global_step % train_args['save_steps'] == 0:
                os.makedirs(checkpoints_path, exist_ok=True)
                checkpoint_dir = os.path.join(checkpoints_path, f'checkpoint_{global_step}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model with proper base model ID for LoRA
                if train_args['make_lora']:
                    model.base_model.config._name_or_path = stuff_and_paths['base_model_hf_id']
                    model.peft_config['default'].base_model_name_or_path = stuff_and_paths['base_model_hf_id']
                    model.save_pretrained(checkpoint_dir)
                else:
                    model.config._name_or_path = stuff_and_paths['base_model_hf_id']
                    model.save_pretrained(checkpoint_dir)
                
                tokenizer.save_pretrained(checkpoint_dir)
                
                # Save training states
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'total_loss': total_loss,
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'early_stopping_counter': early_stopping_counter,
                    'completed_accumulation_steps': completed_accumulation_steps,
                    'batches_processed': batches_processed,
                }
                if scaler is not None:
                    checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                torch.save(checkpoint_data, f"{checkpoint_dir}/checkpoint_step_{global_step}.pth")

                # Push to HF
                if stuff_and_paths['saving_to_hf']:
                    hf_push_checkpoint(checkpoint_dir, f"checkpoints/checkpoint_{global_step}", f"Checkpoint {global_step}")

                print(f"CHECKPOINT: Model saved at Epoch [{epoch+1}/{train_args['num_epochs']}], Batch [{batch_idx+1}/{len(train_dataloader)}], Step {global_step}")
                logger.info(f"CHECKPOINT: Model saved at Epoch [{epoch+1}/{train_args['num_epochs']}], Batch [{batch_idx+1}/{len(train_dataloader)}], Step {global_step}")

            # Validate model every n steps
            if global_step % train_args['validation_steps'] == 0:
                avg_val_loss = evaluate(model, tokenizer, writer, epoch, val_dataloader)
                
                if avg_val_loss is not None:
                    if best_val_loss is None or avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

            # Stop training after n early_stopping_patience steps
            if train_args['early_stopping'] and avg_val_loss is not None and early_stopping_counter >= train_args['early_stopping_patience']:
                print(f"Early stopping triggered!")
                early_stop = True
                break


        # Calculate average loss and perplexity
        if train_args['gradient_accumulation']:
            avg_loss = total_loss / completed_accumulation_steps
        else:
            avg_loss = total_loss / batches_processed
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Print metrics and write them to tensorboard
        print(f"EPOCH DONE {epoch+1}/{train_args['num_epochs']}: "
              f"Loss: {avg_loss:.4f}, "
              f"Perplexity: {avg_perplexity:.4f}, "
              f"Epoch time: {datetime.now() - epoch_start_time}")
        logger.info(f"EPOCH DONE {epoch+1}/{train_args['num_epochs']}: "
                    f"Loss: {avg_loss:.4f}, "
                    f"Perplexity: {avg_perplexity:.4f}, "
                    f"Epoch time: {datetime.now() - epoch_start_time}")
        writer.add_scalar("Train/avg_loss", avg_loss, epoch)
        writer.add_scalar("Train/avg_perplexity", avg_perplexity, epoch)

        if early_stop:
            break

    writer.close()
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 8. Model saving. Save locally OR save locally and to HF
    tokenizer.save_pretrained(model_save_dir)
    if train_args['make_lora']:
        model.base_model.config._name_or_path = stuff_and_paths['base_model_hf_id']
        model.peft_config['default'].base_model_name_or_path = stuff_and_paths['base_model_hf_id']
        merged_model = model.merge_and_unload()
        merged_model.config._name_or_path = stuff_and_paths['base_model_hf_id']
        merged_model.save_pretrained(model_save_dir)
        if stuff_and_paths['saving_to_hf']:
            hf_push_checkpoint(model_save_dir, ".", "Final model")
    else:
        model.config._name_or_path = stuff_and_paths['base_model_hf_id']
        model.save_pretrained(model_save_dir)
        if stuff_and_paths['saving_to_hf']:
            hf_push_checkpoint(model_save_dir, ".", "Final model")