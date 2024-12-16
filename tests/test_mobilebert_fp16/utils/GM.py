from __future__ import absolute_import, division, print_function
import sys
import time
import os
import numpy as np
import torch
import onnx
import dump_utils as dump
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from MobileBert_PyTorch.model.modeling_mobilebert import MobileBertConfig, MobileBertForSequenceClassification
from MobileBert_PyTorch.model.tokenization_mobilebert import BertTokenizer

from MobileBert_PyTorch.callback.optimization.adamw import AdamW
from MobileBert_PyTorch.callback.lr_scheduler import get_linear_schedule_with_warmup
from MobileBert_PyTorch.metrics.glue_compute_metrics import compute_metrics
from MobileBert_PyTorch.processors import glue_output_modes as output_modes
from MobileBert_PyTorch.processors import glue_processors as processors
from MobileBert_PyTorch.processors import glue_convert_examples_to_features as convert_examples_to_features
from MobileBert_PyTorch.processors import collate_fn
from MobileBert_PyTorch.tools.common import seed_everything
from MobileBert_PyTorch.tools.common import init_logger, logger
from MobileBert_PyTorch.callback.progressbar import ProgressBar
from MobileBert_PyTorch.tools.finetuning_argparse import get_argparse

# ~~~~~~~~~~ INTRO ~~~~~~~~~~
# Set the seed for reproducibility
np.random.seed(seed=1)  # <----- Sneed
torch.manual_seed(0)

# Visualize data with more precision
torch.set_printoptions(precision=10, sci_mode=False)

MODEL_CLASSES = {
    "mobilebert": (MobileBertConfig, MobileBertForSequenceClassification, BertTokenizer),
}

args = get_argparse()

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
args.device = device

args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

data_type = args.type
skipgen = args.skipgen
skipgenio = args.skipgenio
skipembed = args.skipembed

# Prepare GLUE task
args.task_name = args.task_name.lower()
if args.task_name not in processors:
    raise ValueError("Task not found: %s" % (args.task_name))
processor = processors[args.task_name]()
args.output_mode = output_modes[args.task_name]
label_list = processor.get_labels()
num_labels = len(label_list)

config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=args.task_name,
    cache_dir=args.cache_dir if args.cache_dir else None,
    num_hidden_layers=args.n_layers,
    hidden_dropout_prob = 0.0,
    attention_probs_dropout_prob=0.0,
)
tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
)
model = model_class.from_pretrained(
    args.eval_model_dir, 
    from_tf=bool(".ckpt" in args.eval_model_dir), 
    config=config, 
    cache_dir=args.cache_dir if args.cache_dir else None
)

model.bfloat16()
model.to(args.device)

# Print data and create data header file
f = open('net_args.h', "w") 

# Setup the compilation parameter for the data type
if data_type == 'float':
    f.write('// Float32 Mobilebert\n#define FLOAT32\n\n')
elif data_type == 'fp16':
    f.write('// Float16 Mobilebert\n#define FLOAT16\n\n')
else: 
    print("Invalid data type selection!!")

# Write sizes in header file
f.write('#define VOCAB_SIZE ' + str(config.vocab_size) + '\n')
f.write('\n')
f.write('#define EMBED_SIZE ' + str(config.embedding_size) + '\n')
f.write('\n')
f.write('#define HIDDEN_SIZE ' + str(config.hidden_size) + '\n')
f.write('\n')
f.write('#define INTERMEDIATE_SIZE ' + str(config.intermediate_size) + '\n')
f.write('\n')
f.write('#define NUM_HEADS ' + str(config.num_attention_heads) + '\n')
f.write('\n')
f.write('#define N_HIDDEN_LAYERS ' + str(config.num_hidden_layers) + '\n')
f.write('\n')
f.write('#define N_FFN ' + str(config.num_feedforward_networks) + '\n')
f.write('\n')
f.write('#define BOTTLENECK_SIZE ' + str(config.intra_bottleneck_size) + '\n')
f.write('\n')
f.write('#define ATTENTION_DROPOUT ' + str(config.attention_probs_dropout_prob) + '\n')
f.write('\n')
f.write('#define HIDDEN_DROPOUT ' + str(config.hidden_dropout_prob) + '\n')
f.write('\n')
f.write('#define TYPE_VOCAB_SIZE ' + str(config.type_vocab_size) + '\n')
f.write('\n')
f.write('#define SEQ_LEN ' + str(args.seq_len) + '\n')
f.write('\n')
f.write("#define INPUT_SIZE " + str(args.seq_len * config.hidden_size) + "\n")
f.write('\n')
f.write("#define OUTPUT_SIZE "+ str(args.seq_len * config.hidden_size) + "\n")
f.write('\n')
f.close()


#f = open('mobilebert_data.h', "w") 
count = 0
count_layer = 0
count_embeddings = 0
count_encoder = 0
count_classifier = 0

for name, param in model.named_parameters():
    print(name)
    print(param.size())
    print(param.nelement())
    #if("word_embeddings" in name):
    #    f.write('PI_L2 ' + data_type + ' word_embeddings[VOCAB_SIZE*EMBEDDING_SIZE] = {'+dump.tensor_to_string(param)+'};\n')
    if("layer.0" in name):
        count_layer = count_layer + param.nelement()
    if("embeddings" in name):
        count_embeddings = count_embeddings + param.nelement()
    if("classifier" in name):
        count_classifier = count_classifier + param.nelement()
    if("encoder" in name):
        count_encoder = count_encoder + param.nelement()
    #f.write(str(name))
    #f.write(str(param))
    count = count + param.nelement()
    #f.write("\n")
    #continue

print("Final count: ")
print("Count of parameters: " + str(count))
print("Size of parameters: " + str(count * 4))
print("Count of encoder layer: " + str(count_layer))
print("Count of embeddings: " + str(count_embeddings))
print("Count of encoder: " + str(count_encoder))
print("Count of classifier: " + str(count_classifier))
#f.close()

def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and 'roberta' in args.model_type:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.max_seq_length,
                                                output_mode=output_mode)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset

if not skipembed:
    # ~~~~~~~~~~ MANAGE VOCABULARY, POSITION AND TYPE EMBEDDING ~~~~~~~~~~

    for name, param in model.named_parameters():
        if("embeddings.word_embeddings.weight" in name):
            f = open("vocabulary.h", "w")
            f.write(
                "PI_L2 fp16 VOCABULARY["
                + str(config.true_hidden_size * config.vocab_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
            f.close()
        if("embeddings.position_embeddings.weight" in name):
            f = open("position_embeds.h", "w")
            f.write(
                "PI_L2 fp16 POS_EMBED["
                + str(config.hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
            f.close()
        if("embeddings.token_type_embeddings.weight" in name):
            f = open("token_type_embeds.h", "w")
            f.write(
                "PI_L2 fp16 TOKEN_TYPE_EMBED["
                + str(config.type_vocab_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
            f.close()

    
    # ~~~~~~~~~~ MANAGE EMBEDDING TRANSFORMATION ~~~~~~~~~~
    f = open("embeddings.h", "w")
    for name, param in model.named_parameters():
        if("embeddings.embedding_transformation.weight" in name):
            f.write(
                "PI_L2 fp16 EMBEDDING_WEIGHTS["
                + str(config.true_hidden_size * 3 * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("embeddings.embedding_transformation.bias" in name):
            f.write(
                "PI_L2 fp16 EMBEDDING_BIASES["
                + str(config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("embeddings.LayerNorm.bias" in name):
            f.write(
                "PI_L2 fp16 EMBEDDING_NORM_BIASES["
                + str(config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("embeddings.LayerNorm.weight" in name):
            f.write(
                "PI_L2 fp16 EMBEDDING_NORM_WEIGHTS["
                + str(config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
    f.close()
        
        

if not skipgenio:
    # ~~~~~~~~~~ MANAGE INPUT ~~~~~~~~~~
    
    # Gnereate random input id sequence
    inp = torch.from_numpy(np.short(np.random.uniform(low=0, high=config.vocab_size, size=(1, args.seq_len))))
    '''
    # Print input data to terminal
    print("------------Input IDs sequence------------")
    print(inp)

    f = open("input-sequence.h", "w")

    f.write("#define INPUT_SIZE " + str(args.seq_len * config.hidden_size) + "\n")
    f.write(
        "PI_L2 int INPUT_IDS["
        + str(args.seq_len)
        + "] = {"
        + dump.tensor_to_string(inp)
        + "};\n\n"
    )
    f.write(
        "PI_L2 fp16 INPUT["
        + str(args.seq_len * config.hidden_size)
        + "];\n\n"
    )

    f.close()
    '''
    # Generate random input data for encoder
    inp = torch.mul(torch.randn(1, args.seq_len, config.hidden_size), 0.001).bfloat16()
    inp.requires_grad = False

    # Print input data to terminal
    print("------------Input sequence------------")
    print(inp)

    # Write transpose of input data to file
    inp_copy = torch.transpose(inp, -1, -2)

    f = open("input-sequence.h", "w")

    f.write(
        "PI_L2 fp16 INPUT[INPUT_SIZE] = {" + dump.tensor_to_string(inp) + "};\n"
    )

    f.close()
    '''
    # Take a real data sequence
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    args.eval_batch_size = 1
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)
    thing = next(iter(eval_dataloader))
    inp = model.bert.embeddings(thing[0].to(args.device))

    f = open("input-sequence.h", "w")

    f.write("#define INPUT_SIZE " + str(inp.numel()) + "\n")
    f.write(
        "PI_L2 fp16 INPUT[INPUT_SIZE] = {" + dump.tensor_to_string(inp) + "};\n"
    )

    f.close()
    '''
    # ~~~~~~~~~~ MANAGE OUTPUT ~~~~~~~~~~
    # Save Encoder's output
    
    f = open("output-sequence.h", "w")

    # COMMENT THE FOLLOWING LINE IF YOU AREN'T WORKING WITH INPUT IDs
    #embed = model.bert.embeddings(inp.int().to(args.device))
    #word_embed = model.bert.embeddings.word_embeddings(inp.int().to(args.device))

    output = model.bert.encoder(inp.to(args.device), head_mask=[None])[0]
    
    f.write(
        "PI_L2 fp16 OUTPUT[OUTPUT_SIZE] = {" + dump.tensor_to_string(output) + "};\n"
    )
    '''
    f.write(
        "PI_L2 fp16 EMBEDDINGS["
        + str(args.seq_len * config.true_hidden_size)
        + "] = {" + dump.tensor_to_string(word_embed) + "};\n"
    )
    '''
    f.close()


if not skipgen:

    # ~~~~~~~~~~ ENCODER LAYER DATA ~~~~~~~~~~

    # ~~~~~~~~~~ ATTENTION ~~~~~~~~~~
    f = open("attention-defines.h", "w")

    for name, param in model.named_parameters():
        # ~~~~~~~~~~ MANAGE QKV WEIGHTS & BIASES ~~~~~~~~~~
        if("layer.0.attention.self.query.weight" in name):
            f.write(
                "PI_L2 fp16 INPUT_WEIGHTS_Q["
                + str(config.true_hidden_size * config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.attention.self.query.bias" in name):
            f.write(
                "PI_L2 fp16 INPUT_BIASES_Q["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )

        if("layer.0.attention.self.key.weight" in name):
            f.write(
                "PI_L2 fp16 INPUT_WEIGHTS_K["
                + str(config.true_hidden_size * config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.attention.self.key.bias" in name):
            f.write(
                "PI_L2 fp16 INPUT_BIASES_K["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )

        if("layer.0.attention.self.value.weight" in name):
            f.write(
                "PI_L2 fp16 INPUT_WEIGHTS_V["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.attention.self.value.bias" in name):
            f.write(
                "PI_L2 fp16 INPUT_BIASES_V["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        
        # ~~~~~~~~~~ MANAGE OUTPUT WEIGHTS & BIASES ~~~~~~~~~~
        if("layer.0.attention.output.dense.weight" in name):
            f.write(
                "PI_L2 fp16 ATTENTION_OUTPUT_WEIGHTS["
                + str(config.true_hidden_size * config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.attention.output.dense.bias" in name):
            f.write(
                "PI_L2 fp16 ATTENTION_OUTPUT_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        
        if("layer.0.attention.output.LayerNorm.weight" in name):
            f.write(
                "PI_L2 fp16 ATTENTION_OUTPUT_NORM_WEIGHTS["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.attention.output.LayerNorm.bias" in name):
            f.write(
                "PI_L2 fp16 ATTENTION_OUTPUT_NORM_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )

    f.close()

    # ~~~~~~~~~~ INTERMEDIATE ~~~~~~~~~~
    f = open("intermediate-defines.h", "w")

    for name, param in model.named_parameters():
        if("layer.0.intermediate.dense.weight" in name):
            f.write(
                "PI_L2 fp16 INTERMEDIATE_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.intermediate.dense.bias" in name):
            f.write(
                "PI_L2 fp16 INTERMEDIATE_BIASES["
                + str(config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )

    f.close()

    # ~~~~~~~~~~ OUTPUT ~~~~~~~~~~
    f = open("output-defines.h", "w")

    for name, param in model.named_parameters():
        if("layer.0.output.dense.weight" in name):
            f.write(
                "PI_L2 fp16 OUTPUT_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.output.dense.bias" in name):
            f.write(
                "PI_L2 fp16 OUTPUT_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        
        if("layer.0.output.LayerNorm.weight" in name):
            f.write(
                "PI_L2 fp16 OUTPUT_NORM_WEIGHTS["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.output.LayerNorm.bias" in name):
            f.write(
                "PI_L2 fp16 OUTPUT_NORM_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )

        if("layer.0.output.bottleneck.dense.weight" in name):
            f.write(
                "PI_L2 fp16 OUTPUT_BOTTLENECK_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.output.bottleneck.dense.bias" in name):
            f.write(
                "PI_L2 fp16 OUTPUT_BOTTLENECK_BIASES["
                + str(config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        
        if("layer.0.output.bottleneck.LayerNorm.weight" in name):
            f.write(
                "PI_L2 fp16 OUTPUT_BOTTLENECK_NORM_WEIGHTS["
                + str(config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.output.bottleneck.LayerNorm.bias" in name):
            f.write(
                "PI_L2 fp16 OUTPUT_BOTTLENECK_NORM_BIASES["
                + str(config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        
    f.close()
        
    # ~~~~~~~~~~ BOTTLENECK ~~~~~~~~~~
    f = open("bottleneck-defines.h", "w")

    for name, param in model.named_parameters():
        # ~~~~~~~~~~ INPUT PART ~~~~~~~~~~
        if("layer.0.bottleneck.input.dense.weight" in name):
            f.write(
                "PI_L2 fp16 BOTTLENECK_INPUT_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.bottleneck.input.dense.bias" in name):
            f.write(
                "PI_L2 fp16 BOTTLENECK_INPUT_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.bfloat16())
                + "};\n\n"
            )
        if("layer.0.bottleneck.input.LayerNorm.weight" in name):
            f.write(
                "PI_L2 fp16 BOTTLENECK_INPUT_NORM_WEIGHTS["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.bottleneck.input.LayerNorm.bias" in name):
            f.write(
                "PI_L2 fp16 BOTTLENECK_INPUT_NORM_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )

        # ~~~~~~~~~~ ATTENTION PART ~~~~~~~~~~
        if("layer.0.bottleneck.attention.dense.weight" in name):
            f.write(
                "PI_L2 fp16 BOTTLENECK_ATTENTION_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.bottleneck.attention.dense.bias" in name):
            f.write(
                "PI_L2 fp16 BOTTLENECK_ATTENTION_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.bottleneck.attention.LayerNorm.weight" in name):
            f.write(
                "PI_L2 fp16 BOTTLENECK_ATTENTION_NORM_WEIGHTS["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.bottleneck.attention.LayerNorm.bias" in name):
            f.write(
                "PI_L2 fp16 BOTTLENECK_ATTENTION_NORM_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )

    f.close()

    # ~~~~~~~~~~ FFN ~~~~~~~~~~
    f = open("ffn-defines.h", "w")

    for name, param in model.named_parameters():
        # ~~~~~~~~~~ FFN 0 ~~~~~~~~~~
        if("layer.0.ffn.0.intermediate.dense.weight" in name):
            f.write(
                "PI_L2 fp16 FFN0_INTERMEDIATE_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.ffn.0.intermediate.dense.bias" in name):
            f.write(
                "PI_L2 fp16 FFN0_INTERMEDIATE_BIASES["
                + str(config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.ffn.0.output.dense.weight" in name):
            f.write(
                "PI_L2 fp16 FFN0_OUTPUT_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.ffn.0.output.dense.bias" in name):
            f.write(
                "PI_L2 fp16 FFN0_OUTPUT_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.ffn.0.output.LayerNorm.weight" in name):
            f.write(
                "PI_L2 fp16 FFN0_OUTPUT_NORM_WEIGHTS["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.ffn.0.output.LayerNorm.bias" in name):
            f.write(
                "PI_L2 fp16 FFN0_OUTPUT_NORM_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        
        # ~~~~~~~~~~ FFN 1 ~~~~~~~~~~
        if("layer.0.ffn.1.intermediate.dense.weight" in name):
            f.write(
                "PI_L2 fp16 FFN1_INTERMEDIATE_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.ffn.1.intermediate.dense.bias" in name):
            f.write(
                "PI_L2 fp16 FFN1_INTERMEDIATE_BIASES["
                + str(config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.ffn.1.output.dense.weight" in name):
            f.write(
                "PI_L2 fp16 FFN1_OUTPUT_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.ffn.1.output.dense.bias" in name):
            f.write(
                "PI_L2 fp16 FFN1_OUTPUT_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.ffn.1.output.LayerNorm.weight" in name):
            f.write(
                "PI_L2 fp16 FFN1_OUTPUT_NORM_WEIGHTS["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.ffn.1.output.LayerNorm.bias" in name):
            f.write(
                "PI_L2 fp16 FFN1_OUTPUT_NORM_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        
        # ~~~~~~~~~~ FFN 2 ~~~~~~~~~~
        if("layer.0.ffn.2.intermediate.dense.weight" in name):
            f.write(
                "PI_L2 fp16 FFN2_INTERMEDIATE_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.ffn.2.intermediate.dense.bias" in name):
            f.write(
                "PI_L2 fp16 FFN2_INTERMEDIATE_BIASES["
                + str(config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.ffn.2.output.dense.weight" in name):
            f.write(
                "PI_L2 fp16 FFN2_OUTPUT_WEIGHTS["
                + str(config.true_hidden_size * config.hidden_size)
                + "] = {"
                + dump.tensor_to_string(param.transpose(0, 1))
                + "};\n\n"
            )
        if("layer.0.ffn.2.output.dense.bias" in name):
            f.write(
                "PI_L2 fp16 FFN2_OUTPUT_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.ffn.2.output.LayerNorm.weight" in name):
            f.write(
                "PI_L2 fp16 FFN2_OUTPUT_NORM_WEIGHTS["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )
        if("layer.0.ffn.2.output.LayerNorm.bias" in name):
            f.write(
                "PI_L2 fp16 FFN2_OUTPUT_NORM_BIASES["
                + str(config.true_hidden_size)
                + "] = {"
                + dump.tensor_to_string(param)
                + "};\n\n"
            )

    f.close()





