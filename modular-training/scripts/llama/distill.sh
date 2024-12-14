STUDENT_CKPT="path/to/student.nemo"  # can also be None (will use default architecture found in examples/nlp/language_modeling/conf/megatron_llama_distill.yaml)
TEACHER_CKPT="path/to/teacher.nemo"
# TOKENIZER="path/to/tokenizer.model"
TOKENIZER=HuggingFaceTokenizer
DATA_PATHS="[1.0,path/to/tokenized/data]"
FINAL_SAVE_FILE="final_checkpoint.nemo"
TP=4

NPROC=$TP
launch_config="torchrun --nproc_per_node=$NPROC"

${launch_config} examples/nlp/language_modeling/megatron_gpt_distillation.py \
    model.restore_from_path=$STUDENT_CKPT \
    model.kd_teacher_restore_from_path=$TEACHER_CKPT \
    model.tensor_model_parallel_size=$TP \
    model.tokenizer.model=$TOKENIZER \
    model.data.data_prefix=$DATA_PATHS \
    model.nemo_path=$FINAL_SAVE_FILE \
    trainer.precision=bf16 \
    trainer.devices=$NPROC