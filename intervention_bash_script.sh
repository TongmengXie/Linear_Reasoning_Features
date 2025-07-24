# unsteered
python features_intervention.py --model_name Llama-3.2-1B --extracting_from mmlu-pro_600 --device "cuda:1"> ../../outputs/Llama-3.2-1B_test_on_MMLU600.log

python features_intervention.py --model_name Llama-3.2-1B-Instruct --extracting_from  mmlu-pro_600 --device "cuda:1"> ../../outputs/Llama-3.2-1B-Instruct_test_on_MMLU600.log 

python features_intervention.py --model_name llama_3.2_1b_instruct_rlhf --extracting_from  mmlu-pro_600 --device "cuda:1" > ../../outputs/llama_3.2_1b_instruct_rlhf_test_on_MMLU600.log 

# steered
python features_intervention.py --model_name Llama-3.2-1B --extracting_from mmlu-pro_600 --device "cuda:1" --Intervention True > ../../outputs/intervened_Llama-3.2-1B_test_on_MMLU600.log

python features_intervention.py --model_name Llama-3.2-1B-Instruct --extracting_from mmlu-pro_600 --device "cuda:1" --Intervention True > ../../outputs/intervened_Llama-3.2-1B-Instruct_test_on_MMLU600.log


# CoT
python features_intervention.py --model_name Llama-3.2-1B-Instruct --extracting_from mmlu-pro_600 --device "cuda:2" --Intervention True > ../../outputs/Llama-3.2-1B-Instruct_test_on_MMLU600.log

python features_intervention.py --model_name DeepSeek-R1-Distill-Qwen-1.5B --extracting_from mmlu-pro_600 --device "cuda:1" --Intervention True > ../../outputs/DeepSeek-R1-Distill-Qwen-1.5B_test_on_MMLU600.log
