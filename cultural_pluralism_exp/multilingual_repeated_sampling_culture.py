import os
import json
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port

import torch.nn.functional as F
import torch

from types import MethodType

from datasets import load_dataset

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name or path",
    )
    parser.add_argument("--language", type=str)
    parser.add_argument("--flag", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--data_name", type=str, choices=['polymath-en', 'amc', 'blend', 'wvs'])
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=8, help="Data parallel size")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--node-size", type=int, default=1, help="Total number of nodes"
    )
    parser.add_argument(
        "--node-rank", type=int, default=0, help="Rank of the current node"
    )
    parser.add_argument(
        "--master-addr", type=str, default="", help="Master node IP address"
    )
    parser.add_argument("--master-port", type=int, default=0, help="Master node port")
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Enforce eager mode execution."
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code."
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=64,
        help=("Maximum number of sequences to be processed in a single iteration."),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help=("Fraction of GPU memory vLLM is allowed to allocate (0.0, 1.0]."),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192
    )
    parser.add_argument(
        "--baseline", 
        type=str,
        default="none",
        choices=["none", "request_diversity", "multilingual_prompting"],
    )
    return parser.parse_args()

path = "prompt_dict_qwen3_final_novelty_expanded.json"
with open(path, 'r') as f:
    lang_prompt_final_dict = json.load(f)

instruction_path = "prompt_dict_qwen3_final.json"
with open(instruction_path, "r") as f:
    instruction_dict = json.load(f)

baseline_instruction = {
    "none": "",
    "request_diversity": "Please try to provide a novel answer.",
    "multilingual_prompting": "",
}

def main(
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    GPUs_per_dp_rank,
    enforce_eager,
    trust_remote_code,
    max_num_seqs,
    gpu_memory_utilization,
):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # CUDA_VISIBLE_DEVICES for each DP rank is set automatically inside the
    # engine processes.
    # Load and prepare prompts with IDs and labels
    language = args.language
    from datasets import load_from_disk
    if args.baseline == "multilingual_prompting":
        assert language != "en"
        if args.data_name == "blend":
            path = f"../data/blend_{language}.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            questions = [item["question"] for item in data]
        elif args.data_name == "wvs":
            path = f"../data/wvs_{language}.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            questions = [item["question"] for item in data]
        
        language_system_prompt = lang_prompt_final_dict['en']['system_prompt']
        language_instruction = instruction_dict['en']['instruction']
        language_prefix = lang_prompt_final_dict['en']['prefix']
    else:
        if args.data_name == "blend":
            path = "../data/blend.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            questions = [item["question"] for item in data]
        elif args.data_name == "wvs":
            path = "../data/wvs.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            questions = [item["question"] for item in data]
    
        language_system_prompt = lang_prompt_final_dict[language]['system_prompt']
        language_instruction = instruction_dict[language]['instruction']
        language_prefix = lang_prompt_final_dict[language]['prefix']

    # Attach prompt template and preserve idx and label
    all_prompts = []
    for idx, q in enumerate(questions):
        item = {"idx": idx, "prompt": f"{language_instruction}\n\nQuestion: {q}\n{baseline_instruction[args.baseline]}"}
        all_prompts.append(item)
    # Prepare for DP distribution
    print(f"Total length of questions: {len(all_prompts)}")
    # --- Create the LLM first (moved earlier) ---
    llm = LLM(
        model=model,
        tensor_parallel_size=GPUs_per_dp_rank,
        enforce_eager=enforce_eager,
        enable_expert_parallel=False,
        trust_remote_code=trust_remote_code,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=16384
    )

    tokenizer = llm.get_tokenizer()

    floor = len(all_prompts) // dp_size
    remainder = len(all_prompts) % dp_size

    # Distribute prompts into even groups.
    def start(rank):
        return rank * floor + min(rank, remainder)

    shard = all_prompts[start(global_dp_rank):start(global_dp_rank + 1)]
    if not shard:
        shard = [{"idx": -1, "prompt": "Placeholder", "label": None}]

    # prompts = [x["prompt"] for x in shard]
    ids = [x["idx"] for x in shard]
    prompts = []
    for x in shard:
        if x["idx"] == -1:
            prompts.append("Placeholder")
        else:
            # Rebuild from the question in your original data row:
            # You stored `item["prompt"]` earlier; safer is to read from question again.
            # If you don't have the raw question now, you can reuse x["prompt"] as the user message.
            # Here I reuse x["prompt"] as the *user content* to keep your current structure.
            msgs = [{"role": "system", "content": language_system_prompt}, {"role": "user", "content": x["prompt"]}]
            p1 = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
            p1 += f"<think>\n{language_prefix}"
            prompts.append(p1)

    sampling_params = SamplingParams(
            temperature=args.temperature, max_tokens=args.max_tokens, n=args.sample_num, seed=42, top_p=0.95, top_k=20,
        )
    results = []
    outputs = llm.generate(prompts, sampling_params)
    for idx_val, output in zip(ids, outputs):
        if idx_val == -1:
            continue
        generations = [o.text for o in output.outputs]
        results.append({
            "idx": idx_val,
            "question": questions[idx_val],
            "prompt": output.prompt,
            "responses": generations
        })
    # Save to file per DP rank
    output_dir = f"{args.data_name}_output"
    os.makedirs(output_dir, exist_ok=True)
    if args.baseline == "none":
        output_path = f"{output_dir}/{model.split('/')[-1]}_{language}_{args.temperature}_{args.sample_num}_{args.flag}_dp_rank_{global_dp_rank}.jsonl"
    else:
        output_path = f"{output_dir}/{model.split('/')[-1]}_{args.baseline}_{language}_{args.temperature}_{args.sample_num}_{args.flag}_dp_rank_{global_dp_rank}.jsonl"
    with open(output_path, "w") as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


    # 🛑 Synchronization block starts here
    import torch.distributed as dist
    init_method=f"tcp://{dp_master_ip}:{dp_master_port+1000}"
    print(
        f"[Rank {global_dp_rank}] Preparing init_process_group with: "
        f"init_method={init_method}, world_size={dp_size}, rank={global_dp_rank}"
    )
    dist.init_process_group(
        backend="gloo",  # "nccl" is preferred for GPU, but "gloo" works fine for barrier-only
        init_method=init_method,
        world_size=dp_size,
        rank=global_dp_rank,
    )
    print(f"[Rank {global_dp_rank}] Waiting at barrier...")
    dist.barrier()
    print(f"[Rank {global_dp_rank}] All ranks done, safe to exit.")
    dist.destroy_process_group()    
    
    
    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":
    args = parse_args()

    DEBUG = False

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size


    if DEBUG:
        print("🚧 Running in SINGLE-PROCESS DEBUG MODE")
        main(
            args.model,
            dp_size,
            0,  # local_dp_rank
            0,  # global_dp_rank
            dp_master_ip,
            dp_master_port,
            tp_size,
            args.enforce_eager,
            args.trust_remote_code,
            args.max_num_seqs,
            args.gpu_memory_utilization,
        )
        exit(0)

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        print(f"local_dp_rank: {local_dp_rank}, global_dp_rank: {global_dp_rank}")
        proc = Process(
            target=main,
            args=(
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                args.enforce_eager,
                args.trust_remote_code,
                args.max_num_seqs,
                args.gpu_memory_utilization,
            ),
        )
        print(f"Process info: name={proc.name}, pid={proc.pid}, exitcode={proc.exitcode}, is_alive={proc.is_alive()}")
        proc.start()
        print(f"Start Process info: name={proc.name}, pid={proc.pid}, exitcode={proc.exitcode}, is_alive={proc.is_alive()}")
        procs.append(proc)
    print(f"All processes starts")
    print(procs)
    print(procs[0])
    exit_code = 0
    for proc in procs:
        print(f"Process info: name={proc.name}, pid={proc.pid}, exitcode={proc.exitcode}, is_alive={proc.is_alive()}")
        proc.join()
        print(f"Process info: name={proc.name}, pid={proc.pid}, exitcode={proc.exitcode}, is_alive={proc.is_alive()}")

    exit(exit_code)
