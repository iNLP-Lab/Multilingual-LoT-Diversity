import argparse
from typing import Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.decomposition import PCA
import os
from types import MethodType
from vllm import LLM
import json
from adjustText import adjust_text
import torch


def plot_single_layer_on_ax(
    ax,
    layer_id,
    language_embedding_mean,
    anchor_language="en",
    max_ring_scale=1.1,
    ring_num=5,
    # --- font/size knobs (bigger by default) ---
    label_fs=22,      # language labels
    title_fs=22,      # subplot title
    tick_fs=22,       # x/y tick labels
    point_s=60,
    anchor_s=70,
    # --- EN label offset (in points) ---
    anchor_text_offset=(3, 3),  # (dx, dy) in points; tweak as you like
):
    lang_dict = language_embedding_mean[layer_id]
    languages = sorted(lang_dict.keys())
    anchor_idx = languages.index(anchor_language)

    # ---- stack embeddings ----
    X = torch.stack([F.normalize(lang_dict[l], dim=-1) for l in languages], dim=0).float()

    # ---- cosine distance ----
    anchor_emb = X[anchor_idx]
    dist_to_anchor = {
        lang: 0.0 if lang == anchor_language
        else 1.0 - torch.dot(anchor_emb, X[i]).item()
        for i, lang in enumerate(languages)
    }

    # ---- PCA directions ----
    X_centered = X - X[anchor_idx]
    pca = PCA(n_components=2)
    X_dir = pca.fit_transform(X_centered.numpy())

    # ---- project to true radius ----
    X_plot = np.zeros_like(X_dir)
    for i, lang in enumerate(languages):
        if lang == anchor_language:
            continue
        r = dist_to_anchor[lang]
        norm = np.linalg.norm(X_dir[i])
        X_plot[i] = X_dir[i] / norm * r if norm > 1e-6 else np.array([r, 0.0])

    # ---- rings ----
    max_dist = max(dist_to_anchor.values()) * max_ring_scale
    ring_step = max_dist / ring_num
    rings = np.arange(ring_step, max_dist + 1e-6, ring_step)
    theta = np.linspace(0, 2 * np.pi, 400)

    for r in rings:
        ax.plot(
            r * np.cos(theta), r * np.sin(theta),
            linestyle='--', color='gray', alpha=0.25, linewidth=0.8
        )

    # ---- points + labels ----
    texts = []
    en_text = None

    for i, lang in enumerate(languages):
        x, y = X_plot[i]

        if lang == anchor_language:
            ax.scatter(x, y, c='red', s=anchor_s, zorder=4)

            en_text = ax.text(
                x, y, lang.upper(),
                fontsize=label_fs + 1,
                fontweight="bold",
                ha="center", va="center",
                zorder=5,
            )
            texts.append(en_text)  #
        else:
            ax.scatter(x, y, c='blue', s=point_s, alpha=0.9)
            txt = ax.text(x, y, lang, fontsize=label_fs, ha='center', va='center')
            texts.append(txt)
            ax.plot([0, x], [0, y], ':', alpha=0.25, linewidth=0.8)

    # ---- adjust text ----
    adjust_text(
        texts,
        ax=ax,
        only_move={'points': 'y', 'texts': 'xy'},
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.35, lw=0.7),
        force_text=0.7,      # 
        force_points=0.35,   #
    )


    if en_text is not None:
        x_txt, y_txt = en_text.get_position()
        x_pt, y_pt = X_plot[anchor_idx]
        pull_ratio = 0.35

        new_x = x_txt + pull_ratio * (x_pt - x_txt)
        new_y = y_txt + pull_ratio * (y_pt - y_txt)

        en_text.set_position((new_x, new_y))

    # ---- cosmetics ----
    lim = max(rings) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')

    ax.axhline(0, color='black', linewidth=0.5, alpha=0.6)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.6)

    ax.set_title(f"Layer {layer_id}", fontsize=title_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)


def plot_multilayer_language_geometry(
    language_embedding_mean,
    layers_to_plot,
    save_path,
    anchor_language="en",
    layout="row",
    max_ring_scale=1.1,
    ring_num=5,
):
    assert len(layers_to_plot) == 4, "Please provide exactly 4 layers."

    # ---------- layout ----------
    if layout == "row":
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    elif layout == "grid":
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
    else:
        raise ValueError("layout must be 'row' or 'grid'")

    for ax, layer_id in zip(axes, layers_to_plot):
        plot_single_layer_on_ax(
            ax=ax,
            layer_id=layer_id,
            language_embedding_mean=language_embedding_mean,
            anchor_language=anchor_language,
            max_ring_scale=max_ring_scale,
            ring_num=ring_num,
        )

    plt.tight_layout()
    fig_path = f"{save_path}_layers_{'_'.join(map(str, layers_to_plot))}_{layout}.png"
    pdf_path = f"{save_path}_layers_{'_'.join(map(str, layers_to_plot))}_{layout}.pdf"
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved picture to {fig_path}")
    print(f"Saved picture to {pdf_path}")


def process_instances_vllm(
    instances,
    pic_output_file,
    rep_output_file,
    llm_model_path,
    args
):  
    
    if os.path.exists(rep_output_file):
        language_embedding_mean = torch.load(rep_output_file)
        print(f"Loaded language embeddings from {rep_output_file}")
    else:
        llm = LLM(model=llm_model_path, task="embed")

        lang_to_texts = {}
        for inst in instances:
            for text, lang in zip(inst["responses"], inst["languages"]):
                if lang not in lang_to_texts:
                    lang_to_texts[lang] = []
                lang_to_texts[lang].append(text)

        num_layers = len(llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)
        layer_embeddings = {lang: [[] for _ in range(num_layers)] for lang in lang_to_texts.keys()}

        # -------- factory hook --------
        def factory_collect(layer_id, lang):
            def hook(self,
                    positions: torch.Tensor,
                    hidden_states: torch.Tensor,
                    residual: Optional[torch.Tensor]):
                if residual is None:
                    residual = hidden_states
                    hidden_states = self.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = self.input_layernorm(hidden_states, residual)

                layer_embeddings[lang][layer_id].append(hidden_states.detach().cpu())

                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                )

                # Fully Connected
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual)
                hidden_states = self.mlp(hidden_states)
                return hidden_states, residual
            
            return hook

        language_embedding_mean = {}

        for lang, texts in lang_to_texts.items():

            emb_ratio = args.emb_ratio if lang != "en" else 0.0
            for layer_id, layer_obj in enumerate(llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers):
                layer_obj.forward_before = layer_obj.forward
                layer_obj.forward = MethodType(factory_collect(layer_id, lang), layer_obj)
            _ = llm.embed(texts)

            for layer_obj in llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers:
                layer_obj.forward = layer_obj.forward_before
                del layer_obj.forward_before


        
            layers = layer_embeddings[lang]
            for layer_id, layer_list in enumerate(layers):
                all_tokens = torch.cat(layer_list, dim=0)  # [num_tokens, H]
                mean_vec = all_tokens.mean(dim=0)          # [H]
                if layer_id not in language_embedding_mean:
                    language_embedding_mean[layer_id] = {}
                language_embedding_mean[layer_id][lang] = mean_vec

        torch.save(language_embedding_mean, rep_output_file)
        print(f"Saved language embeddings to {rep_output_file}")
    
    layers_to_plot = [1, 10, 20, 35]
    plot_multilayer_language_geometry(
        language_embedding_mean=language_embedding_mean,
        layers_to_plot=layers_to_plot,
        save_path=pic_output_file,
        layout="grid"   # or "row"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory to save evaluation results", default=""
    )
    parser.add_argument("--emb_layer", type=int, default=0)
    parser.add_argument("--emb_ratio", type=float, default=0.0)
    parser.add_argument(
        "--output-dir", help="Directory to save evaluation results", required=True
    )

    parser.add_argument(
        "--language", type=str, default="en"
    )

    parser.add_argument(
        "--temperature", type=float
    )

    parser.add_argument(
        "--data", choices=['novelty', 'eqbench', 'infinity'],
    )

    parser.add_argument(
        "--model", type=str, default="Qwen3-8B"
    )

    parser.add_argument(
        "--flag", type=int, default=0
    )

    parser.add_argument(
        "--sample_num", type=int, default=20
    )

    args = parser.parse_args()
    eval_dir = args.eval_dir
    model = args.model
    model_to_load = model
    model = model.split("/")[-1]
    temperature = args.temperature
    flag = args.flag
    sample_num = args.sample_num
    instances = []
    if eval_dir:
        all_data = {}
        languages = ['en', 'it', 'ms', 'zh', 'ru', 'de', 'iw', 'bg', 'da', 'no', 'sv', 'es', 'tl', 'oc', 'fr']
        for language in languages:
            file_name = f"{eval_dir}/{model}_{language}_{temperature}_{sample_num}_0_dp_rank_0.jsonl"
            with open(file_name, 'r') as f:
                lang_data = [json.loads(l) for l in f]
            all_data[language] = lang_data
        item_num = len(all_data[languages[0]])
        tmp_instances = [{"idx":i, "question": None, "responses": [], "languages": []} for i in range(item_num)]

        for language, lang_data in all_data.items():
            assert len(lang_data) == item_num
            for idx, item in enumerate(lang_data):
                question = item['question']
                if not tmp_instances[idx]['question']:
                    tmp_instances[idx]['question'] = question
                else:
                    assert tmp_instances[idx]['question'] == question
                if "gen_prompt" in item:
                    gen_prompts = item['gen_prompt']
                    gen_prompt = gen_prompts[0]
                    thinking_process = gen_prompt.split("\n<think>\n")[-1].split("\n</think>")[0].strip()
                    tmp_instances[idx]['responses'].append(thinking_process)
                    tmp_instances[idx]['languages'].append(language)
        instances = []
        for tmp_instance in tmp_instances:
            if len(tmp_instance['responses']) > 0:
                instances.append(tmp_instance)
        for instance in instances:
            print(len(instance['responses']))
    os.makedirs(args.output_dir, exist_ok=True)
    rep_output_file = f"{args.output_dir}/{model}_{args.emb_layer}_{args.emb_ratio}_multilingual_{temperature}_{sample_num}_{flag}_embedding.pt"
    pic_output_file = f"{args.output_dir}/{model}_{args.emb_layer}_{args.emb_ratio}_multilingual_{temperature}_{sample_num}_{flag}"
    process_instances_vllm(instances, pic_output_file, rep_output_file, model_to_load, args)


if __name__ == "__main__":
    main()