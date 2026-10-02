import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from .utils import count_clean_text_tokens, get_hidden_text_vision, get_attn_text_vision, pooling
import random
import math
from torch.nn.utils.rnn import pad_sequence

class VisCosineReg(nn.Module):
    def __init__(self, args):
        super(VisCosineReg, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.kd_loss_weight = self.args.kd_weight
        
        # KLD Loss thường dùng reduction='batchmean' cho xác suất
        self.kld_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors


    def mi_loss(
        self,
        student_features: torch.Tensor,  # [B, D]
        teacher_features: torch.Tensor,  # [B, D]
        temperature: float = 0.02,
    ) -> torch.Tensor:
        """Equation (13) của PGA-KD: student_i khớp teacher_i."""
        if student_features.ndim != 2 or student_features.shape != teacher_features.shape:
            raise ValueError("student_features và teacher_features phải cùng shape [B, D]")
        if temperature <= 0:
            raise ValueError("temperature phải > 0")

        student = F.normalize(student_features.float(), dim=-1)
        teacher = F.normalize(teacher_features.float(), dim=-1)

        # logits[i, j] = cosine(student_i, teacher_j) / temperature
        logits = student @ teacher.T / temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)

        return F.cross_entropy(logits, labels)

    def extract_last_mean_attention(self, attentions, attention_mask):
        attn = attentions[-1].mean(dim=1)  # [B, L, L]
        valid = attention_mask.to(device=attn.device, dtype=torch.bool)

        result = []
        for i in range(attn.shape[0]):
            valid_idx = valid[i].nonzero(as_tuple=True)[0]  # [L_i]

            if valid_idx.numel() == 0:
                raise ValueError(f"Sample {i} không có token hợp lệ")

            last_idx = valid_idx[-1]                # query hợp lệ cuối
            weights = attn[i, last_idx, :]          # [L]
            result.append(weights)

        return torch.stack(result)


    def get_image_hidden_states(self, 
                                num_student_text_qry_tokens, 
                                num_student_text_pos_tokens,
                                selected_layer,
                                student_qry_input,
                                student_pos_input,
                                student_qry_output, 
                                student_pos_output):
        
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        batch_size = student_qry_reps.size(0)
        
        vision_hidden_states = []
        cur_idx_qry_img = 0
        cur_idx_pos_img = 0
        for i in range(batch_size):
            # 1. QUERY Processing
            if student_qry_image_features is not None:
                if cur_idx_qry_img < len(student_qry_image_features):
                    # --- Vision ---
                    stu_feat = student_qry_image_features[cur_idx_qry_img]
                    
                    stu_text_hidden_state, vision_hidden_state = get_hidden_text_vision(
                        student_qry_hidden_states[selected_layer][i],
                        num_student_text_qry_tokens[i].item(),
                        stu_feat.size(0),
                        student_qry_input['attention_mask'][i]
                    )
                    vision_hidden_states.append(vision_hidden_state)
                    cur_idx_qry_img += 1

            # 2. POSITIVE Processing (Tương tự Query)
            if student_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features):
                    # --- Vision ---
                    stu_feat_pos = student_pos_image_features[cur_idx_pos_img]

                    stu_text_hidden_state, vision_hidden_state = get_hidden_text_vision(
                        student_pos_hidden_states[selected_layer][i],
                        num_student_text_pos_tokens[i].item(),
                        stu_feat_pos.size(0),
                        student_pos_input['attention_mask'][i]
                    )
                    vision_hidden_states.append(vision_hidden_state)
                    cur_idx_pos_img += 1

        return vision_hidden_states


    def vision_diversity_loss(
        self,
        early_states: list[torch.Tensor],
        selected_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Mỗi list gồm các tensor [N_i, D] theo cùng thứ tự sample.

        Returns:
            loss
            threshold: mean cosine layer đầu của batch
            selected_mean: mean cosine layer được regularize
        """
        if not early_states:
            return None, None, None

        if len(early_states) != len(selected_states):
            raise ValueError("Hai list phải chứa cùng số ảnh")

        for i, (early, selected) in enumerate(zip(early_states, selected_states)):
            if early.shape != selected.shape:
                raise ValueError(
                    f"Ảnh {i}: shape layer đầu {tuple(early.shape)} "
                    f"khác layer được chọn {tuple(selected.shape)}"
                )

        device = selected_states[0].device
        lengths = torch.tensor(
            [tokens.size(0) for tokens in selected_states],
            device=device,
        )
        usable = lengths >= 2

        if not usable.any():
            return None, None, None

        # [B, N_max, D]. Layer đầu chỉ dùng làm mốc, không nhận gradient từ loss này.
        early_padded = pad_sequence(
            [tokens.detach() for tokens in early_states],
            batch_first=True,
            padding_value=0.0,
        )
        selected_padded = pad_sequence(
            selected_states,
            batch_first=True,
            padding_value=0.0,
        )

        B, N_max, D = selected_padded.shape
        valid = torch.arange(N_max, device=device)[None, :] < lengths[:, None]

        pair_mask = valid[:, :, None] & valid[:, None, :]
        diagonal = torch.eye(N_max, device=device, dtype=torch.bool)
        pair_mask = pair_mask & ~diagonal[None, :, :]

        with torch.autocast(device_type=device.type, enabled=False):
            # [2, B, N_max, D]: 0 = layer đầu, 1 = layer được chọn.
            x = torch.stack(
                [early_padded.float(), selected_padded.float()],
                dim=0,
            )
            x = F.normalize(x, p=2, dim=-1)

            # Gộp hai layer vào batch để tính cosine cùng lúc.
            x = x.reshape(2 * B, N_max, D)
            cosine = torch.bmm(x, x.transpose(1, 2))
            cosine = cosine.reshape(2, B, N_max, N_max)

            pair_sum = cosine.masked_fill(
                ~pair_mask[None, :, :, :], 0.0
            ).sum(dim=(-1, -2))  # [2, B]

            num_pairs = lengths * (lengths - 1)
            mean_per_image = pair_sum[:, usable] / num_pairs[usable][None, :]

            threshold = mean_per_image[0].mean().detach()
            selected_cosine = mean_per_image[1]

            loss = F.relu(selected_cosine - threshold).mean()

        return loss, threshold, selected_cosine.detach().mean()

    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model
        projectors = model_wrapper.projectors        
        student_processor = model_wrapper.get_processor()
        student_tokenizer = student_processor.tokenizer 

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device

        teacher_qry_reps, teacher_pos_reps = input_data["teacher_qry_caches"], input_data["teacher_pos_caches"] # list of objects, each object is a tensor of shape [batch_size, hidden_dim]
        teacher_qry_reps = torch.stack([rep['rep'] for rep in teacher_qry_reps], dim=0)
        teacher_pos_reps = torch.stack([rep['rep'] for rep in teacher_pos_reps], dim=0)

        teacher_qry_reps = teacher_qry_reps.to(device)
        teacher_pos_reps = teacher_pos_reps.to(device)
        
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
            all_teacher_qry_reps = self._dist_gather_tensor(teacher_qry_reps)
            all_teacher_pos_reps = self._dist_gather_tensor(teacher_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            all_teacher_qry_reps = teacher_qry_reps
            all_teacher_pos_reps = teacher_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / model_wrapper.temperature, target)

        student_special_ids = torch.tensor(
            list(set(list(student_tokenizer.added_tokens_encoder.values()) + student_tokenizer.all_special_ids) 
                    - set([student_tokenizer.eos_token_id])),
            device=student_qry_reps.device,
            dtype=torch.long
        )

        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)
        

        selected_layer = self.args.num_layers

        first_vision_hidden_states = self.get_image_hidden_states(
                                               num_student_text_qry_tokens, 
                                               num_student_text_pos_tokens,
                                               1,
                                               student_qry_input,
                                               student_pos_input,
                                               student_qry_output, 
                                               student_pos_output)
        
        vision_hidden_states = self.get_image_hidden_states(
                                       num_student_text_qry_tokens, 
                                       num_student_text_pos_tokens,
                                       selected_layer,
                                       student_qry_input,
                                       student_pos_input,
                                       student_qry_output, 
                                       student_pos_output)
        
        vision_cosine_reg, early_threshold, _ = self.vision_diversity_loss(first_vision_hidden_states, vision_hidden_states)

        loss = contrastive_loss + self.kd_loss_weight * vision_cosine_reg

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': vision_cosine_reg,
            'sigreg_loss': early_threshold
        }
