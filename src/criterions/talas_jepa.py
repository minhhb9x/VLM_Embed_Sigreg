import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from src.criterions.utils import count_clean_text_tokens, get_hidden_text_vision, pooling
import random
import math
from torch.nn.utils.rnn import pad_sequence
from torch.distributed.nn.functional import all_reduce as grad_all_reduce
from scipy.optimize import linear_sum_assignment

class TalasJepa(nn.Module):
    def __init__(self, args):
        super(TalasJepa, self).__init__()
        self.args = args
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
        self.kd_weight = args.kd_weight

        self.counter = 0
        self.warm_up_sigreg = 0

        self.centroids = nn.Parameter(torch.randn(args.num_centroids, args.centroid_hidden_size))
    
    def _dist_gather_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def cosine_loss(self, student_embeddings, teacher_embeddings):
        cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1)
        cos_sim_loss = 1 - cos_sim
        return cos_sim_loss.mean()

    def structure_loss(self, student_embeddings, teacher_embeddings):
        student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)

        student_similarity = student_embeddings @ student_embeddings.transpose(-1, -2)
        teacher_similarity = teacher_embeddings @ teacher_embeddings.transpose(-1, -2)

        loss = F.mse_loss(student_similarity, teacher_similarity)

        return loss

    def distillcse_kd_loss(self, S1, S2, T1, T2, tau=0.02):
        """
        Distill teacher similarity distribution over in-batch negatives.

        Student and teacher dimensions do not need to match because
        distillation is applied to pairwise similarity matrices.
        """
        S1 = F.normalize(S1.float(), p=2, dim=-1)

        S2 = F.normalize(S2.float(), p=2, dim=-1)

        T1 = F.normalize(T1.float(), p=2, dim=-1)

        T2 = F.normalize(T2.float(), p=2, dim=-1,)

        s_logits = (S1 @ S2.transpose(0, 1)) / tau

        t_logits = (T1 @ T2.transpose(0, 1)) / tau

        # Positive query-passage pairs are on the diagonal.
        # DistillCSE KD here focuses on the negative distribution.
        mask = torch.eye(s_logits.size(0), device=s_logits.device, dtype=torch.bool,)

        s_logits = s_logits.masked_fill(mask, torch.finfo(s_logits.dtype).min,)

        t_logits = t_logits.masked_fill( mask,torch.finfo(t_logits.dtype).min,)

        teacher_probs = F.softmax(t_logits,dim=1,).detach()

        student_log_probs = F.log_softmax(s_logits,dim=1,)

        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean",)
    
    def sigreg(self, x: torch.Tensor, num_slices: int = 256) -> torch.Tensor:
        device = x.device
        # =====================================================
        # 1. Random projection seed
        #
        # Chỉ rank 0 sinh seed.
        # Sau đó broadcast để tất cả GPU dùng cùng seed.
        # =====================================================
        if self.process_rank == 0:
            projection_seed = random.randint(0, 2**63 - 1)
        else:
            projection_seed = 0

        if self.world_size > 1:
            seed_tensor = torch.tensor(projection_seed, dtype=torch.int64, device=device,)
            dist.broadcast(seed_tensor, src=0)
            projection_seed = seed_tensor.item()

        # =====================================================
        # 2. Local generator
        # =====================================================
        g = torch.Generator(device=device)
        g.manual_seed(projection_seed)

        A = torch.randn(x.size(1), num_slices, generator=g,  device=device, dtype=x.dtype,)

        A = A / A.norm(p=2, dim=0, keepdim=True, ).clamp_min(1e-12)

        # =====================================================
        # 3. Epps-Pulley statistic
        # =====================================================
        t = torch.linspace(-5, 5, 17, device=device, dtype=x.dtype,)

        exp_f = torch.exp(-0.5 * t.square())

        # x:   [N, K]
        # A:   [K, M]
        # x@A: [N, M]
        # x_t: [N, M, T]
        x_t = (x @ A).unsqueeze(-1) * t

        # [M, T]
        ecf = torch.exp(1j * x_t).mean(dim=0)

        # =====================================================
        # 4. Aggregate across GPUs
        # =====================================================
        if self.world_size > 1:
            dist.all_reduce(ecf, op=dist.ReduceOp.SUM,)
            ecf = ecf / self.world_size

        # =====================================================
        # 5. Weighted L2 distance
        # =====================================================
        err = ((ecf - exp_f).abs().square().mul(exp_f))

        global_batch_size = x.size(0) * self.world_size

        sigreg_per_slice = (torch.trapezoid(err, t, dim=1,) * global_batch_size)

        return sigreg_per_slice.mean()

    def kd_sigreg(
        self,
        student_x: torch.Tensor,
        teacher_x: torch.Tensor,
        num_slices: int = 256,
        num_t: int = 17,
        t_max: float = 5.0,
    ) -> torch.Tensor:
        if student_x.ndim != 2 or teacher_x.ndim != 2:
            raise ValueError("student_x và teacher_x phải có shape [N, D]")
        if student_x.size(0) != teacher_x.size(0):
            raise ValueError("Student và teacher phải có cùng số mẫu")
        if student_x.device != teacher_x.device:
            raise ValueError("Student và teacher phải ở cùng device")

        device = student_x.device
        distributed = self.world_size > 1

        projection_seed = random.randint(0, 2**63 - 1) if self.process_rank == 0 else 0
        if distributed:
            seed_tensor = torch.tensor(projection_seed, dtype=torch.int64, device=device)
            dist.broadcast(seed_tensor, src=0)
            projection_seed = int(seed_tensor.item())

        generator = torch.Generator(device=device)
        generator.manual_seed(projection_seed)

        # Tính phép chiếu và ECF bằng float32 ngay cả khi model chạy bf16.
        with torch.autocast(device_type=device.type, enabled=False):
            A = torch.randn(student_x.size(1), num_slices, generator=generator,
                            device=device, dtype=torch.float32)
            A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)

            B = torch.randn(teacher_x.size(1), num_slices, generator=generator,
                            device=device, dtype=torch.float32)
            B = B / B.norm(dim=0, keepdim=True).clamp_min(1e-12)

            student_proj = student_x.float() @ A
            teacher_proj = teacher_x.float() @ B
            t = torch.linspace(-t_max, t_max, num_t, device=device, dtype=torch.float32)

            student_phase = student_proj.unsqueeze(-1) * t
            student_sum = torch.stack(
                [student_phase.cos().sum(dim=0), student_phase.sin().sum(dim=0)],
                dim=-1,
            )
            
            teacher_phase = teacher_proj.unsqueeze(-1) * t
            teacher_sum = torch.stack(
                [teacher_phase.cos().sum(dim=0), teacher_phase.sin().sum(dim=0)],
                dim=-1,
            )

            count = torch.tensor(student_x.size(0), device=device, dtype=torch.float32)
            if distributed:
                student_sum = grad_all_reduce(student_sum, op=dist.ReduceOp.SUM)
                with torch.no_grad():
                    dist.all_reduce(teacher_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count, op=dist.ReduceOp.SUM)

            student_ecf = student_sum / count  # [M, T, 2]
            teacher_ecf = teacher_sum / count  # [M, T, 2]

            weight = torch.exp(-0.5 * t.square())
            sqrt_weight = (weight / weight.sum()).sqrt()
            student_feat = (student_ecf * sqrt_weight[None, :, None]).reshape(num_slices, -1)
            teacher_feat = (teacher_ecf * sqrt_weight[None, :, None]).reshape(num_slices, -1)

            cost = torch.cdist(student_feat, teacher_feat, p=2).square()  # [M, M]

            # Chọn phép ghép 1–1 có tổng cost nhỏ nhất.
            with torch.no_grad():
                row_np, col_np = linear_sum_assignment(cost.detach().cpu().numpy())

            row = torch.as_tensor(row_np, device=device, dtype=torch.long)
            col = torch.as_tensor(col_np, device=device, dtype=torch.long)

            return count * cost[row, col].mean()


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


    def sigreg_dualview(self, z_list: list[torch.Tensor], eos_query: torch.Tensor,
                        num_slices: int = 256, tau: float = 0.05, alpha: float = 0.9):
        B = len(z_list)
        if B == 0:
            return 0.0

        device, dtype = z_list[0].device, z_list[0].dtype
        D = z_list[0].shape[-1]

        # ==========================================
        # 0. PADDING & MASK
        # ==========================================
        lengths = torch.tensor([x.size(0) for x in z_list], device=device)
        N_max = lengths.max().item()
        z_padded = pad_sequence(z_list, batch_first=True, padding_value=0.0)  # [B, N_max, D]

        idx = torch.arange(N_max, device=device).unsqueeze(0)   # [1, N_max]
        mask = idx < lengths.view(B, 1)                          # [B, N_max]

        # ==========================================
        # 1. VIEW 1: ATTENTION-WEIGHTED POOLING
        # ==========================================
        q = F.normalize(eos_query.to(device=device, dtype=dtype), p=2, dim=-1)   # [B, D]
        k = F.normalize(z_padded, p=2, dim=-1)                                   # [B, N_max, D]

        score = torch.einsum('bd,bnd->bn', q, k) / tau                           # [B, N_max]
        score = score.masked_fill(~mask, -float('inf'))
        attn_w = torch.softmax(score, dim=-1)                                    # [B, N_max]

        attn_view = torch.einsum('bn,bnd->bd', attn_w, z_padded)                 # [B, D]

        z_k_concepts = attn_view.unsqueeze(0)

        if alpha < 1.0:
            noise = torch.randn_like(z_k_concepts)
            z_mixed = math.sqrt(alpha) * z_k_concepts + math.sqrt(1.0 - alpha) * noise
        else:
            z_mixed = z_k_concepts

        if self.process_rank == 0:
            projection_seed = random.randint(0, 2**63 - 1)
        else:
            projection_seed = 0
        g = torch.Generator(device=device)
        g.manual_seed(projection_seed)
        
        A = torch.randn(D, num_slices, generator=g, device=device, dtype=dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        t = torch.linspace(-5, 5, 17, device=device, dtype=dtype)
        exp_f = torch.exp(-0.5 * t.square())

        x_proj = z_mixed @ A                   # [K, B, num_slices]
        x_t = x_proj.unsqueeze(-1) * t         # [K, B, num_slices, 17]

        ecf_real = torch.cos(x_t).mean(dim=1)  # [K, num_slices, 17]
        ecf_imag = torch.sin(x_t).mean(dim=1)  # [K, num_slices, 17]

        err = ((ecf_real - exp_f).square() + ecf_imag.square()).mul(exp_f)

        loss_sigreg = torch.trapezoid(err, t, dim=-1).mean(dim=-1) * B

        return loss_sigreg.mean()
    

    def sigreg_sinkhorn(
        self,
        z_list: list[torch.Tensor],
        concept_queries: torch.Tensor,
        num_slices: int = 256,
        tau: float = 0.05,
        n_iters: int = 3,
    ) -> torch.Tensor:
        B = len(z_list)
        if B == 0:
            raise ValueError("z_list không được rỗng")
        if tau <= 0 or n_iters < 1:
            raise ValueError("Cần tau > 0 và n_iters >= 1")

        K, D = concept_queries.shape
        device = z_list[0].device
        if self.centroids.device != device:
            self.centroids.data = self.centroids.data.to(device)
        lengths = torch.tensor([z.size(0) for z in z_list], device=device)

        if K == 0 or (lengths == 0).any():
            raise ValueError("Cần ít nhất một concept và một image token mỗi ảnh")
        if any(z.ndim != 2 or z.size(1) != D for z in z_list):
            raise ValueError("Mỗi phần tử z_list phải có shape [N_i, D]")

        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

        with torch.autocast(device_type=device.type, enabled=False):
            z_padded = pad_sequence(z_list, batch_first=True, padding_value=0).float()
            N_max = z_padded.size(1)
            valid = torch.arange(N_max, device=device)[None, :] < lengths[:, None]

            # Normalize chỉ để tính cosine cost, không thay token dùng để tạo centroid.
            q = F.normalize(concept_queries.float(), dim=-1)
            z_for_cost = F.normalize(z_padded, dim=-1)
            cost = 1 - torch.einsum("kd,bnd->bkn", q, z_for_cost)

            neg_large = -1e9
            log_kernel = (-cost / tau).masked_fill(~valid[:, None, :], neg_large)
            log_u = torch.zeros(B, K, device=device)
            log_v = torch.zeros(B, N_max, device=device).masked_fill(~valid, neg_large)
            log_token_mass = -lengths.float().log()

            # Giữ gradient qua Sinkhorn để concept_queries có thể học.
            for _ in range(n_iters):
                log_u = -math.log(K) - torch.logsumexp(
                    log_kernel + log_v[:, None, :], dim=2
                )
                next_log_v = log_token_mass[:, None] - torch.logsumexp(
                    log_kernel + log_u[:, :, None], dim=1
                )
                log_v = next_log_v.masked_fill(~valid, neg_large)

            log_plan = log_kernel + log_u[:, :, None] + log_v[:, None, :]
            plan = log_plan.masked_fill(~valid[:, None, :], neg_large).exp()
            weights = plan / plan.sum(dim=2, keepdim=True).clamp_min(1e-12)
            z_centroids = torch.bmm(weights, z_padded)  # [B, K, D]

            # Không L2-normalize centroid, không thêm noise.
            x = z_centroids.reshape(B * K, D)         # [B*K, D]

            # Cùng hướng chiếu trên các GPU nếu dùng DDP.
            seed = random.randint(0, 2**63 - 1) if not distributed or dist.get_rank() == 0 else 0
            if distributed:
                seed_tensor = torch.tensor(seed, device=device, dtype=torch.int64)
                dist.broadcast(seed_tensor, src=0)
                seed = int(seed_tensor.item())

            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            A = torch.randn(D, num_slices, generator=generator, device=device, dtype=torch.float32)
            A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)

            t = torch.linspace(-5, 5, 17, device=device, dtype=torch.float32)
            target_cf = torch.exp(-0.5 * t.square())

            phase = (x @ A).unsqueeze(-1) * t         # [B*K, M, 17]
            ecf_sum = torch.stack(
                [phase.cos().sum(dim=0), phase.sin().sum(dim=0)],
                dim=-1,
            )                                         # [M, 17, 2]

            image_count = torch.tensor(B, device=device, dtype=torch.float32)
            if distributed:
                ecf_sum = grad_all_reduce(ecf_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(image_count, op=dist.ReduceOp.SUM)

            ecf = ecf_sum / (image_count * K)
            err = (
                (ecf[..., 0] - target_cf).square() + ecf[..., 1].square()
            ) * target_cf

            return torch.trapezoid(err, t, dim=1).mean() * image_count
    
    def get_image_hidden_states(self, student_tokenizer, 
                                student_qry_input,
                                student_pos_input,
                                student_qry_output, 
                                student_pos_output):
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        student_special_ids = torch.tensor(
            list(set(list(student_tokenizer.added_tokens_encoder.values()) + student_tokenizer.all_special_ids) 
                    - set([student_tokenizer.eos_token_id])),
            device=student_qry_reps.device,
            dtype=torch.long
        )

        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)


        selected_layer = self.args.num_layers

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



    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model
        student_processor = model_wrapper.get_processor()
        student_tokenizer = student_processor.tokenizer 
        projectors = model_wrapper.projectors

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)
        self.counter += batch_size

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device
        dtype = student_qry_reps.dtype

        teacher_qry, teacher_pos = input_data["teacher_qry_caches"], input_data["teacher_pos_caches"]

        teacher_qry_reps = torch.stack([rep['rep'] for rep in teacher_qry], dim=0).to(device, dtype=dtype)
        teacher_pos_reps = torch.stack([rep['rep'] for rep in teacher_pos], dim=0).to(device, dtype=dtype)

        tea_img_qry_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_qry], dim=0).to(device, dtype=dtype) if teacher_qry[0]['mean_last_img_token'] is not None else None
        tea_img_pos_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_pos], dim=0).to(device, dtype=dtype) if teacher_pos[0]['mean_last_img_token'] is not None else None

        tea_text_qry_reps = torch.stack([rep['mean_last_text_token'] for rep in teacher_qry], dim=0).to(device, dtype=dtype) if teacher_qry[0]['mean_last_text_token'] is not None else None
        tea_text_pos_reps = torch.stack([rep['mean_last_text_token'] for rep in teacher_pos], dim=0).to(device, dtype=dtype) if teacher_pos[0]['mean_last_text_token'] is not None else None
        
        if getattr(self, 'world_size', 1) > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
            all_teacher_qry_reps = self._dist_gather_tensor(teacher_qry_reps)
            all_teacher_pos_reps = self._dist_gather_tensor(teacher_pos_reps)
            student_qry_hidden_states = [self._dist_gather_tensor(h) for h in student_qry_hidden_states]
            student_pos_hidden_states = [self._dist_gather_tensor(h) for h in student_pos_hidden_states]
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

        
        ##################################
        # student_special_ids = torch.tensor(
        #     list(set(list(student_tokenizer.added_tokens_encoder.values()) + student_tokenizer.all_special_ids) 
        #          - set([student_tokenizer.eos_token_id])),
        #     device=student_qry_input['input_ids'].device,
        #     dtype=torch.long
        # )

        # num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        # num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)


        # ==============================================================

        kd_loss = torch.zeros_like(contrastive_loss)
        if self.args.use_distill_loss:
            
            unnorm_student_qry_reps = pooling(student_qry_hidden_states[-1], student_qry_input['attention_mask'], mode='eos', normalize=False)
            unnorm_student_pos_reps = pooling(student_pos_hidden_states[-1], student_pos_input['attention_mask'], mode='eos', normalize=False)
            student_rkd_reps = torch.cat([unnorm_student_qry_reps, unnorm_student_pos_reps], dim=0)
            teacher_rkd_reps = torch.cat([all_teacher_qry_reps, all_teacher_pos_reps], dim=0)

            kd_loss = self.structure_loss(student_rkd_reps, teacher_rkd_reps)
            # print(f'structure_loss: {kd_loss.item()}')

        sigreg_loss = torch.zeros_like(contrastive_loss)       
        if self.args.use_sigreg_loss:

            vision_hidden_states = self.get_image_hidden_states(
                                student_tokenizer, 
                                student_qry_input,
                                student_pos_input,
                                student_qry_output, 
                                student_pos_output)
            sigreg_loss = self.sigreg_sinkhorn(vision_hidden_states, 
                self.centroids, num_slices=256, tau=0.05, n_iters=3)

            # print(f'sigreg_loss: {sigreg_loss.item()}')


        # overall loss
        loss = contrastive_loss + self.kd_weight * kd_loss + self.args.sigreg_weight * sigreg_loss
        
        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': kd_loss,
            'sigreg_loss': sigreg_loss,
        }