from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from .config import SDHTGModelConfig
from .hierarchy import HierarchyOutput, HierarchyLevel

NodeType = str
EdgeType = tuple[str, str, str]

@dataclass
class GraphBuildResult:
    graphs: list[HeteroData]
    status_counts: Tensor
    action_counts: Tensor
    entity_counts: Tensor

def _empty_edge_index(device: torch.device) -> Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)

def _empty_edge_attr(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.empty((0, 2), dtype=dtype, device=device)

def local_temporal_edges(positions: Tensor, radius: int, self_loops: bool) -> tuple[Tensor, Tensor]:
    count = positions.numel(); device = positions.device; dtype = positions.dtype
    if count == 0: return _empty_edge_index(device), _empty_edge_attr(device, dtype)
    offsets = torch.arange(-radius, radius + 1, device=device)
    if not self_loops: offsets = offsets[offsets != 0]
    src = torch.arange(count, device=device).unsqueeze(1)
    tgt = src + offsets.unsqueeze(0)
    valid = (tgt >= 0) & (tgt < count)
    src = src.expand(-1, offsets.size(0))[valid]
    tgt = tgt[valid]
    if src.numel() == 0: return _empty_edge_index(device), _empty_edge_attr(device, dtype)
    delta = (positions[tgt] - positions[src]).abs()
    weight = torch.exp(-delta / max(float(radius), 1.0))
    return torch.stack((src, tgt), dim=0), torch.stack((weight, delta), dim=-1)

def semantic_edges(semantic_ids: Tensor, positions: Tensor, maximum_neighbors: int) -> tuple[Tensor, Tensor]:
    count = semantic_ids.numel(); device = semantic_ids.device; dtype = positions.dtype
    if count < 2: return _empty_edge_index(device), _empty_edge_attr(device, dtype)
    valid_mask = semantic_ids > 1
    if not valid_mask.any(): return _empty_edge_index(device), _empty_edge_attr(device, dtype)
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    order = torch.argsort(semantic_ids[valid_idx], stable=True)
    sid_sorted = semantic_ids[valid_idx][order]
    pos_sorted = positions[valid_idx][order]
    valid_sorted = valid_idx[order]
    _, counts = torch.unique_consecutive(sid_sorted, return_counts=True)
    src_list, tgt_list, w_list, d_list = [], [], [], []
    start = 0
    for m in counts.tolist():
        if m < 2:
            start += m
            continue
        p = pos_sorted[start:start + m]
        dist = (p.unsqueeze(1) - p.unsqueeze(0)).abs()
        dist.fill_diagonal_(float('inf'))
        k = min(int(maximum_neighbors), m - 1)
        topk_val, topk_idx = dist.topk(k, dim=1, largest=False)
        local_src = torch.arange(m, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        src_idx = valid_sorted[start:start + m][local_src]
        tgt_idx = valid_sorted[start:start + m][topk_idx.reshape(-1)]
        src_list.append(src_idx); tgt_list.append(tgt_idx)
        w_list.append((1.0 / (1.0 + topk_val)).reshape(-1))
        d_list.append(topk_val.reshape(-1))
        start += m
    if not src_list: return _empty_edge_index(device), _empty_edge_attr(device, dtype)
    return torch.stack([torch.cat(src_list), torch.cat(tgt_list)], dim=0), torch.stack([torch.cat(w_list), torch.cat(d_list)], dim=-1)

def containment_edges(membership: Tensor, source_count: int, target_count: int, source_positions: Tensor, target_positions: Tensor, minimum_weight: float, hard_edge_weight: bool = False, maximum_per_target: int = 0) -> tuple[Tensor, Tensor]:
    device = membership.device; dtype = membership.dtype
    if source_count == 0 or target_count == 0: return _empty_edge_index(device), _empty_edge_attr(device, dtype)
    weights = membership[:source_count, :target_count]
    indices = torch.nonzero(weights > minimum_weight, as_tuple=False)
    if indices.numel() == 0: return _empty_edge_index(device), _empty_edge_attr(device, dtype)
    source = indices[:, 0]; target = indices[:, 1]
    if maximum_per_target > 0 and maximum_per_target < source_count:
        # Keep at most Km high-weight containment edges per target node
        # (manuscript 4.7.3); selection is discrete, weights keep gradients.
        # Vectorized per-target top-K: two stable sorts (by target, then by
        # weight descending, then by target again) make each target group
        # consecutive with members ordered by descending weight.
        selected_weights = weights[source, target]
        first = torch.argsort(target, stable=True)
        source = source[first]; target = target[first]
        selected_weights = selected_weights[first]
        second = torch.argsort(selected_weights, descending=True, stable=True)
        source = source[second]; target = target[second]
        selected_weights = selected_weights[second]
        third = torch.argsort(target, stable=True)
        source = source[third]; target = target[third]
        unique_targets, counts = torch.unique_consecutive(target, return_counts=True)
        starts = torch.zeros_like(counts)
        starts[1:] = torch.cumsum(counts[:-1], dim=0)
        group_index = torch.searchsorted(unique_targets, target)
        position = (
            torch.arange(target.numel(), device=device) - starts[group_index]
        )
        keep = position < maximum_per_target
        source = source[keep]; target = target[keep]
        if source.numel() == 0: return _empty_edge_index(device), _empty_edge_attr(device, dtype)
    selected = torch.ones_like(weights[source, target]) if hard_edge_weight else weights[source, target]
    delta = (source_positions[source] - target_positions[target]).abs()
    return torch.stack((source, target), dim=0), torch.stack((selected, delta), dim=-1)

class HeterogeneousGraphBuilder:
    def __init__(self, config: SDHTGModelConfig): self.config = config
    def build(self, hierarchy: HierarchyOutput) -> GraphBuildResult:
        bs = hierarchy.status.features.shape[0]; device = hierarchy.status.features.device; dtype = hierarchy.status.features.dtype
        # One CPU sync per level instead of one per sample.
        sc = hierarchy.status.mask.sum(dim=1).cpu().tolist()
        ac = hierarchy.action.mask.sum(dim=1).cpu().tolist()
        ec = hierarchy.entity.mask.sum(dim=1).cpu().tolist()
        import itertools as _it
        s_off = [0] + list(_it.accumulate(sc)); a_off = [0] + list(_it.accumulate(ac)); e_off = [0] + list(_it.accumulate(ec))
        s_x,s_b=self._cat_batch(hierarchy.status,sc,s_off,device)
        a_x,a_b=self._cat_batch(hierarchy.action,ac,a_off,device)
        e_x,e_b=self._cat_batch(hierarchy.entity,ec,e_off,device)
        TE=self.config.graph_relations.temporal and self.config.ablation.use_temporal_edges
        SE=self.config.graph_relations.semantic and self.config.ablation.use_semantic_edges
        CL=self.config.graph_relations.containment and self.config.ablation.use_cross_level_messages
        ei_t,ea_t,ei_s,ea_s={},{},{},{}
        for n,co,of in [('status',sc,s_off),('action',ac,a_off),('entity',ec,e_off)]:
            l=getattr(hierarchy,n)
            (ei_t[n],ea_t[n])=self._batch_temporal(l,co,of,self.config.local_temporal_radius[n],self.config.graph_relations.self_loops) if TE else(_empty_edge_index(device),_empty_edge_attr(device,dtype))
            (ei_s[n],ea_s[n])=self._batch_semantic(l,co,of,self.config.semantic_neighbors[n]) if SE else(_empty_edge_index(device),_empty_edge_attr(device,dtype))
        if CL:
            sa_ei,sa_ea=self._batch_containment(hierarchy.status_to_action,hierarchy.status,hierarchy.action,sc,ac,s_off,a_off,self.config.hierarchy.edge_minimum_weight,self.config.hierarchy.max_edges_per_target)
            ae_ei,ae_ea=self._batch_containment(hierarchy.action_to_entity,hierarchy.action,hierarchy.entity,ac,ec,a_off,e_off,self.config.hierarchy.edge_minimum_weight,self.config.hierarchy.max_edges_per_target)
        else: sa_ei=ae_ei=_empty_edge_index(device); sa_ea=ae_ea=_empty_edge_attr(device,dtype)
        if self.config.ablation.detach_boundary_from_graph: sa_ea=sa_ea.detach(); ae_ea=ae_ea.detach()
        g=HeteroData()
        for n,x,b in [('status',s_x,s_b),('action',a_x,a_b),('entity',e_x,e_b)]: g[n].x=x; g[n].batch=b
        g._num_graphs=bs
        for n in ['status','action','entity']:
            g[(n,'temporal',n)].edge_index=ei_t[n]; g[(n,'temporal',n)].edge_attr=ea_t[n]
            g[(n,'semantic',n)].edge_index=ei_s[n]; g[(n,'semantic',n)].edge_attr=ea_s[n]
        g[('status','belongs_to','action')].edge_index=sa_ei; g[('status','belongs_to','action')].edge_attr=sa_ea
        g[('action','belongs_to','entity')].edge_index=ae_ei; g[('action','belongs_to','entity')].edge_attr=ae_ea
        if self.config.graph_relations.reverse_containment and CL:
            g[('action','contains','status')].edge_index=sa_ei.flip(0) if sa_ei.numel() else _empty_edge_index(device); g[('action','contains','status')].edge_attr=sa_ea
            g[('entity','contains','action')].edge_index=ae_ei.flip(0) if ae_ei.numel() else _empty_edge_index(device); g[('entity','contains','action')].edge_attr=ae_ea
        else:
            for et in [('action','contains','status'),('entity','contains','action')]: g[et].edge_index=_empty_edge_index(device); g[et].edge_attr=_empty_edge_attr(device,dtype)
        return GraphBuildResult(graphs=[g],status_counts=torch.tensor(sc,device=device,dtype=torch.long),action_counts=torch.tensor(ac,device=device,dtype=torch.long),entity_counts=torch.tensor(ec,device=device,dtype=torch.long))
    @staticmethod
    def _cat_batch(level,counts,offsets,device):
        pts,bps=[],[]
        for i,c in enumerate(counts):
            if c==0: continue
            pts.append(level.features[i,:c]); bps.append(torch.full((c,),i,device=device,dtype=torch.long))
        cat=torch.cat
        return (cat(pts,dim=0) if pts else torch.empty(0,level.features.size(-1),device=device)), (cat(bps,dim=0) if bps else torch.empty(0,device=device,dtype=torch.long))
    def _batch_temporal(self,level,counts,offsets,radius,self_loops):
        d=level.features.device; eiL,eaL=[],[]
        for i,c in enumerate(counts):
            if c<2: continue
            ei,ea=local_temporal_edges(level.positions[i,:c],radius,self_loops)
            if ei.numel(): eiL.append(ei+offsets[i]); eaL.append(ea)
        return (_empty_edge_index(d),_empty_edge_attr(d,level.features.dtype)) if not eiL else (torch.cat(eiL,dim=1),torch.cat(eaL,dim=0))
    def _batch_semantic(self,level,counts,offsets,max_n):
        d=level.features.device; eiL,eaL=[],[]
        for i,c in enumerate(counts):
            if c<2: continue
            ei,ea=semantic_edges(level.semantic_id[i,:c],level.positions[i,:c],max_n)
            if ei.numel(): eiL.append(ei+offsets[i]); eaL.append(ea)
        return (_empty_edge_index(d),_empty_edge_attr(d,level.features.dtype)) if not eiL else (torch.cat(eiL,dim=1),torch.cat(eaL,dim=0))
    def _batch_containment(self,mem,src,tgt,sc_cnt,tc_cnt,sc_off,tc_off,mw,kpt):
        d=mem.device; eiL,eaL=[],[]
        for i in range(len(sc_cnt)):
            s,t=sc_cnt[i],tc_cnt[i]; 
            if s==0 or t==0: continue
            ei,ea=containment_edges(mem[i,:s,:t],s,t,src.positions[i,:s],tgt.positions[i,:t],mw,self.config.ablation.hard_edge_weight,kpt)
            if ei.numel(): eiL.append(torch.stack([ei[0]+sc_off[i],ei[1]+tc_off[i]],dim=0)); eaL.append(ea)
        return (_empty_edge_index(d),_empty_edge_attr(d,tgt.features.dtype)) if not eiL else (torch.cat(eiL,dim=1),torch.cat(eaL,dim=0))
