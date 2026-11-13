"""EventHPE official losses (event_pose_estimation/loss_funcs.py at 2af451f), vendored: `compute_losses` and
`compute_flow_loss` verbatim except the marked edits (joint / vertex counts inferred from the tensors, the
flow-validity threshold exposed with the official value 4 px as default, `.view` -> `.reshape` on a permuted
tensor). See README.md.
"""
import torch
import torch.nn.functional as F

from .geometry import batch_rodrigues, projection_torch  # noqa: F401


def compute_losses(out, target, mse_func, device, args):
    losses = {}
    if args.delta_tran_loss > 0:
        delta_tran_loss = torch.mean(torch.pow(out['delta_tran'], 2))
        losses['delta_tran'] = args.delta_tran_loss * delta_tran_loss
    else:
        losses['delta_tran'] = torch.tensor([0], device=device).float()

    if args.tran_loss > 0:
        tran_loss = mse_func(out['tran'], target['tran'])
        losses['tran'] = args.tran_loss * tran_loss
    else:
        losses['tran'] = torch.tensor([0], device=device).float()

    if args.theta_loss > 0:
        B, T, _ = target['theta'].size()
        target_rotmats = batch_rodrigues(target['theta'].view(-1, 3)).view(B, T, -1, 3, 3)  # vendoring edit: 24 -> -1

        if args.use_geodesic_loss:
            eps = 1e-6
            # square geodesic loss arccos[(Tr(R1R2^T) -1 )/2]
            trace_rrt = torch.sum(out['pred_rotmats'] * target_rotmats, dim=(-2, -1))  # [B, T, 24]
            degree_dif = torch.acos(torch.clamp(0.5 * (trace_rrt - 1), -1 + eps, 1 - eps))
            theta_loss = torch.mean(degree_dif)
            # _pred = out['pred_rotmats'].view(-1, 3, 3)
            # _target = target_rotmats.view(-1, 3, 3)
            # m = torch.bmm(_pred, _target.transpose(1, 2))  # batch*3*3
            # cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
            # cos = torch.min(cos, torch.ones([_target.size(0)], requires_grad=True, device=_pred.device))
            # cos = torch.max(cos, - torch.ones([_target.size(0)], requires_grad=True, device=_pred.device))
            # theta = torch.acos(cos)
            # theta_loss = torch.mean(theta)
        else:
            theta_loss = mse_func(out['pred_rotmats'], target_rotmats)
        losses['theta'] = args.theta_loss * theta_loss
    else:
        losses['theta'] = torch.tensor([0], device=device).float()

    if args.joints3d_loss > 0:
        joints3d_loss = mse_func(out['joints3d'], target['joints3d'])
        losses['joints3d'] = args.joints3d_loss * joints3d_loss
    else:
        losses['joints3d'] = torch.tensor([0], device=device).float()

    if args.joints2d_loss > 0:
        # print(torch.max(out['joints2d'].detach()), torch.max(target['joints2d'].detach()))
        pred_joints2d = torch.clamp(out['joints2d'], 0, 1)
        joints2d_loss = mse_func(pred_joints2d, target['joints2d'])
        losses['joints2d'] = args.joints2d_loss * joints2d_loss
    else:
        losses['joints2d'] = torch.tensor([0], device=device).float()

    if args.flow_loss > 0 and args.flow_loss:
        flow_loss = compute_flow_loss(out['verts'], target['flows'], out['cam_intr'], device,
                                      getattr(args, 'flow_valid_px', 4))  # vendoring edit: threshold exposed (official 4)
        losses['flow'] = args.flow_loss * flow_loss
        # losses['flow'] = torch.tensor([0], device=device).float()
    else:
        losses['flow'] = torch.tensor([0], device=device).float()
    return losses


def compute_flow_loss(verts, pred_flows, cam_intr, device, valid_px=4):
    # verts: [B, T+1, 6890, 3]
    # pred_flows: [B, T, 2, H, W]
    B, T, _, H, W = pred_flows.size()
    verts_2d = projection_torch(verts, cam_intr)  # [B, T+1, 6890, 2]
    verts_flow = verts_2d[:, 1:, :, :] - verts_2d[:, :-1, :, :]  # [B, T, 6890, 2] <t+1> - <t>

    scale = torch.tensor([W, H], device=device)
    V = verts.size(2)  # vendoring edit: 6890 -> V (MANO has 778 vertices)
    flow_indices = 2. * verts_2d[:, :-1, :, :].detach().reshape(-1, V, 2) / scale - 1.  # [BT, V, 2]
    flow_indices = F.pad(flow_indices, [0, 0, 0, H * W - V, 0, 0],
                         mode='constant', value=-1).view(-1, H, W, 2)  # [BT, H, W, 2]
    _flows = pred_flows.view(-1, 2, H, W)  # [BT, 2, H, W]
    sampled_flow = F.grid_sample(_flows, flow_indices)  # [BT, 2, H, W]  (u, v)
    sampled_flow = sampled_flow.permute(0, 2, 3, 1).reshape(B, T, H * W, 2)[:, :, :V, :].clone()

    # error = verts_flow - sampled_flow
    # error = torch.clamp(error, min=-5, max=5)
    # loss = torch.mean(torch.pow(error, 2))

    # loss = torch.mean(1 - F.cosine_similarity(verts_flow, sampled_flow, dim=3))

    valid = torch.norm(sampled_flow, p=2, dim=-1) > valid_px
    sim = valid * (1 - F.cosine_similarity(verts_flow, sampled_flow, dim=3))  # [B, T, 6890]
    loss = torch.mean(sim)
    return loss
