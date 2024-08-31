import argparse
from typing import List, Optional



def config_IGEV_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    args = parser.parse_args()
    return args

dinoV2_config_dir_dic = {
    'vit_l': '../model_pack/dinoV2/dinov2/configs/eval/vitl14_pretrain.yaml',
    'vit_l_16': '../model_pack/dinoV2/dinov2/configs/eval/vitl16_pretrain.yaml',
    'vit_b': '../model_pack/dinoV2/dinov2/configs/eval/vitb14_pretrain.yaml',
    'vit_s': '../model_pack/dinoV2/dinov2/configs/eval/vits14_pretrain.yaml',
    'vitl_r': '../model_pack/dinoV2/dinov2/configs/eval/vitl14_reg4_pretrain.yaml',
    'vitb_r': '../model_pack/dinoV2/dinov2/configs/eval/vitb14_reg4_pretrain.yaml',
    'vits_r': '../model_pack/dinoV2/dinov2/configs/eval/vits14_reg4_pretrain.yaml',
}
dinoV2_ckpt_dir_dic = {
    'vit_l': '../toolkit/models/dinoV2/dinov2_vitl14_pretrain.pth',
    'vit_l_16': '../toolkit/models/dinoV2/dinov2_vitl16_pretrain.pth',
    'vit_b': '../toolkit/models/dinoV2/dinov2_vitb14_pretrain.pth',
    'vit_s': '../toolkit/models/dinoV2/dinov2_vits14_pretrain.pth',
    'vitl_r': '../toolkit/models/dinoV2/dinov2_vitl14_reg4_pretrain.pth',
    'vitb_r': '../toolkit/models/dinoV2/dinov2_vitb14_reg4_pretrain.pth',
    'vits_r': '../toolkit/models/dinoV2/dinov2_vits14_reg4_pretrain.pth',
}

def get_dinov2_args_parser_2(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default='models/dinoV2/outputs',
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser

def get_dinov2_args_parser_1(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_dinov2_args_parser_2(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--nb_knn",
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--n-per-class-list",
        nargs="+",
        type=int,
        help="Number to take per class",
    )
    parser.add_argument(
        "--n-tries",
        type=int,
        help="Number of tries",
    )
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        nb_knn=[10, 20, 100, 200],
        temperature=0.07,
        batch_size=256,
        n_per_class_list=[-1],
        n_tries=1,
    )
    return parser
