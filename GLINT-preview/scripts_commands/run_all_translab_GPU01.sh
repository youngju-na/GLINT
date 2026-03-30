# synthetic scenes
# evc-train -c configs/exps/envgs/blender_synthetic/7fe08405-d4de-48f9-9435-ba3c18de84b6.yaml exp_name=envgs/blender_synthetic/synthetic-0825-run/7fe08405-d4de-48f9-9435-ba3c18de84b6
CUDA_VISIBLE_DEVICES=1 evc-train -c configs/exps/envgs/translab/scene_01.yaml exp_name=envgs/translab/0903-translab/scene_01
CUDA_VISIBLE_DEVICES=1 evc-train -c configs/exps/envgs/translab/scene_02.yaml exp_name=envgs/translab/0903-translab/scene_02
CUDA_VISIBLE_DEVICES=1 evc-train -c configs/exps/envgs/translab/scene_03.yaml exp_name=envgs/translab/0903-translab/scene_03
CUDA_VISIBLE_DEVICES=1 evc-train -c configs/exps/envgs/translab/scene_04.yaml exp_name=envgs/translab/0903-translab/scene_04


