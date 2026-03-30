python inference_svd_rgbx.py --config configs/rgbx_inference.yaml inference_input_dir="/home2/guest/datasets/nerfstudio/dl3dv_selected/6b42314a2f8a18a193826e2b58e45729453e74524078283f740b8f8d330c3d2f/images"   inference_save_dir="/home2/guest/datasets/nerfstudio/dl3dv_selected_inference_output/6b42314a2f8a18a193826e2b58e45729453e74524078283f740b8f8d330c3d2f"   inference_n_frames=24 inference_n_steps=20 model_passes="['basecolor','normal','depth','diffuse_albedo']" inference_res="[512,512]"

python inference_svd_rgbx.py --config configs/rgbx_inference.yaml \
  inference_input_dir="/path/to/input_videos" \
  inference_save_dir="output_inference_delighting/" \
  inference_n_frames=24 inference_n_steps=20 model_passes="['basecolor','normal','depth','diffuse_albedo']" inference_res="[512,512]" 


python inference_svd_rgbx.py --config configs/rgbx_inference.yaml \
 inference_input_dir="/home2/guest/datasets/nerfstudio/dl3dv_selected/6b42314a2f8a18a193826e2b58e45729453e74524078283f740b8f8d330c3d2f/images" \
 inference_save_dir="/home2/guest/datasets/nerfstudio/dl3dv_selected_inference_all_frames_test/6b42314a2f8a18a193826e2b58e45729453e74524078283f740b8f8d330c3d2f" \
 inference_n_frames=60 \
 inference_n_steps=20 \
 model_passes="['basecolor','normal','depth','diffuse_albedo']" \
 inference_res="[512,512]" \
 chunk_mode="all"
