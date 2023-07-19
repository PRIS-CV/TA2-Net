################### Test URL Model with Task-specific Adapters (TSA) ###################
# url + residual adapters (matrix, initialized as identity matrices) + pre-classifier alignment 
#CUDA_VISIBLE_DEVICES=0 python test_extractor_tsa.py --model.name=imagenet-net --model.dir ./saved_results/sdl \
#--test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha --test.tsa-init eye --test.mode sdl --test.type 5shot
CUDA_VISIBLE_DEVICES=2 python main.py --model.name=url --model.dir ./saved_results/url \
 --test.mode mdl --test.type 5shot

