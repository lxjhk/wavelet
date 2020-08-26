.PHONY: buildmem
buildmem:
	python cifar_mem.py -a vgg19 --epochs 3 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19 --manualSeed 55
	
.PHONY: buildgrad
buildgrad:
	python test_grad.py -a vgg19 --epochs 3 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19 --manualSeed 55

.PHONY: conda
conda:
	conda activate sm_70_cuda