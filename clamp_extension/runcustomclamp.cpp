#include <torch/torch.h>
#include <iostream>

torch::Tensor clamp_custom(torch::Tensor input, double min_val, double max_val);

int main() {
	if (!torch::cuda::is_available()) {
		std::cout << "NO CUDA WTF";
		return 1;	
	}

	try {
		torch::Device device(torch::kCUDA, 0);

		auto input = torch::rand({4096, 4096},
			torch::TensorOptions()
				.device(device)
				.requires_grad(false));

		input = input.contiguous();

		auto output = clamp_custom(input, 0.0, 1.0);

		std::cout << "Input first elements: " << input.index({0, torch::indexing::Slice(0, 5)}) << std::endl;
        	std::cout << "Output first elements: " << output.index({0, torch::indexing::Slice(0, 5)}) << std::endl;
        
        	std::cout << "Clamp operation completed successfully!" << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "Error" << e.what() << std::endl;
		return 1;
	}
	return 0;
}
