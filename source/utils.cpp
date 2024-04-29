#include "utils.hpp"

VkCommandBuffer Utils::AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool)
{
	VkCommandBufferAllocateInfo cmdAllocInfo = nvvk::make<VkCommandBufferAllocateInfo>();
	cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdAllocInfo.commandPool = cmdPool;
	cmdAllocInfo.commandBufferCount = 1;
	VkCommandBuffer cmdBuffer;
	NVVK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer));
	VkCommandBufferBeginInfo beginInfo = nvvk::make<VkCommandBufferBeginInfo>();
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
	return cmdBuffer;
}

void Utils::EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer)
{
	NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));
	VkSubmitInfo submitInfo = nvvk::make<VkSubmitInfo>();
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuffer;
	NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
	NVVK_CHECK(vkQueueWaitIdle(queue));
	vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

void Utils::parse(int argc, char* argv[], Options* options)
{
	const std::vector<std::string_view> args(argv + 1, argv + argc);

	for (auto it = args.begin(), end = args.end(); it != end; ++it)
	{
		if (*it == "-h" || *it == "--help")
		{
			std::cout << "Complex Materials Renderer help:\n";
			std::cout << "\t-o\t--out\tSets the name of the output file (default: 'out')\n";
			std::cout << "\t-s\t--samples\tSets the sample count for the render (default: 256)\n";
			std::cout << "\t-b\t--background\tSets the axis-aligned texture for diffuse background (default: 1)\n";
			std::cout << "\t\t0\tNone\n";
			std::cout << "\t\t1\tCheckerboard pattern\n";
			std::cout << "\t\t2\tCornell box (paints vertical planes based on their normals)\n";
			exit(0);
		}

		if (*it == "-o" || *it == "--out") {
			if (it + 1 != end)
			{
				auto& val = *(it + 1);
				options->outName = val;
				it++;
			}
			continue;
		}

		if (*it == "-s" || *it == "--samples") {
			if (it + 1 != end)
			{
				auto& val = *(it + 1);
				std::from_chars(val.data(), val.data() + val.size(), options->numSamples);
				it++;
			}
			continue;
		}

		if (*it == "-b" || *it == "--background") {
			if (it + 1 != end)
			{
				auto& val = *(it + 1);
				std::from_chars(val.data(), val.data() + val.size(), options->backgroundTexture);
				it++;
				if (options->backgroundTexture > 2 || options->backgroundTexture < 0)
				{
					options->backgroundTexture = 0;
				}
			}
			continue;
		}
		options->objPath = *it;
	}
}
