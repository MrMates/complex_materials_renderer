# Complex Materials Renderer
This is source code for my bachelor thesis on "Complex Materials Rendering". It's a renderer written for Vulkan, with host code written in C++ and device code written in GLSL, implementing participating media path tracing.

## Hardware requirements
Please note that this project is using `ray query` extension for Vulkan. This *should* (not guaranteed) be supported on all Nvidia GPUs from GTX 1060 and AMD GPUs from Radeon RX 5600 XT. The extension is supported on some ARM and Mac platforms as well.

You can check if your particular GPU supports this extension at Vulkan's official [feature coverage website](https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_KHR_ray_query&platform=all). Without support, this project cannot be compiled nor ran from precompiled binaries either.

## Important note!
Since the program takes more or less exclusive control over the GPU, on Windows you can run into the issue of the program being shut down by Windows Timeout Detection and Recovery (TDR). To avoid this, the registry key TdrDelay has to be properly raised (at least 30-60 seconds). Please refer to official Microsoft documentation to do so: [https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys).

## Dependencies
This project is using **LunarG Vulkan SDK**, available from the official website [https://www.lunarg.com/vulkan-sdk/](https://www.lunarg.com/vulkan-sdk/).

This project also uses [nvpro_core](https://github.com/nvpro-samples/nvpro_core) library from NVIDIA, available under the Apache 2.0 license. This library is required if you intend to compile the source code yourself. The steps to clone the library are specified below.

## Setup
*Note:* If you already have the source code (for example from a physical medium), the provided files are preferred. Forwards compatibility with newer versions of nvpro_core is possible, but not guaranteed.

Clone the required repositories and the renderer project itself
```
git clone --recursive --shallow-submodules https://github.com/nvpro-samples/nvpro_core.git
git clone https://github.com/MrMates/complex_materials_renderer.git
```
and then use CMake to build the renderer source files. The preferred and the only tested method is compiling with MSVC (Visual Studio) for Windows 10, but Vulkan is platform agnostic and so *should* be this code.

If you already have the `complex_materials_renderer` source code, use the first clone command below in the same directory where the root folder of this project (`complex_materials_renderer/`) is located. If you already have both, you should be good to go.

## Running the renderer
When the renderer is compiled, you can typically find the build in `bin_x64/Release` or similar directory. The renderer is off-line and has no GUI, so it needs to be ran from the command line. The binary can be executed directly, without arguments

```
.\complex_materials_renderer_source.exe
```
which will render a default showcase scene.

To render a specific scene, just reference the `.obj` file
```
.\complex_materials_renderer_source.exe my_scene.obj
```
but make sure that both **.mtl** (.obj materials) and **.json** (media definitions) files are present in the same directory as your `.obj` file.

The renderer also supports commandline arguments for setting up the render. Available arguments can be listed using `--help`.

## Setting up a custom scene
Every scene needs to be exported in .obj, with .mtl export enabled. Then, a .json file specifying the media needs to be created. For that, you can use the script `mat_parser.py` located in `resources\scenes\`.

```
python mat_parser.py my_scene.obj
```

The script will walk you through the setup process and then create the output .json file.
