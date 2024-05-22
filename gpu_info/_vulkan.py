import typing

from ._exceptions import GPUInfoProviderNotAvailable

try:
    import vulkan as vk
except ImportError:
    raise GPUInfoProviderNotAvailable()

def get_gpu_info() -> typing.List[typing.Tuple[int, int]]:
    def find_max_allocatable_memory(device: int, memory_type_index: int, heap_size: int) -> int:
        memory_allocate_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=heap_size,
            memoryTypeIndex=memory_type_index
        )
        
        low = 0
        high = heap_size
        max_allocatable = 0

        while low <= high:
            mid = (low + high) // 2
            memory_allocate_info.allocationSize = mid
            try:
                memory = vk.vkAllocateMemory(device, memory_allocate_info, None)
                vk.vkFreeMemory(device, memory, None)
                max_allocatable = mid
                low = mid + 1
            except vk.VkError:
                high = mid - 1

        return max_allocatable

    # Initialize Vulkan instance
    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName='Vulkan Memory Info',
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName='No Engine',
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_API_VERSION_1_0
    )

    create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info
    )

    instance = vk.vkCreateInstance(create_info, None)

    # Enumerate physical devices
    physical_devices = vk.vkEnumeratePhysicalDevices(instance)
    memory_info_list = []

    for device in physical_devices:
        mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(device)
        max_total_memory = 0
        max_free_memory = 0

        for i in range(mem_properties.memoryHeapCount):
            heap = mem_properties.memoryHeaps[i]
            total_memory = heap.size

            for j in range(mem_properties.memoryTypeCount):
                if mem_properties.memoryTypes[j].heapIndex == i:
                    max_allocatable_memory = find_max_allocatable_memory(device, j, total_memory)
                    if max_allocatable_memory > max_free_memory:
                        max_free_memory = max_allocatable_memory
            
            if total_memory > max_total_memory:
                max_total_memory = total_memory

        memory_info_list.append((max_total_memory, max_free_memory))

    # Clean up Vulkan instance
    vk.vkDestroyInstance(instance, None)

    return memory_info_list