import typing

from ._types import GPUInfo
from ._exceptions import GPUInfoProviderNotAvailable

BACKEND = "vulkan"

try:
    import vulkan as vk
except ImportError:
    raise GPUInfoProviderNotAvailable()


def get_gpu_info() -> typing.List[typing.Tuple[int, int]]:
    def find_max_allocatable_memory(
        device, memory_type_index: int, heap_size: int
    ) -> int:
        memory_allocate_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=heap_size,
            memoryTypeIndex=memory_type_index,
        )

        low = 0
        high = heap_size
        max_allocatable = 0

        while low <= high:
            # if less than 100mb, then we can allocate it
            if high - low < 100000000:
                return high
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
        pApplicationName="Vulkan Memory Info",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="No Engine",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_API_VERSION_1_0,
    )

    create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app_info
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
                    # Create a logical device
                    queue_family_index = 0  # Assuming first queue family for simplicity
                    queue_create_info = vk.VkDeviceQueueCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                        queueFamilyIndex=queue_family_index,
                        queueCount=1,
                        pQueuePriorities=[1.0],
                    )
                    device_create_info = vk.VkDeviceCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                        queueCreateInfoCount=1,
                        pQueueCreateInfos=[queue_create_info],
                    )
                    logical_device = vk.vkCreateDevice(device, device_create_info, None)

                    max_allocatable_memory = find_max_allocatable_memory(
                        logical_device, j, total_memory
                    )
                    vk.vkDestroyDevice(logical_device, None)

                    if max_allocatable_memory > max_free_memory:
                        max_free_memory = max_allocatable_memory

            if total_memory > max_total_memory:
                max_total_memory = total_memory

        memory_info_list.append((max_total_memory, max_free_memory))

    # Clean up Vulkan instance
    vk.vkDestroyInstance(instance, None)

    return memory_info_list


def get_info():
    return [
        GPUInfo(backend="vulkan", total_memory=total, free_memory=free)
        for total, free in get_gpu_info()
    ]
