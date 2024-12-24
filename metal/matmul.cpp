#include <tt_metal/host_api.hpp>

int main() {
    Device* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();

    constexpr uint32_t N = 1024;
    
    // allocate L1
    tt::tt_metal::InterleavedBufferConfig l1_config {
        .device=device,
        .size=0x2000,
        .page_size=0x2000,
        .buffer_type = tt::tt_metal::BufferType::L1
    };

    std::shared_ptr<tt::tt_metal::Buffer> l1_buffer = CreateBuffer(l1_config);
    
    // allocate dram
    tt::tt_metal::InterleavedBufferConfig dram_config {
        .device = device,
        .size = 0x1000,
        .page_size = 0x1000,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    {
        std::vector<uint32_t> input_vec;
        for (int i = 0; i < 0x400; i++) input_vec.push_back(6);
        EnqueueWriteBuffer(cq, src0_dram_buffer, input_vec, false);
    }

    {
        std::vector<uint32_t> input_vec;
        for (int i = 0; i < 0x400; i++) input_vec.push_back(25);
        EnqueueWriteBuffer(cq, src1_dram_buffer, input_vec, false);
    } 

    //printf("0x%x, 0x%x, 0x%x\n", src0_dram_buffer->address(), src1_dram_buffer->address(), dst_dram_buffer->address());
    //printf("%zu %zu\n", src0_dram_buffer->noc_coordinates().x, src0_dram_buffer->noc_coordinates().y); // 1 0
    //printf("%zu %zu\n", src1_dram_buffer->noc_coordinates().x, src1_dram_buffer->noc_coordinates().y); // 1 0
    //printf("%zu %zu\n", dst_dram_buffer->noc_coordinates().x, dst_dram_buffer->noc_coordinates().y);   // 1 0
    //printf("%zu %zu\n", l1_buffer->noc_coordinates().x, l1_buffer->noc_coordinates().y);               // 6 5

    // build kernel
    Program program = CreateProgram();
    CoreRange core({0, 0}, {0, 0});
    KernelHandle kernel_id = CreateKernel(program, "./kernel.cpp", core, DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    const std::vector<uint32_t> runtime_args {
        l1_buffer->address(),
        src0_dram_buffer->address(),
        src1_dram_buffer->address(),
        dst_dram_buffer->address(),
    };

    SetRuntimeArgs(program, kernel_id, core, runtime_args);

    // run kernel
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // read output
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);
    printf("result: %d\n", result_vec[0]);

    assert(CloseDevice(device));
}
