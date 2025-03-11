#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/test_tiles.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/tilize_untilize.hpp>
#include <tt-metalium/device_impl.hpp>

void golden_matmul(std::vector<bfloat16>& a, std::vector<bfloat16>& b, std::vector<bfloat16>& output,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    float c_f;
    float float_tmp;
    // std::vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j+ (i * N);
            idx_a = i * K;
            idx_b = j;
            c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
                c_f += float_tmp;
                idx_a += 1;
                idx_b += N;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}

void matmul_single_core(std::vector<bfloat16>& a, std::vector<bfloat16>& b, std::vector<bfloat16>& c,
                        bool broadcast_batch, uint32_t M, uint32_t N, uint32_t K, uint32_t B, IDevice* device) {

    CommandQueue& cq = device->command_queue();
    Program program{};
    CoreRange core({0,0}, {0,0});

    uint32_t Mt = M / tt::constants::TILE_HEIGHT;
    uint32_t Kt = K / tt::constants::TILE_WIDTH;
    uint32_t Nt = N / tt::constants::TILE_WIDTH;

    uint32_t single_tile_size = 2 * 32 * 32; // 32 * 32 bfloat16 elements, each 2 bytes so 1024 * 2 = 2048 bytes

    // create DRAM buffers
    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // (2048 * 20 * 20 == 819,200 bytes)
    uint32_t dram_buffer_B_size = single_tile_size * Kt * Nt;
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;

    tt::tt_metal::InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = dram_buffer_A_size,
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM
    };

    tt::tt_metal::InterleavedBufferConfig dram_config_B{
        .device = device,
        .size = dram_buffer_B_size,
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM
    };
    
    tt::tt_metal::InterleavedBufferConfig dram_config_C{
        .device = device,
        .size = dram_buffer_C_size,
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_A = CreateBuffer(dram_config_A);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_B = CreateBuffer(dram_config_B);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer_C = CreateBuffer(dram_config_C);

    // create circular buffers for communication between read, compute and write engines
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t num_input_tiles = 2;
    uint32_t num_output_tiles = 2;

    uint32_t input_cb_A_index = tt::CBIndex::c_0; 
    CircularBufferConfig cb_A_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{input_cb_A_index, cb_data_format}})
        .set_page_size(input_cb_A_index, single_tile_size);

    uint32_t input_cb_B_index = tt::CBIndex::c_1;
    CircularBufferConfig cb_B_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{input_cb_B_index, cb_data_format}})
        .set_page_size(input_cb_B_index, single_tile_size);

    uint32_t output_cb_C_index = tt::CBIndex::c_16;
    CircularBufferConfig cb_C_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_C_index, cb_data_format}})
        .set_page_size(output_cb_C_index, single_tile_size);

    CBHandle cb_A = CreateCircularBuffer(program, core, cb_A_config);
    CBHandle cb_B = CreateCircularBuffer(program, core, cb_B_config);
    CBHandle cb_C = CreateCircularBuffer(program, core, cb_C_config);

    // create kernels
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "./kernels/dataflow/reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "./kernels/dataflow/writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}       
    );

    std::vector<uint32_t> compute_kernel_args {B, Mt, Kt, Nt};
    KernelHandle matmul_single_core_kernel_id = CreateKernel(
        program,
        "./kernels/compute/single_core_matmul_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args}
    );

    // runtime args for reader/writer kernels
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {dram_buffer_A->address(), dram_buffer_B->address(), Mt, Kt, Nt, Mt*Kt, Kt*Nt, B, uint32_t(broadcast_batch ? 1 : 0)}
    );
    
    SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        {dram_buffer_C->address(), 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
    );

    EnqueueWriteBuffer(cq, dram_buffer_A, a.data(), false);
    EnqueueWriteBuffer(cq, dram_buffer_B, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dram_buffer_C, c.data(), true);
    Finish(cq);
}

int main() {

    // setup device
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    constexpr uint32_t M = 640;
    constexpr uint32_t N = 640;
    constexpr uint32_t K = 640;
    constexpr uint32_t B = 1; 

    uint32_t Mt = M / tt::constants::TILE_HEIGHT; // 640 / 32 = 20
    uint32_t Kt = K / tt::constants::TILE_WIDTH;  // 640 / 32 = 20
    uint32_t Nt = N / tt::constants::TILE_WIDTH;  // 640 / 32 = 20

    constexpr uint32_t single_tile_size = 2 * (32 * 32); // 2048 bytes

    // number of tiles of bfloat16 
    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // (2048 * 20 * 20 == 819,200 bytes)
    uint32_t dram_buffer_B_size = single_tile_size * Kt * Nt;
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;

    // input data vectors
    std::vector<bfloat16> input_vec1 = create_random_vector_of_bfloat16_native(dram_buffer_A_size, 1, 123);
    std::vector<bfloat16> input_vec2 = create_random_vector_of_bfloat16_native(dram_buffer_A_size, 1, 12522);

    // host output vector
    std::vector<bfloat16> output_vec(dram_buffer_C_size/sizeof(bfloat16), 0);

    // CPU Matmul
    std::vector<bfloat16> golden_vec(M * N, 0);
    golden_matmul(input_vec1, input_vec2, golden_vec, M, N, K, B);
    std::cout << "CPU matmul: " << golden_vec[0] << '\n';
    
    tilize(input_vec1, M, K);
    tilize(input_vec2, K, N);

    matmul_single_core(input_vec1, input_vec2, output_vec, 0, M, N, K, B, device);
    
    untilize(output_vec, M, N);
    
    std::cout << "Device matmul: " << output_vec[0] << '\n';   





    assert(CloseDevice(device));

}

