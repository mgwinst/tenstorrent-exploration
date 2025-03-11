#include <debug/dprint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dram_buffer_A_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_buffer_B_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t MtKt = get_arg_val<uint32_t>(5);
    uint32_t KtNt = get_arg_val<uint32_t>(6);
    uint32_t batch = get_arg_val<uint32_t>(7);
    uint32_t broadcast_B = get_arg_val<uint32_t>(8);  // if 1 we broadcast B to batch

    DPRINT << "Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt << " MtKt=" << MtKt << "KtNt=" << KtNt << ENDL();
    DPRINT << "dram_buffer_A: " << HEX() << dram_buffer_A_addr << " dram_buffer_B: " << HEX() << dram_buffer_B_addr << ENDL();
    DPRINT << "batch= " << batch << ENDL();

    constexpr uint32_t cb_id_A = 0;
    constexpr uint32_t cb_id_B = 1;

    constexpr uint32_t single_tile = 0;
    constexpr bool is_dram = 1;
    const DataFormat data_format = get_dataformat(0);
    

    const uint32_t bytes_per_tile_CB_A = get_tile_size(cb_id_A);
    const uint32_t bytes_per_tile_CB_B = get_tile_size(cb_id_B);

    uint32_t itileA_batch = 0;
    uint32_t itileB_batch = 0;

    const InterleavedAddrGenFast<is_dram> sA = {
        .bank_base_address = dram_buffer_A_addr,
        .page_size = bytes_per_tile_CB_A,
        .data_format = data_format
    };

    const InterleavedAddrGenFast<is_dram> sB = {
        .bank_base_address = dram_buffer_B_addr,
        .page_size = bytes_per_tile_CB_B,
        .data_format = data_format
    };

    for (uint32_t nb = 0; nb < batch; nb++) {
        uint32_t itileA = itileA_batch;
        for (uint32_t mt = 0; mt < Mt; mt++) {
            uint32_t itileB = itileB_batch;
            for (uint32_t nt = 0; nt < Nt; nt++) {
                for (uint32_t kt = 0; kt < Kt; kt++) {

                    // read A's tile at (mt,kt)
                    {
                        cb_reserve_back(cb_id_A, single_tile);
                        uint32_t l1_write_addr_A = get_write_ptr(cb_id_A);
                        noc_async_read_tile(itileA, sA, l1_write_addr_A);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_A, single_tile);
                    }

                    // read B's tile at (kt, nt)
                    {
                        cb_reserve_back(cb_id_B, single_tile);
                        uint32_t l1_write_addr_B = get_write_ptr(cb_id_B);
                        noc_async_read_tile(itileB, sB, l1_write_addr_B);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_B, single_tile);
                    }

                    itileA += 1; // A is MK
                    itileB += Nt; // B is KN so k++ is striding by Nt
                }
                itileB -= KtNt;
                itileB += 1;
                itileA -= Kt;
            }
            itileA += Kt;
        }
        itileA_batch += MtKt;
        if (broadcast_B == 0) {
            itileB_batch += KtNt;
        }   
    }
}


