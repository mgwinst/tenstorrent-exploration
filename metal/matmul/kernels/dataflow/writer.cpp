#include "dataflow_api.h"
#include <debug/dprint.h>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t batch = get_arg_val<uint32_t>(7);

    constexpr bool is_dram = 1;

    constexpr int single_tile = 1;
    constexpr uint32_t cb_id_C = 16;
    const uint32_t bytes_per_tile = get_tile_size(cb_id_C);
    uint32_t itileC = 0;
    const DataFormat data_format = get_dataformat(cb_id_C);

    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = dst_addr, .page_size = bytes_per_tile, .data_format = data_format
    };

    // C is MN so we iterate in tile RM order
    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {      // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) {  // output tile index of C
                // bmm will generate C's tiles C=A*B, MN=MK*KN, in row major order, we just read them from CB and write
                // out to DRAM
                cb_wait_front(cb_id_C, single_tile);
                uint32_t l1_read_addr = get_read_ptr(cb_id_C);
                noc_async_write_tile(itileC, s, l1_read_addr);
                noc_async_write_barrier();  // This will wait until the write is done. As an alternative,
                                            // noc_async_write_flushed() can be faster because it waits
                                            // until the write request is sent. In that case, you have to
                                            // use noc_async_write_barrier() at least once at the end of
                                            // data movement kernel to make sure all writes are done.
                cb_pop_front(cb_id_C, single_tile);
                // DPRINT << 'W' << 'C' << itileC << ' ' << 'a' << dst_addr << ENDL();
                // DPRINT << itileC << ' ' << uint32_t(dst_noc_addr) << ENDL();
                itileC++;
            }
        }
    }
}