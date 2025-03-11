#include <debug/dprint.h>
#include <compute_kernel_api/matmul.h>
#include <compute_kernel_api/tile_move_copy.h>
#include <compute_kernel_api/reg_api.h>

namespace NAMESPACE {
void MAIN {
    // DPRINT << "hello from compute kernel\n"; 

    constexpr int single_tile = 1;

    int dst_tile_index = 0;
    int in_block_index = 0;

    uint32_t B = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);

    mm_init();

    for (uint32_t batch = 0; batch < B; batch++) {
        for (uint32_t mt_C = 0; mt_C < Mt; mt_C++) {
            for (uint32_t nt_C = 0; nt_C < Nt; nt_C++) {

                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    // wait for 1 tile to be in both CB A B
                    cb_wait_front(tt::CBIndex::c_0, single_tile);
                    cb_wait_front(tt::CBIndex::c_1, single_tile);

                    // pull tile from both CB A and B
                    matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, false);

                    // pop from CB's A B
                    cb_pop_front(tt::CBIndex::c_0, single_tile);
                    cb_pop_front(tt::CBIndex::c_1, single_tile);
                }

                // block until tile is free in output CB
                cb_reserve_back(tt::CBIndex::c_16, single_tile);
                // pack tile from DST register into output CB
                pack_tile(0, tt::CBIndex::c_16);
                cb_push_back(tt::CBIndex::c_16, single_tile);

                release_dst();
            }
        }
    }
}
} // namespace NAMESPACE