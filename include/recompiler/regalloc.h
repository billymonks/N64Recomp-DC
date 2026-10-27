#ifndef __RECOMP_REGALLOC_H__
#define __RECOMP_REGALLOC_H__

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace N64Recomp {
    // Masks have bit n set for GPR n (bit 0 unused).
    using GprMask = uint32_t;

    // How one function's GPRs are emitted (codegen.regcache / codegen.abi_poison).
    struct FunctionRegPlan {
        // GPRs are C locals, synced with the context at the points below.
        bool locals = false;
        // Context-only output with the ABI's dead registers overwritten after calls and at entry.
        bool poison = false;
        // abi_trace: the same registers tainted instead (TRACE_SET), at the function's address + 1 for its
        // entry and at the call's address after a call.
        bool trace = false;
        uint32_t vram = 0;
        // Registers that are locals. The rest (those live across a call) stay in the context, where they need no
        // syncing and the compiler doesn't have to keep them in callee-saved registers or spill them.
        GprMask cached = 0;
        GprMask declared = 0;
        GprMask entry_load = 0;
        struct CallSync {
            GprMask writeback = 0;
            GprMask reload = 0;
            // abi_poison: registers overwritten after the call.
            GprMask poison = 0;
            // The callee is runtime code (reimplemented, stubbed or a native hook), which abi_trace can't follow.
            bool runtime = false;
            // regcache = "gpr+fpr": FPR webs written back before the call and reloaded after it.
            std::vector<int> fpr_writeback;
            std::vector<int> fpr_reload;
        };
        // regcache = "gpr+fpr": each web of an even FPR (its definitions joined through the uses they reach) that is
        // a C local, named f<reg>_<index>: an fpr32 for singles and raw words, an fpr for doubles and raw doublewords.
        // A web is either only defined in the function or only loaded from the context (at entry, after a call), so
        // a local is never written back before it holds a value. Any call may read or write any FPR: a web that may
        // be newer than the context is written back before calls and returns.
        struct FprWeb {
            int reg;
            bool is64;
            // Used in floating-point operations: synced as float or double. Otherwise it only moves raw bits, synced
            // as words, and lwc1/swc1 stay integer moves (the SH-4 has displacement addressing for those, not fmov).
            bool arith;
            // A doubleword web whose halves are also written or read as words (the high one through the odd
            // register): synced as the raw doubleword.
            bool halves;
        };
        std::vector<FprWeb> fpr_webs;
        // In a writeback list: only the high half (u32h) of this doubleword web.
        static constexpr int FPR_HIGH_HALF = 1 << 30;
        // The web an instruction's write or read of a register belongs to, by instruction index * 32 + register.
        std::unordered_map<uint32_t, int> fpr_def_web;
        std::unordered_map<uint32_t, int> fpr_use_web;
        std::vector<int> fpr_entry;
        std::unordered_map<size_t, std::vector<int>> fpr_exits;
        std::vector<int> fpr_end;
        // abi_poison: registers overwritten at entry.
        GprMask poison_entry = 0;
        // Keyed by the index of the instruction that makes the call or leaves the function.
        std::unordered_map<size_t, CallSync> calls;
        std::unordered_map<size_t, GprMask> exits;
        GprMask end_writeback = 0;
        // A hook on the first instruction runs before the entry loads.
        bool hoist_entry_hook = false;
        // codegen.skip_saves: stack saves and restores of callee-saved registers and ra, by instruction index, done
        // in C. form 'n': nothing (the function never changes the register otherwise); 'c': the save is dropped and
        // the restore reloads the local from the context, which still holds the entry value there; 'l': the save
        // copies the register to the local save_r<reg> and the restore copies it back.
        struct SaveOp {
            int reg;
            char form;
            bool restore;
        };
        std::unordered_map<size_t, SaveOp> save_ops;
        GprMask save_locals = 0;
        // codegen.inline_leaves: a small function without calls, declared RECOMP_LEAF_FUNC so the compiler may
        // inline it into callers in the same file.
        bool leaf_inline = false;
    };

    // What the register analysis found about a function, for its callers and the report.
    struct FunctionRegSummary {
        bool analyzed = false;
        std::string failure;
        // Restores every callee-saved register it writes and leaves sp as it found it.
        bool preserves = true;
        std::string preserve_failure;
        // Caller-saved registers read before being written (arguments excepted).
        GprMask temps_at_entry = 0;
        // Caller-saved registers read after some call without being written in between (v0, v1 excepted).
        GprMask temps_after_call = 0;
        GprMask live_in = 0;
        size_t calls = 0;
        // Direct callees (function indices), and per call site the caller-saved registers read after it
        // (callee -1: an indirect call).
        std::vector<size_t> callees;
        std::vector<std::pair<int, GprMask>> after_call_sites;
    };

    class Context;
    // Fills context.reg_summaries and context.regcache_extra_sync, and writes codegen.abi_report if set.
    bool analyze_registers(Context& context);
}

#endif
