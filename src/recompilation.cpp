#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <cassert>
#include <climits>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <optional>

#include "rabbitizer.hpp"
#include "fmt/format.h"
#include "fmt/ostream.h"

#include "recompiler/context.h"
#include "analysis.h"
#include "recompiler/operations.h"
#include "recompiler/generator.h"

enum class JalResolutionResult {
    NoMatch,
    Match,
    CreateStatic,
    Ambiguous,
    Error
};

JalResolutionResult resolve_jal(const N64Recomp::Context& context, size_t cur_section_index, uint32_t target_func_vram, size_t& matched_function_index) {
    // Skip resolution if all function calls should use lookup and just return Ambiguous.
    if (context.use_lookup_for_all_function_calls) {
        return JalResolutionResult::Ambiguous;
    }

    // Look for symbols with the target vram address
    const N64Recomp::Section& cur_section = context.sections[cur_section_index];
    const auto matching_funcs_find = context.functions_by_vram.find(target_func_vram);
    uint32_t section_vram_start = cur_section.ram_addr;
    uint32_t section_vram_end = cur_section.ram_addr + cur_section.size;
    bool in_current_section = target_func_vram >= section_vram_start && target_func_vram < section_vram_end;
    bool exact_match_found = false;

    // Use a thread local to prevent reallocation across runs and to allow multi-threading in the future.
    thread_local std::vector<size_t> matched_funcs{};
    matched_funcs.clear();

    // Evaluate any functions with the target address to see if they're potential candidates for JAL resolution.
    if (matching_funcs_find != context.functions_by_vram.end()) {
        for (size_t target_func_index : matching_funcs_find->second) {
            const auto& target_func = context.functions[target_func_index];

            // Zero-sized symbol handling. unless there's only one matching target.
            if (target_func.words.empty()) {
                if (!N64Recomp::is_manual_patch_symbol(target_func.vram)) {
                    continue;
                }
            }

            // Immediately accept a function in the same section as this one, since it must also be loaded if the current function is.
            if (target_func.section_index == cur_section_index) {
                exact_match_found = true;
                matched_funcs.clear();
                matched_funcs.push_back(target_func_index);
                break;
            }

            // If the function's section isn't relocatable, add the function as a candidate.
            const auto& target_func_section = context.sections[target_func.section_index];
            if (!target_func_section.relocatable) {
                matched_funcs.push_back(target_func_index);
            }
        }
    }

    // If the target vram is in the current section, only allow exact matches.
    if (in_current_section) {
        // If an exact match was found, use it.
        if (exact_match_found) {
            matched_function_index = matched_funcs[0];
            return JalResolutionResult::Match;
        }
        // Otherwise, create a static function at the target address.
        else {
            return JalResolutionResult::CreateStatic;
        }
    }
    // Otherwise, disambiguate based on the matches found.
    else {
        // If there were no matches then JAL resolution has failed.
        // A static can't be created as the target section is unknown.
        if (matched_funcs.size() == 0) {
            return JalResolutionResult::NoMatch;
        }
        // If there was an exact match, use it.
        else if (matched_funcs.size() == 1) {
            matched_function_index = matched_funcs[0];
            return JalResolutionResult::Match;
        }
        // If there's more than one match, use an indirect jump to resolve the function at runtime.
        else {
            return JalResolutionResult::Ambiguous;
        }
    }

    // This should never be hit, so return an error.
    return JalResolutionResult::Error;
}

using InstrId = rabbitizer::InstrId::UniqueId;
using Cop0Reg = rabbitizer::Registers::Cpu::Cop0;

std::string_view ctx_gpr_prefix(int reg) {
    if (reg != 0) {
        return "ctx->r";
    }
    return "";
}

namespace {
    std::string trace_statement(const rabbitizer::InstructionCpu& instr, InstrId id, uint32_t vram);
}

template <typename GeneratorType>
bool process_instruction(GeneratorType& generator, const N64Recomp::Context& context, const N64Recomp::Function& func, size_t func_index, const N64Recomp::FunctionStats& stats, const std::unordered_set<uint32_t>& jtbl_lw_instructions, size_t instr_index, const std::vector<rabbitizer::InstructionCpu>& instructions, std::ostream& output_file, bool indent, bool emit_link_branch, int link_branch_index, size_t reloc_index, bool& needs_link_branch, bool& is_branch_likely, bool tag_reference_relocs, std::span<std::vector<uint32_t>> static_funcs_out) {
    using namespace N64Recomp;

    const auto& section = context.sections[func.section_index];
    const auto& instr = instructions[instr_index];
    needs_link_branch = false;
    is_branch_likely = false;
    uint32_t instr_vram = instr.getVram();
    InstrId instr_id = instr.getUniqueId();

    auto print_indent = [&]() {
        fmt::print(output_file, "    ");
    };

    auto hook_find = func.function_hooks.find(instr_index);
    if (hook_find != func.function_hooks.end()) {
        fmt::print(output_file, "    {}\n", hook_find->second);
        if (indent) {
            print_indent();
        }
    }

    // Output a comment with the original instruction
    print_indent();
    if (instr.isBranch() || instr_id == InstrId::cpu_j) {
        generator.emit_comment(fmt::format("0x{:08X}: {}", instr_vram, instr.disassemble(0, fmt::format("L_{:08X}", (uint32_t)instr.getBranchVramGeneric()))));
    } else if (instr_id == InstrId::cpu_jal) {
        generator.emit_comment(fmt::format("0x{:08X}: {}", instr_vram, instr.disassemble(0, fmt::format("0x{:08X}", (uint32_t)instr.getBranchVramGeneric()))));
    } else {
        generator.emit_comment(fmt::format("0x{:08X}: {}", instr_vram, instr.disassemble(0)));
    }

    // Replace loads for jump table entries into addiu. This leaves the jump table entry's address in the output register
    // instead of the entry's value, which can then be used to determine the offset from the start of the jump table.
    if (jtbl_lw_instructions.contains(instr_vram)) {
        assert(instr_id == InstrId::cpu_lw);
        instr_id = InstrId::cpu_addiu;
    }

    if (context.codegen.abi_trace) {
        std::string trace = trace_statement(instr, instr_id, instr_vram);
        if (!trace.empty()) {
            print_indent();
            fmt::print(output_file, "{}\n", trace);
        }
    }

    if (generator.emit_planned_save(instr_index)) {
        return true;
    }

    if (context.gpr_32bit) {
        static const std::unordered_set<InstrId> gpr64_instrs {
            InstrId::cpu_dadd, InstrId::cpu_daddu, InstrId::cpu_daddi, InstrId::cpu_daddiu,
            InstrId::cpu_dsub, InstrId::cpu_dsubu,
            InstrId::cpu_dsll, InstrId::cpu_dsrl, InstrId::cpu_dsra,
            InstrId::cpu_dsll32, InstrId::cpu_dsrl32, InstrId::cpu_dsra32,
            InstrId::cpu_dsllv, InstrId::cpu_dsrlv, InstrId::cpu_dsrav,
            InstrId::cpu_dmult, InstrId::cpu_dmultu, InstrId::cpu_ddiv, InstrId::cpu_ddivu,
            InstrId::cpu_ld, InstrId::cpu_ldl, InstrId::cpu_ldr,
            InstrId::cpu_sd, InstrId::cpu_sdl, InstrId::cpu_sdr,
            InstrId::cpu_dmfc1, InstrId::cpu_dmtc1,
        };
        if (gpr64_instrs.contains(instr_id)) {
            fmt::print(stderr, "Instruction at 0x{:08X} in {} needs 64-bit GPRs, which gpr_32bit mode doesn't support: {}\n",
                instr_vram, func.name, instr.disassemble(0));
            return false;
        }
    }

    N64Recomp::RelocType reloc_type = N64Recomp::RelocType::R_MIPS_NONE;
    bool has_reloc = false;
    uint32_t reloc_section = 0;
    uint32_t reloc_target_section_offset = 0;
    size_t reloc_reference_symbol = (size_t)-1;

    uint32_t func_vram_end = func.vram + func.words.size() * sizeof(func.words[0]);

    uint16_t imm = instr.Get_immediate();

    // Check if this instruction has a reloc.
    if (section.relocs.size() > 0 && section.relocs[reloc_index].address == instr_vram) {
        has_reloc = true;
        // Get the reloc data for this instruction
        const auto& reloc = section.relocs[reloc_index];
        reloc_section = reloc.target_section;

        // Check if the relocation references a relocatable section.
        bool target_relocatable = false;
        if (!reloc.reference_symbol && reloc_section != N64Recomp::SectionAbsolute) {
            const auto& target_section = context.sections[reloc_section];
            target_relocatable = target_section.relocatable;
        }

        // Only process this relocation if the target section is relocatable or if this relocation targets a reference symbol.
        if (target_relocatable || reloc.reference_symbol) {
            // Record the reloc's data.
            reloc_type = reloc.type;
            reloc_target_section_offset = reloc.target_section_offset;
            // Ignore all relocs that aren't MIPS_HI16, MIPS_LO16 or MIPS_26.
            if (reloc_type == N64Recomp::RelocType::R_MIPS_HI16 || reloc_type == N64Recomp::RelocType::R_MIPS_LO16 || reloc_type == N64Recomp::RelocType::R_MIPS_26) {
                if (reloc.reference_symbol) {
                    reloc_reference_symbol = reloc.symbol_index;
                    // Don't try to relocate special section symbols.
                    if (context.is_regular_reference_section(reloc.target_section) || reloc_section == N64Recomp::SectionAbsolute) {
                        // TODO this may not be needed anymore as HI16/LO16 relocs to non-relocatable sections is handled directly in elf parsing.
                        bool ref_section_relocatable = context.is_reference_section_relocatable(reloc.target_section);
                        // Resolve HI16 and LO16 reference symbol relocs to non-relocatable sections by patching the instruction immediate.
                        if (!ref_section_relocatable && (reloc_type == N64Recomp::RelocType::R_MIPS_HI16 || reloc_type == N64Recomp::RelocType::R_MIPS_LO16)) {
                            // The reloc has been processed, so set it to none to prevent it getting processed a second time during instruction code generation.
                            reloc_type = N64Recomp::RelocType::R_MIPS_NONE;
                            reloc_reference_symbol = (size_t)-1;
                        }
                    }
                }
            }

            // Repoint bss relocations at their non-bss counterpart section.
            auto find_bss_it = context.bss_section_to_section.find(reloc_section);
            if (find_bss_it != context.bss_section_to_section.end()) {
                reloc_section = find_bss_it->second;
            }
        }
    }

    auto process_delay_slot = [&](bool use_indent) {
        if (instr_index < instructions.size() - 1) {
            bool dummy_needs_link_branch;
            bool dummy_is_branch_likely;
            size_t next_reloc_index = reloc_index;
            uint32_t next_vram = instr_vram + 4;
            if (reloc_index + 1 < section.relocs.size() && next_vram > section.relocs[reloc_index].address) {
                next_reloc_index++;
            }
            if (!process_instruction(generator, context, func, func_index, stats, jtbl_lw_instructions, instr_index + 1, instructions, output_file, use_indent, false, link_branch_index, next_reloc_index, dummy_needs_link_branch, dummy_is_branch_likely, tag_reference_relocs, static_funcs_out)) {
                return false;
            }
        }
        return true;
    };

    auto print_link_branch = [&]() {
        if (needs_link_branch) {
            print_indent();
            generator.emit_goto(fmt::format("after_{}", link_branch_index));
        }
    };

    auto print_return_with_delay_slot = [&]() {
        if (!process_delay_slot(false)) {
            return false;
        }
        print_indent();
        generator.emit_exit_sync(instr_index);
        generator.emit_return(context, func_index);
        print_link_branch();
        return true;
    };

    auto print_goto_with_delay_slot = [&](const std::string& target) {
        if (!process_delay_slot(false)) {
            return false;
        }
        print_indent();
        generator.emit_goto(target);
        print_link_branch();
        return true;
    };

    auto print_func_call_by_register = [&](int reg) {
        if (!process_delay_slot(false)) {
            return false;
        }
        print_indent();
        generator.emit_call_sync_before(instr_index);
        generator.emit_function_call_by_register(reg);
        generator.emit_call_sync_after(instr_index);
        print_link_branch();
        return true;
    };

    auto print_func_call_by_address = [&generator, reloc_target_section_offset, has_reloc, reloc_section, reloc_reference_symbol, reloc_type, &context, &func, &static_funcs_out, &needs_link_branch, &print_indent, &process_delay_slot, &print_link_branch, instr_index]
        (uint32_t target_func_vram, bool tail_call = false, bool indent = false)
    {
        bool call_by_lookup = false;
        bool call_by_name = false;
        // Event symbol, emit a call to the runtime to trigger this event.
        if (reloc_section == N64Recomp::SectionEvent) {
            needs_link_branch = !tail_call;
            if (indent) {
                print_indent();
            }
            if (!process_delay_slot(false)) {
                return false;
            }
            print_indent();
            generator.emit_call_sync_before(instr_index);
            generator.emit_trigger_event((uint32_t)reloc_reference_symbol);
            generator.emit_call_sync_after(instr_index);
            print_link_branch();
        }
        // Normal symbol or reference symbol, 
        else {
            std::string jal_target_name{};
            size_t matched_func_index = (size_t)-1;
            if (reloc_reference_symbol != (size_t)-1) {
                if (reloc_type != N64Recomp::RelocType::R_MIPS_26) {
                    fmt::print(stderr, "Unsupported reloc type {} on jal instruction in {}\n", (int)reloc_type, func.name);
                    return false;
                }

                if (!context.skip_validating_reference_symbols) {
                    const auto& ref_symbol = context.get_reference_symbol(reloc_section, reloc_reference_symbol);
                    if (ref_symbol.section_offset != reloc_target_section_offset) {
                        fmt::print(stderr, "Function {} uses a MIPS_R_26 addend, which is not supported yet\n", func.name);
                        return false;
                    }
                }
            }
            else {
                uint32_t target_section = func.section_index;
                // If this instruction has a reloc and the target section is a normal section, use the section of the reloc when searching for a matching target function. 
                if (has_reloc && reloc_section < 65500) {
                    target_section = reloc_section;
                }
                JalResolutionResult jal_result = resolve_jal(context, target_section, target_func_vram, matched_func_index);

                switch (jal_result) {
                    case JalResolutionResult::NoMatch:
                        fmt::print(stderr, "No function found for jal target: 0x{:08X}\n", target_func_vram);
                        return false;
                    case JalResolutionResult::Match:
                        jal_target_name = context.functions[matched_func_index].name;
                        break;
                    case JalResolutionResult::CreateStatic:
                        // Create a static function add it to the static function list for this section.
                        jal_target_name = fmt::format("static_{}_{:08X}", func.section_index, target_func_vram);
                        static_funcs_out[func.section_index].push_back(target_func_vram);
                        call_by_name = true;
                        break;
                    case JalResolutionResult::Ambiguous:
                        // Print a warning if lookup isn't forced for all non-reloc function calls.
                        if (!context.use_lookup_for_all_function_calls) {
                            fmt::print(stderr, "[Info] Ambiguous jal target 0x{:08X} in function {}, falling back to function lookup\n", target_func_vram, func.name);
                        }
                        // Relocation isn't necessary for jumps inside a relocatable section, as this code path will never run if the target vram
                        // is in the current function's section (see the branch for `in_current_section` above).
                        // If a game ever needs to jump between multiple relocatable sections, relocation will be necessary here.
                        call_by_lookup = true;
                        break;
                    case JalResolutionResult::Error:
                        fmt::print(stderr, "Internal error when resolving jal to address 0x{:08X} in function {}. Please report this issue.\n", target_func_vram, func.name);
                        return false;
                }
            }
            needs_link_branch = !tail_call;
            if (indent) {
                print_indent();
            }
            if (!process_delay_slot(false)) {
                return false;
            }
            print_indent();
            generator.emit_call_sync_before(instr_index);
            if (reloc_reference_symbol != (size_t)-1) {
                generator.emit_function_call_reference_symbol(context, reloc_section, reloc_reference_symbol, reloc_target_section_offset);
            }
            else if (call_by_lookup) {
                generator.emit_function_call_lookup(target_func_vram);
            }
            else if (call_by_name) {
                generator.emit_named_function_call(jal_target_name);
            }
            else {
                generator.emit_function_call(context, matched_func_index);
            }
            generator.emit_call_sync_after(instr_index);
            print_link_branch();
        }
        return true;
    };

    auto print_branch = [&](uint32_t branch_target) {
        // If the branch target is outside the current function, check if it can be treated as a tail call.
        if (branch_target < func.vram || branch_target >= func_vram_end) {
            // If the branch target is the start of some known function, this can be handled as a tail call.
            // FIXME: how to deal with static functions?
            if (context.functions_by_vram.find(branch_target) != context.functions_by_vram.end()) {
                fmt::print("Tail call in {} to 0x{:08X}\n", func.name, branch_target);
                if (!print_func_call_by_address(branch_target, true, true)) {
                    return false;
                }
                print_indent();
                generator.emit_exit_sync(instr_index);
                generator.emit_return(context, func_index);
                // TODO check if this branch close should exist.
                // print_indent();
                // generator.emit_branch_close();
                return true;
            }

            fmt::print(stderr, "[Warn] Function {} is branching outside of the function (to 0x{:08X})\n", func.name, branch_target);
        }

        if (!process_delay_slot(true)) {
            return false;
        }

        print_indent();
        print_indent();
        generator.emit_goto(fmt::format("L_{:08X}", branch_target));
        // TODO check if this link branch ever exists.
        if (needs_link_branch) {
            print_indent();
            print_indent();
            generator.emit_goto(fmt::format("after_{}", link_branch_index));
        }
        return true;
    };

    if (indent) {
        print_indent();
    }

    int rd = (int)instr.GetO32_rd();
    int rs = (int)instr.GetO32_rs();
    int rt = (int)instr.GetO32_rt();
    int sa = (int)instr.Get_sa();

    int fd = (int)instr.GetO32_fd();
    int fs = (int)instr.GetO32_fs();
    int ft = (int)instr.GetO32_ft();

    int cop1_cs = (int)instr.Get_cop1cs();

    bool handled = true;

    switch (instr_id) {
    case InstrId::cpu_nop:
        fmt::print(output_file, "\n");
        break;
    // Cop0 (Limited functionality)
    case InstrId::cpu_mfc0:
        {
            Cop0Reg reg = instr.Get_cop0d();
            switch (reg) {
            case Cop0Reg::COP0_Status:
                print_indent();
                generator.emit_cop0_status_read(rt);
                break;
            default:
                fmt::print(stderr, "Unhandled cop0 register in mfc0: {}\n", (int)reg);
                return false;
            }
            break;
        }
    case InstrId::cpu_mtc0:
        {
            Cop0Reg reg = instr.Get_cop0d();
            switch (reg) {
            case Cop0Reg::COP0_Status:
                print_indent();
                generator.emit_cop0_status_write(rt);
                break;
            default:
                fmt::print(stderr, "Unhandled cop0 register in mtc0: {}\n", (int)reg);
                return false;
            }
            break;
        }
    // Arithmetic
    case InstrId::cpu_add:
    case InstrId::cpu_addu:
        {
            // Check if this addu belongs to a jump table load
            auto find_result = std::find_if(stats.jump_tables.begin(), stats.jump_tables.end(),
                [instr_vram](const N64Recomp::JumpTable& jtbl) {
                return jtbl.addu_vram == instr_vram;
            });
            // If so, create a temp to preserve the addend register's value
            if (find_result != stats.jump_tables.end()) {
                const N64Recomp::JumpTable& cur_jtbl = *find_result;
                print_indent();
                generator.emit_jtbl_addend_declaration(cur_jtbl, cur_jtbl.addend_reg);
            }
        }
        break;
    case InstrId::cpu_mult:
    case InstrId::cpu_dmult:
    case InstrId::cpu_multu:
    case InstrId::cpu_dmultu:
    case InstrId::cpu_div:
    case InstrId::cpu_ddiv:
    case InstrId::cpu_divu:
    case InstrId::cpu_ddivu:
        print_indent();
        generator.emit_muldiv(instr_id, rs, rt);
        break;
    // Branches
    case InstrId::cpu_jal:
        if (!print_func_call_by_address(instr.getBranchVramGeneric())) {
            return false;
        }
        break;
    case InstrId::cpu_jalr:
        // jalr can only be handled with $ra as the return address register
        if (rd != (int)rabbitizer::Registers::Cpu::GprO32::GPR_O32_ra) {
            fmt::print(stderr, "Invalid return address reg for jalr: f{}\n", rd);
            return false;
        }
        needs_link_branch = true;
        print_func_call_by_register(rs);
        break;
    case InstrId::cpu_j:
    case InstrId::cpu_b:
        {
            uint32_t branch_target = instr.getBranchVramGeneric();
            if (branch_target == instr_vram) {
                print_indent();
                generator.emit_pause_self();
            }
            // Check if the branch is within this function
            else if (branch_target >= func.vram && branch_target < func_vram_end) {
                print_goto_with_delay_slot(fmt::format("L_{:08X}", branch_target));
            }
            // This may be a tail call in the middle of the control flow due to a previous check
            // For example:
            // ```c
            // void test() {
            //     if (SOME_CONDITION) {
            //         do_a();
            //     } else {
            //         do_b();
            //     }
            // }
            // ```
            // FIXME: how to deal with static functions?
            else if (context.functions_by_vram.find(branch_target) != context.functions_by_vram.end()) {
                fmt::print("[Info] Tail call in {} to 0x{:08X}\n", func.name, branch_target);
                if (!print_func_call_by_address(branch_target, true)) {
                    return false;
                }
                print_indent();
                generator.emit_exit_sync(instr_index);
                generator.emit_return(context, func_index);
            }
            else {
                fmt::print(stderr, "Unhandled branch in {} at 0x{:08X} to 0x{:08X}\n", func.name, instr_vram, branch_target);
                return false;
            }
        }
        break;
    case InstrId::cpu_jr:
        if (rs == (int)rabbitizer::Registers::Cpu::GprO32::GPR_O32_ra) {
            print_return_with_delay_slot();
        } else {
            auto jtbl_find_result = std::find_if(stats.jump_tables.begin(), stats.jump_tables.end(),
                [instr_vram](const N64Recomp::JumpTable& jtbl) {
                    return jtbl.jr_vram == instr_vram;
                });

            if (jtbl_find_result != stats.jump_tables.end()) {
                const N64Recomp::JumpTable& cur_jtbl = *jtbl_find_result;
                if (!process_delay_slot(false)) {
                    return false;
                }
                print_indent();
                generator.emit_switch(context, cur_jtbl, rs);
                for (size_t entry_index = 0; entry_index < cur_jtbl.entries.size(); entry_index++) {
                    print_indent();
                    print_indent();
                    generator.emit_case(entry_index, fmt::format("L_{:08X}", cur_jtbl.entries[entry_index]));
                }
                print_indent();
                print_indent();
                generator.emit_switch_error(instr_vram, cur_jtbl.vram);
                print_indent();
                generator.emit_switch_close();
                break;
            }

            fmt::print("[Info] Indirect tail call in {}\n", func.name);
            print_func_call_by_register(rs);
            print_indent();
            generator.emit_exit_sync(instr_index);
            generator.emit_return(context, func_index);
            break;
        }
        break;
    case InstrId::cpu_syscall:
        print_indent();
        generator.emit_call_sync_before(instr_index);
        generator.emit_syscall(instr_vram);
        // syscalls don't link, so treat it like a tail call
        print_indent();
        generator.emit_exit_sync(instr_index);
        generator.emit_return(context, func_index);
        break;
    case InstrId::cpu_break:
        print_indent();
        generator.emit_do_break(instr_vram);
        break;

    // Cop1 rounding mode
    case InstrId::cpu_ctc1:
        if (cop1_cs != 31) {
            fmt::print(stderr, "Invalid FP control register for ctc1: {}\n", cop1_cs);
            return false;
        }
        print_indent();
        generator.emit_cop1_cs_write(rt);
        break;
    case InstrId::cpu_cfc1:
        if (cop1_cs != 31) {
            fmt::print(stderr, "Invalid FP control register for cfc1: {}\n", cop1_cs);
            return false;
        }
        print_indent();
        generator.emit_cop1_cs_read(rt);
        break;
    default:
        handled = false;
        break;
    }

    InstructionContext instruction_context{};
    instruction_context.rd = rd;
    instruction_context.rs = rs;
    instruction_context.rt = rt;
    instruction_context.sa = sa;
    instruction_context.fd = fd;
    instruction_context.fs = fs;
    instruction_context.ft = ft;
    instruction_context.cop1_cs = cop1_cs;
    instruction_context.imm16 = imm;
    instruction_context.instr_index = instr_index;
    instruction_context.reloc_tag_as_reference = (reloc_reference_symbol != (size_t)-1) && tag_reference_relocs;
    instruction_context.reloc_type = reloc_type;
    instruction_context.reloc_section_index = reloc_section;
    instruction_context.reloc_target_section_offset = reloc_target_section_offset;
    
    auto do_check_fr = [](const GeneratorType& generator, const InstructionContext& ctx, Operand operand) {
        switch (operand) {
            case Operand::Fd:
            case Operand::FdDouble:
            case Operand::FdU32L:
            case Operand::FdU32H:
            case Operand::FdU64:
                generator.emit_check_fr(ctx.fd);
                break;
            case Operand::Fs:
            case Operand::FsDouble:
            case Operand::FsU32L:
            case Operand::FsU32H:
            case Operand::FsU64:
                generator.emit_check_fr(ctx.fs);
                break;
            case Operand::Ft:
            case Operand::FtDouble:
            case Operand::FtU32L:
            case Operand::FtU32H:
            case Operand::FtU64:
                generator.emit_check_fr(ctx.ft);
                break;
            default:
                // No MIPS3 float check needed for non-float operands.
                break;
        }
    };
    
    auto do_check_nan = [](const GeneratorType& generator, const InstructionContext& ctx, Operand operand) {
        switch (operand) {
            case Operand::Fd:
                generator.emit_check_nan(ctx.fd, false);
                break;
            case Operand::Fs:
                generator.emit_check_nan(ctx.fs, false);
                break;
            case Operand::Ft:
                generator.emit_check_nan(ctx.ft, false);
                break;
            case Operand::FdDouble:
                generator.emit_check_nan(ctx.fd, true);
                break;
            case Operand::FsDouble:
                generator.emit_check_nan(ctx.fs, true);
                break;
            case Operand::FtDouble:
                generator.emit_check_nan(ctx.ft, true);
                break;
            default:
                // No NaN checks needed for non-float operands.
                break;
        }
    };

    auto find_binary_it = binary_ops.find(instr_id);
    if (find_binary_it != binary_ops.end()) {
        print_indent();
        const BinaryOp& op = find_binary_it->second;
        
        if (op.check_fr) {
            do_check_fr(generator, instruction_context, op.output);
            do_check_fr(generator, instruction_context, op.operands.operands[0]);
            do_check_fr(generator, instruction_context, op.operands.operands[1]);
        }

        if (op.check_nan) {
            do_check_nan(generator, instruction_context, op.operands.operands[0]);
            do_check_nan(generator, instruction_context, op.operands.operands[1]);
            fmt::print(output_file, "\n");
            print_indent();
        }

        generator.process_binary_op(op, instruction_context);
        handled = true;
    }

    auto find_unary_it = unary_ops.find(instr_id);
    if (find_unary_it != unary_ops.end()) {
        print_indent();
        const UnaryOp& op = find_unary_it->second;
        
        if (op.check_fr) {
            do_check_fr(generator, instruction_context, op.output);
            do_check_fr(generator, instruction_context, op.input);
        }

        if (op.check_nan) {
            do_check_nan(generator, instruction_context, op.input);
            fmt::print(output_file, "\n");
            print_indent();
        }

        generator.process_unary_op(op, instruction_context);
        handled = true;
    }

    auto find_conditional_branch_it = conditional_branch_ops.find(instr_id);
    uint32_t branch_target = (uint32_t)instr.getBranchVramGeneric();
    if (find_conditional_branch_it != conditional_branch_ops.end() && context.codegen.delay_once &&
        !find_conditional_branch_it->second.likely && !find_conditional_branch_it->second.link &&
        branch_target >= func.vram && branch_target < func_vram_end && instr_index + 1 < instructions.size()) {
        // The delay slot runs once, between the condition and the jump; the main loop then skips its own copy of
        // it like a likely branch's (the copy stays for anything jumping straight to the slot).
        print_indent();
        generator.emit_branch_value(find_conditional_branch_it->second, instruction_context);
        if (!process_delay_slot(true)) {
            return false;
        }
        print_indent();
        print_indent();
        generator.emit_branch_on_value(fmt::format("L_{:08X}", branch_target));
        is_branch_likely = true;
        handled = true;
    }
    else if (find_conditional_branch_it != conditional_branch_ops.end()) {
        print_indent();
        // TODO combining the branch condition and branch target into one generator call would allow better optimization in the runtime's JIT generator.
        // This would require splitting into a conditional jump method and conditional function call method.
        generator.emit_branch_condition(find_conditional_branch_it->second, instruction_context);

        print_indent();
        if (find_conditional_branch_it->second.link) {
            if (!print_func_call_by_address(instr.getBranchVramGeneric())) {
                return false;
            }
        }
        else {
            if (!print_branch((uint32_t)instr.getBranchVramGeneric())) {
                return false;
            }
        }

        print_indent();
        generator.emit_branch_close();
        
        is_branch_likely = find_conditional_branch_it->second.likely;
        handled = true;
    }

    auto find_store_it = store_ops.find(instr_id);
    if (find_store_it != store_ops.end()) {
        print_indent();
        const StoreOp& op = find_store_it->second;

        if (op.type == StoreOpType::SDC1) {
            do_check_fr(generator, instruction_context, op.value_input);
        }

        generator.process_store_op(op, instruction_context);
        handled = true;
    }

    if (!handled) {
        fmt::print(stderr, "Unhandled instruction: {}\n", instr.getOpcodeName());
        return false;
    }

    // TODO is this used?
    if (emit_link_branch) {
        print_indent();
        generator.emit_label(fmt::format("after_{}", link_branch_index));
    }

    return true;
}

// Register analysis for codegen.regcache: per-function GPR dataflow over the MIPS code, as the C generator
// emits it (delay slots copied into each path), and the o32 checks the caching protocol relies on.
namespace {
    using N64Recomp::GprMask;

    constexpr GprMask gpr_bit(int reg) { return reg > 0 && reg < 32 ? (1u << reg) : 0; }
    constexpr GprMask mask_range(int first, int last) {
        GprMask m = 0;
        for (int r = first; r <= last; r++) {
            m |= gpr_bit(r);
        }
        return m;
    }
    constexpr GprMask ALL_GPRS = 0xFFFFFFFEu;
    constexpr GprMask ARG_REGS = mask_range(4, 7);
    constexpr GprMask RETURN_REGS = mask_range(2, 3);
    constexpr GprMask SP_REG = gpr_bit(29);
    constexpr GprMask CALLEE_SAVED = mask_range(16, 23) | gpr_bit(28) | gpr_bit(30) | gpr_bit(31);
    // at, v0, v1, a0-a3, t0-t9, k0, k1.
    constexpr GprMask CALLER_SAVED = mask_range(1, 15) | mask_range(24, 27);
    // A call writes back what the callee may read: arguments, sp, and the callee-saved registers its prologue
    // stores, so its stack frame holds the same values as with context-only output.
    constexpr GprMask CALL_USES = ARG_REGS | SP_REG | CALLEE_SAVED;
    constexpr GprMask EXIT_REGS = RETURN_REGS | SP_REG | CALLEE_SAVED;

    struct RegNode {
        enum class Kind : uint8_t { Plain, Call, TailCall, Barrier, Exit };
        Kind kind = Kind::Plain;
        GprMask use = 0;
        GprMask def = 0;
        // Definitions that leave the local equal to the context (skip_saves restores from the context).
        GprMask clean = 0;
        // Calls: what the callee may read, and what it may leave changed in the context.
        GprMask call_use = 0;
        GprMask call_def = 0;
        // Calls: what the callee certainly writes, so the value from before the call is dead after it. Anything
        // else may come through the call unchanged (apply_call_kills).
        GprMask call_kill = 0;
        // Calls: what the callee and everything it calls never read or write, so a local needs no syncing around it.
        GprMask call_keep = 0;
        // Instruction index the plan is keyed by (calls and exits), or the instruction a Plain node executes.
        size_t key = SIZE_MAX;
        size_t instr = SIZE_MAX;
        // Calls: the callee's function index if known, or whether it's an indirect call; a direct call's target.
        int callee = -1;
        bool indirect = false;
        uint32_t target = 0;
        std::vector<int> succs;
    };

    struct RegGraph {
        std::vector<RegNode> nodes;
        std::string failure;
        size_t calls = 0;
    };

    void gpr_effects(const rabbitizer::InstructionCpu& instr, InstrId id, GprMask& use, GprMask& def) {
        using namespace N64Recomp;
        int rd = (int)instr.GetO32_rd();
        int rs = (int)instr.GetO32_rs();
        int rt = (int)instr.GetO32_rt();
        auto reg_of = [&](Operand op) {
            switch (op) {
                case Operand::Rd: return rd;
                case Operand::Rs: return rs;
                case Operand::Rt: return rt;
                default: return 0;
            }
        };
        auto binary = binary_ops.find(id);
        if (binary != binary_ops.end()) {
            def |= gpr_bit(reg_of(binary->second.output));
            use |= gpr_bit(reg_of(binary->second.operands.operands[0])) | gpr_bit(reg_of(binary->second.operands.operands[1]));
            BinaryOpType type = binary->second.type;
            if (type == BinaryOpType::LWL || type == BinaryOpType::LWR || type == BinaryOpType::LDL || type == BinaryOpType::LDR) {
                use |= gpr_bit(reg_of(binary->second.output));
            }
        }
        auto unary = unary_ops.find(id);
        if (unary != unary_ops.end()) {
            def |= gpr_bit(reg_of(unary->second.output));
            use |= gpr_bit(reg_of(unary->second.input));
        }
        auto store = store_ops.find(id);
        if (store != store_ops.end()) {
            use |= gpr_bit(reg_of(store->second.value_input)) | gpr_bit(rs);
        }
        auto branch = conditional_branch_ops.find(id);
        if (branch != conditional_branch_ops.end()) {
            use |= gpr_bit(reg_of(branch->second.operands.operands[0])) | gpr_bit(reg_of(branch->second.operands.operands[1]));
        }
        switch (id) {
            case InstrId::cpu_mult: case InstrId::cpu_multu: case InstrId::cpu_div: case InstrId::cpu_divu:
            case InstrId::cpu_dmult: case InstrId::cpu_dmultu: case InstrId::cpu_ddiv: case InstrId::cpu_ddivu:
                use |= gpr_bit(rs) | gpr_bit(rt);
                break;
            case InstrId::cpu_mfc0: case InstrId::cpu_cfc1:
                def |= gpr_bit(rt);
                break;
            case InstrId::cpu_mtc0: case InstrId::cpu_ctc1:
                use |= gpr_bit(rt);
                break;
            default:
                break;
        }
    }

    // How a direct call is treated: the o32 protocol, or flushing and reloading everything.
    bool call_is_barrier(const N64Recomp::Context& context, const N64Recomp::Function& func, uint32_t instr_vram, uint32_t target, RegGraph& g, int& callee_index) {
        callee_index = -1;
        const auto& section = context.sections[func.section_index];
        uint32_t target_section = func.section_index;
        for (const auto& reloc : section.relocs) {
            if (reloc.address != instr_vram) {
                continue;
            }
            if (reloc.reference_symbol || reloc.target_section == N64Recomp::SectionEvent) {
                return reloc.target_section == N64Recomp::SectionEvent;
            }
            if (reloc.target_section < 65500) {
                target_section = reloc.target_section;
            }
        }
        size_t matched = (size_t)-1;
        switch (resolve_jal(context, target_section, target, matched)) {
            case JalResolutionResult::Match: {
                const auto& callee = context.functions[matched];
                callee_index = (int)matched;
                if (callee.reimplemented || callee.stubbed) {
                    return false;
                }
                if (matched >= context.reg_summaries.size()) {
                    return true;
                }
                const auto& summary = context.reg_summaries[matched];
                return !summary.analyzed || !summary.preserves;
            }
            case JalResolutionResult::Ambiguous:
                return false;
            case JalResolutionResult::CreateStatic:
                return true;
            default:
                g.failure = fmt::format("unresolved call to 0x{:08X}", target);
                return true;
        }
    }

    bool build_reg_graph(const N64Recomp::Context& context, const N64Recomp::Function& func, const std::vector<rabbitizer::InstructionCpu>& instructions,
                         const N64Recomp::FunctionStats& stats, const std::unordered_set<uint32_t>& jtbl_lw, GprMask extra_sync, RegGraph& g) {
        size_t n = instructions.size();
        uint32_t func_vram_end = func.vram + (uint32_t)(n * 4);
        g.nodes.assign(n, RegNode{});
        auto add = [&](RegNode node) {
            g.nodes.push_back(std::move(node));
            return (int)g.nodes.size() - 1;
        };
        RegNode end_node{};
        end_node.kind = RegNode::Kind::Exit;
        end_node.key = SIZE_MAX;
        int end = add(end_node);
        auto seq = [&](size_t i) { return i < n ? (int)i : end; };
        auto index_of = [&](uint32_t vram) -> int {
            if (vram < func.vram || vram >= func_vram_end) {
                return -1;
            }
            return (int)((vram - func.vram) / 4);
        };
        auto plain_effects = [&](size_t i, RegNode& node) {
            InstrId id = instructions[i].getUniqueId();
            if (jtbl_lw.contains(instructions[i].getVram())) {
                id = InstrId::cpu_addiu;
            }
            gpr_effects(instructions[i], id, node.use, node.def);
            node.instr = i;
        };
        auto delay_copy = [&](size_t i, int succ) {
            RegNode node{};
            if (i + 1 < n) {
                plain_effects(i + 1, node);
            }
            node.succs = { succ };
            return add(std::move(node));
        };
        auto call_node = [&](size_t key, bool barrier, GprMask extra_use, RegNode::Kind kind, int succ, int callee = -1, bool indirect = false) {
            RegNode node{};
            node.kind = kind;
            node.key = key;
            node.use = extra_use;
            node.callee = callee;
            node.indirect = indirect;
            node.call_use = barrier ? ALL_GPRS : (CALL_USES | extra_sync);
            node.call_def = barrier ? ALL_GPRS : (CALLER_SAVED);
            if (succ >= 0) {
                node.succs = { succ };
            }
            g.calls++;
            return add(std::move(node));
        };
        auto exit_node = [&](size_t key) {
            RegNode node{};
            node.kind = RegNode::Kind::Exit;
            node.key = key;
            return add(std::move(node));
        };
        auto tail_call = [&](size_t key, bool barrier, GprMask extra_use, int callee = -1, bool indirect = false) {
            int e = exit_node(key);
            return call_node(key, barrier, extra_use, RegNode::Kind::TailCall, e, callee, indirect);
        };
        auto direct_call = [&](size_t i, uint32_t vram, uint32_t target, RegNode::Kind kind, int succ) {
            int callee = -1;
            bool barrier = call_is_barrier(context, func, vram, target, g, callee);
            int id = kind == RegNode::Kind::TailCall ? tail_call(i, barrier, 0, callee) : call_node(i, barrier, 0, kind, succ, callee);
            g.nodes[id].target = target;
            return id;
        };

        for (size_t i = 0; i < n; i++) {
            const auto& instr = instructions[i];
            InstrId id = instr.getUniqueId();
            uint32_t vram = instr.getVram();
            RegNode& node = g.nodes[i];
            node.instr = i;
            int rs = (int)instr.GetO32_rs();
            auto cond = N64Recomp::conditional_branch_ops.find(id);
            if (cond != N64Recomp::conditional_branch_ops.end()) {
                plain_effects(i, node);
                uint32_t target = (uint32_t)instr.getBranchVramGeneric();
                int taken;
                if (cond->second.link) {
                    int c = direct_call(i, vram, target, RegNode::Kind::Call, seq(i + 2));
                    taken = delay_copy(i, c);
                }
                else if (index_of(target) >= 0) {
                    taken = delay_copy(i, index_of(target));
                }
                else if (context.functions_by_vram.contains(target)) {
                    taken = delay_copy(i, direct_call(i, vram, target, RegNode::Kind::TailCall, -1));
                }
                else {
                    g.failure = fmt::format("branch out of the function at 0x{:08X}", vram);
                    return false;
                }
                g.nodes[i].succs = { taken, seq(cond->second.likely ? i + 2 : i + 1) };
                continue;
            }
            switch (id) {
                case InstrId::cpu_jal: {
                    uint32_t target = (uint32_t)instr.getBranchVramGeneric();
                    int c = direct_call(i, vram, target, RegNode::Kind::Call, seq(i + 2));
                    g.nodes[i].succs = { delay_copy(i, c) };
                    break;
                }
                case InstrId::cpu_jalr: {
                    int c = call_node(i, false, gpr_bit(rs), RegNode::Kind::Call, seq(i + 2), -1, true);
                    g.nodes[i].succs = { delay_copy(i, c) };
                    break;
                }
                case InstrId::cpu_j:
                case InstrId::cpu_b: {
                    uint32_t target = (uint32_t)instr.getBranchVramGeneric();
                    if (target == vram) {
                        // pause_self: never comes back.
                        break;
                    }
                    if (index_of(target) >= 0) {
                        g.nodes[i].succs = { delay_copy(i, index_of(target)) };
                    }
                    else if (context.functions_by_vram.contains(target)) {
                        g.nodes[i].succs = { delay_copy(i, direct_call(i, vram, target, RegNode::Kind::TailCall, -1)) };
                    }
                    else {
                        g.failure = fmt::format("jump out of the function at 0x{:08X}", vram);
                        return false;
                    }
                    break;
                }
                case InstrId::cpu_jr: {
                    if (rs == (int)rabbitizer::Registers::Cpu::GprO32::GPR_O32_ra) {
                        g.nodes[i].succs = { delay_copy(i, exit_node(i)) };
                        break;
                    }
                    auto jtbl = std::find_if(stats.jump_tables.begin(), stats.jump_tables.end(),
                        [vram](const N64Recomp::JumpTable& t) { return t.jr_vram == vram; });
                    if (jtbl != stats.jump_tables.end()) {
                        RegNode dispatch{};
                        for (uint32_t entry : jtbl->entries) {
                            int target = index_of(entry);
                            if (target < 0) {
                                g.failure = fmt::format("jump table entry out of the function at 0x{:08X}", vram);
                                return false;
                            }
                            dispatch.succs.push_back(target);
                        }
                        int d = add(std::move(dispatch));
                        g.nodes[i].succs = { delay_copy(i, d) };
                        break;
                    }
                    g.nodes[i].succs = { delay_copy(i, tail_call(i, false, gpr_bit(rs), -1, true)) };
                    break;
                }
                case InstrId::cpu_syscall: {
                    int e = exit_node(i);
                    g.nodes[i].succs = { call_node(i, true, 0, RegNode::Kind::TailCall, e) };
                    break;
                }
                default:
                    plain_effects(i, node);
                    g.nodes[i].succs = { seq(i + 1) };
                    break;
            }
            if (!g.failure.empty()) {
                return false;
            }
        }
        return true;
    }

    struct RegFlow {
        std::vector<GprMask> dirty_in;
        std::vector<GprMask> live_in;
        std::vector<GprMask> live_out;
        std::vector<GprMask> writeback;
        std::vector<bool> reached;
    };

    // What a call ends in the sync protocol's liveness: around a barrier everything dirty is written back and
    // everything live reloaded, so values come through it in the context; otherwise only what the callee certainly
    // writes, as a value it leaves alone has to be in the context before the call to be there after it.
    GprMask protocol_kill(const RegNode& node) {
        return node.call_def == ALL_GPRS ? ALL_GPRS : node.call_kill;
    }

    // Registers written back at a node: what the callee or the caller may read that the context doesn't have yet.
    GprMask node_writeback(const RegNode& node, GprMask dirty, GprMask exit_regs) {
        switch (node.kind) {
            case RegNode::Kind::Call: return dirty & node.call_use & ~node.call_keep;
            case RegNode::Kind::TailCall: return dirty & (node.call_use | exit_regs);
            case RegNode::Kind::Barrier: return dirty;
            case RegNode::Kind::Exit: return dirty & exit_regs;
            default: return 0;
        }
    }

    void solve_reg_flow(const RegGraph& g, GprMask exit_regs, RegFlow& f) {
        size_t count = g.nodes.size();
        f.dirty_in.assign(count, 0);
        f.reached.assign(count, false);
        f.live_in.assign(count, 0);
        f.live_out.assign(count, 0);
        f.writeback.assign(count, 0);
        // Forward: registers whose local may be newer than the context.
        std::vector<int> work{ 0 };
        f.reached[0] = true;
        while (!work.empty()) {
            int id = work.back();
            work.pop_back();
            const RegNode& node = g.nodes[id];
            GprMask in = f.dirty_in[id];
            GprMask out;
            switch (node.kind) {
                case RegNode::Kind::Plain: out = (in | node.def) & ~node.clean; break;
                // A local the callee may leave alone stays the newest value unless it was written back; one that is
                // still needed after the call is then live across it and stays in the context instead.
                case RegNode::Kind::Call: out = in & ~((node.call_use | protocol_kill(node)) & ~node.call_keep); break;
                default: out = 0; break;
            }
            for (int s : node.succs) {
                if (!f.reached[s] || (f.dirty_in[s] | out) != f.dirty_in[s]) {
                    f.reached[s] = true;
                    f.dirty_in[s] |= out;
                    work.push_back(s);
                }
            }
        }
        for (size_t id = 0; id < count; id++) {
            f.writeback[id] = f.reached[id] ? node_writeback(g.nodes[id], f.dirty_in[id], exit_regs) : 0;
        }
        // Backward: registers whose local value is needed (writebacks count as reads).
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t k = count; k-- > 0;) {
                const RegNode& node = g.nodes[k];
                GprMask out = 0;
                for (int s : node.succs) {
                    out |= f.live_in[s];
                }
                GprMask in;
                switch (node.kind) {
                    case RegNode::Kind::Plain: in = node.use | (out & ~node.def); break;
                    case RegNode::Kind::Call: in = node.use | f.writeback[k] | (out & ~protocol_kill(node)); break;
                    case RegNode::Kind::TailCall: in = node.use | f.writeback[k]; break;
                    case RegNode::Kind::Barrier: in = f.writeback[k]; break;
                    case RegNode::Kind::Exit: in = f.writeback[k]; break;
                }
                if (out != f.live_out[k] || in != f.live_in[k]) {
                    f.live_out[k] = out;
                    f.live_in[k] = in;
                    changed = true;
                }
            }
        }
    }

    // Every callee-saved register the function writes is saved to a stack slot at entry and reloaded from it before
    // every exit, and sp is back where it started.
    bool check_preserves(const RegGraph& g, const std::vector<rabbitizer::InstructionCpu>& instructions, std::string& why) {
        constexpr int UNKNOWN = INT32_MIN;
        size_t count = g.nodes.size();
        std::vector<int> delta(count, UNKNOWN);
        std::vector<bool> seen(count, false);
        auto sp_step = [&](const RegNode& node, int in) -> int {
            if (node.kind != RegNode::Kind::Plain || !(node.def & SP_REG)) {
                return in;
            }
            const auto& instr = instructions[node.instr];
            if (instr.getUniqueId() == InstrId::cpu_addiu && (int)instr.GetO32_rs() == 29 && in != UNKNOWN) {
                return in + (int16_t)instr.Get_immediate();
            }
            return UNKNOWN;
        };
        std::vector<int> work{ 0 };
        delta[0] = 0;
        seen[0] = true;
        while (!work.empty()) {
            int id = work.back();
            work.pop_back();
            int out = sp_step(g.nodes[id], delta[id]);
            for (int s : g.nodes[id].succs) {
                if (!seen[s]) {
                    seen[s] = true;
                    delta[s] = out;
                    work.push_back(s);
                }
                else if (delta[s] != out && delta[s] != UNKNOWN) {
                    delta[s] = UNKNOWN;
                    work.push_back(s);
                }
            }
        }
        for (size_t id = 0; id < count; id++) {
            const RegNode& node = g.nodes[id];
            bool leaves = node.kind == RegNode::Kind::TailCall || (node.kind == RegNode::Kind::Exit && node.key != SIZE_MAX);
            if (seen[id] && leaves && delta[id] != 0) {
                why = "sp not restored";
                return false;
            }
        }
        GprMask written = 0;
        for (size_t id = 0; id < count; id++) {
            if (seen[id] && g.nodes[id].kind == RegNode::Kind::Plain) {
                written |= g.nodes[id].def;
            }
        }
        for (int reg = 16; reg < 32; reg++) {
            if (!(CALLEE_SAVED & gpr_bit(reg)) || !(written & gpr_bit(reg))) {
                continue;
            }
            // Slots holding the entry value: stores of the register before anything writes it.
            auto is_sp_access = [&](const RegNode& node, bool store, int& slot) {
                if (node.kind != RegNode::Kind::Plain || node.instr == SIZE_MAX) {
                    return false;
                }
                const auto& instr = instructions[node.instr];
                InstrId id = instr.getUniqueId();
                if ((int)instr.GetO32_rt() != reg || (int)instr.GetO32_rs() != 29) {
                    return false;
                }
                if (store ? id != InstrId::cpu_sw : id != InstrId::cpu_lw) {
                    return false;
                }
                slot = (int)(int16_t)instr.Get_immediate();
                return true;
            };
            // 0 unvisited, 1 entry value, 2 restored from a save slot, 3 anything else.
            std::vector<uint8_t> state(count, 0);
            std::set<int> save_slots;
            std::set<int> other_stores;
            for (int pass = 0; pass < 2; pass++) {
                std::fill(state.begin(), state.end(), 0);
                state[0] = 1;
                std::vector<int> stack{ 0 };
                while (!stack.empty()) {
                    int id = stack.back();
                    stack.pop_back();
                    const RegNode& node = g.nodes[id];
                    uint8_t in = state[id];
                    uint8_t out = in;
                    int slot;
                    if (pass == 0 && is_sp_access(node, true, slot) && in == 1 && delta[id] != UNKNOWN) {
                        save_slots.insert(delta[id] + slot);
                    }
                    if (node.kind == RegNode::Kind::Plain && (node.def & gpr_bit(reg))) {
                        out = 3;
                        if (is_sp_access(node, false, slot) && delta[id] != UNKNOWN && save_slots.contains(delta[id] + slot)) {
                            out = 2;
                        }
                    }
                    for (int s : node.succs) {
                        uint8_t merged = state[s] == 0 ? out : (state[s] == out ? out : ((state[s] != 3 && out != 3) ? 2 : 3));
                        if (merged != state[s]) {
                            state[s] = merged;
                            stack.push_back(s);
                        }
                    }
                }
            }
            // Other stores into a save slot would change what gets restored.
            for (size_t id = 0; id < count; id++) {
                const RegNode& node = g.nodes[id];
                if (!seen[id] || node.kind != RegNode::Kind::Plain || node.instr == SIZE_MAX || delta[id] == UNKNOWN) {
                    continue;
                }
                const auto& instr = instructions[node.instr];
                auto store = N64Recomp::store_ops.find(instr.getUniqueId());
                if (store == N64Recomp::store_ops.end() || (int)instr.GetO32_rs() != 29) {
                    continue;
                }
                int slot = delta[id] + (int)(int16_t)instr.Get_immediate();
                int width = 4;
                switch (store->second.type) {
                    case N64Recomp::StoreOpType::SB: width = 1; break;
                    case N64Recomp::StoreOpType::SH: width = 2; break;
                    case N64Recomp::StoreOpType::SD: case N64Recomp::StoreOpType::SDC1:
                    case N64Recomp::StoreOpType::SDL: case N64Recomp::StoreOpType::SDR: width = 8; break;
                    default: break;
                }
                for (int saved : save_slots) {
                    bool overlaps = slot < saved + 4 && saved < slot + width;
                    bool is_save = instr.getUniqueId() == InstrId::cpu_sw && (int)instr.GetO32_rt() == reg && slot == saved;
                    if (overlaps && !is_save) {
                        why = fmt::format("stack slot of r{} overwritten", reg);
                        return false;
                    }
                }
            }
            for (size_t id = 0; id < count; id++) {
                const RegNode& node = g.nodes[id];
                bool leaves = node.kind == RegNode::Kind::TailCall || (node.kind == RegNode::Kind::Exit && node.key != SIZE_MAX);
                if (seen[id] && leaves && state[id] == 3) {
                    why = fmt::format("r{} not restored", reg);
                    return false;
                }
            }
        }
        return true;
    }

    // codegen.skip_saves: for each callee-saved register (ra included), the instructions that store its entry value
    // to a stack slot and those that load it back from one, when no other instruction reads or writes those slots
    // and a save runs on every path to each restore. other_defs: written by something besides its restores.
    struct SaveSlots {
        GprMask regs = 0;
        GprMask other_defs = 0;
        std::vector<size_t> saves[32];
        std::vector<size_t> restores[32];
        // For the report: why a saved register doesn't qualify.
        std::string why;
    };

    void find_saves(const RegGraph& g, const std::vector<rabbitizer::InstructionCpu>& instructions, SaveSlots& out) {
        using namespace N64Recomp;
        constexpr int UNKNOWN = INT32_MIN;
        size_t count = g.nodes.size();
        out = SaveSlots{};
        std::vector<int> delta(count, UNKNOWN);
        std::vector<bool> seen(count, false);
        std::vector<int> work{ 0 };
        delta[0] = 0;
        seen[0] = true;
        bool sp_lost = false;
        while (!work.empty()) {
            int id = work.back();
            work.pop_back();
            const RegNode& node = g.nodes[id];
            int o = delta[id];
            if (node.kind == RegNode::Kind::Plain && (node.def & SP_REG)) {
                const auto& instr = instructions[node.instr];
                o = (instr.getUniqueId() == InstrId::cpu_addiu && (int)instr.GetO32_rs() == 29 && o != UNKNOWN) ?
                    o + (int16_t)instr.Get_immediate() : UNKNOWN;
            }
            for (int s : node.succs) {
                if (!seen[s]) {
                    seen[s] = true;
                    delta[s] = o;
                    work.push_back(s);
                }
                else if (delta[s] != o && delta[s] != UNKNOWN) {
                    delta[s] = UNKNOWN;
                    work.push_back(s);
                }
            }
        }
        // Every sp-relative access: (node, slot, width, store, register).
        struct Access { int node; int slot; int width; bool store; int reg; InstrId id; };
        std::vector<Access> accesses;
        for (size_t id = 0; id < count; id++) {
            const RegNode& node = g.nodes[id];
            if (!seen[id] || node.kind != RegNode::Kind::Plain || node.instr == SIZE_MAX) {
                continue;
            }
            const auto& instr = instructions[node.instr];
            if ((int)instr.GetO32_rs() != 29) {
                continue;
            }
            InstrId iid = instr.getUniqueId();
            int width = 0;
            bool store = false;
            auto st = store_ops.find(iid);
            if (st != store_ops.end()) {
                store = true;
                switch (st->second.type) {
                    case StoreOpType::SB: width = 1; break;
                    case StoreOpType::SH: width = 2; break;
                    case StoreOpType::SD: case StoreOpType::SDC1: case StoreOpType::SDL: case StoreOpType::SDR: width = 8; break;
                    default: width = 4; break;
                }
            }
            auto ld = binary_ops.find(iid);
            if (ld != binary_ops.end()) {
                switch (ld->second.type) {
                    case BinaryOpType::LB: case BinaryOpType::LBU: width = 1; break;
                    case BinaryOpType::LH: case BinaryOpType::LHU: width = 2; break;
                    case BinaryOpType::LW: case BinaryOpType::LWU: case BinaryOpType::LWL: case BinaryOpType::LWR: width = 4; break;
                    case BinaryOpType::LD: case BinaryOpType::LDL: case BinaryOpType::LDR: width = 8; break;
                    default: break;
                }
            }
            if (width == 0) {
                continue;
            }
            if (delta[id] == UNKNOWN) {
                if (!sp_lost) {
                    out.why = fmt::format("sp unknown at 0x{:08X}", instr.getVram());
                }
                sp_lost = true;
                continue;
            }
            int slot = delta[id] + (int)(int16_t)instr.Get_immediate();
            // Unaligned forms touch bytes on both sides of the address.
            bool unaligned = iid == InstrId::cpu_lwl || iid == InstrId::cpu_lwr || iid == InstrId::cpu_ldl || iid == InstrId::cpu_ldr ||
                             iid == InstrId::cpu_swl || iid == InstrId::cpu_swr || iid == InstrId::cpu_sdl || iid == InstrId::cpu_sdr;
            if (unaligned) {
                slot -= 7;
                width = 15;
            }
            accesses.push_back({ (int)id, slot, width, store, (int)instr.GetO32_rt(), iid });
        }
        if (sp_lost) {
            return;
        }
        std::vector<std::vector<int>> preds(count);
        for (size_t id = 0; id < count; id++) {
            if (seen[id]) {
                for (int s : g.nodes[id].succs) {
                    preds[s].push_back((int)id);
                }
            }
        }
        // A property that holds on every path to a node: the entry edge gives `at_entry`, a node passes on what it
        // got through `step`. Starts from true everywhere and only falls.
        auto on_every_path = [&](uint8_t at_entry, auto step) {
            std::vector<uint8_t> in(count, 1);
            for (bool changed = true; changed;) {
                changed = false;
                for (size_t id = 0; id < count; id++) {
                    if (!seen[id]) {
                        continue;
                    }
                    uint8_t v = id == 0 ? at_entry : 1;
                    for (int p : preds[id]) {
                        v &= step(p, in[p]);
                    }
                    if (v != in[id]) {
                        in[id] = v;
                        changed = true;
                    }
                }
            }
            return in;
        };
        auto plain_instr = [&](int id) -> const rabbitizer::InstructionCpu* {
            const RegNode& node = g.nodes[id];
            return node.kind == RegNode::Kind::Plain && node.instr != SIZE_MAX ? &instructions[node.instr] : nullptr;
        };
        for (int reg = 16; reg < 32; reg++) {
            if (!(CALLEE_SAVED & gpr_bit(reg))) {
                continue;
            }
            // Where the register still holds its entry value, then the stores of it to the stack there (saves).
            std::vector<uint8_t> entry_value = on_every_path(1, [&](int p, uint8_t v) -> uint8_t {
                const RegNode& node = g.nodes[p];
                bool writes = node.kind == RegNode::Kind::Plain ? (node.def & gpr_bit(reg)) != 0 : node.kind != RegNode::Kind::Exit;
                return writes ? 0 : v;
            });
            std::set<int> slots;
            std::set<int> save_nodes;
            for (size_t id = 0; id < count; id++) {
                const auto* instr = seen[id] ? plain_instr((int)id) : nullptr;
                if (instr && entry_value[id] && delta[id] != UNKNOWN && instr->getUniqueId() == InstrId::cpu_sw &&
                    (int)instr->GetO32_rs() == 29 && (int)instr->GetO32_rt() == reg) {
                    slots.insert(delta[id] + (int)(int16_t)instr->Get_immediate());
                    save_nodes.insert((int)id);
                }
            }
            if (slots.empty()) {
                continue;
            }
            std::vector<uint8_t> saved = on_every_path(0, [&](int p, uint8_t v) -> uint8_t {
                return save_nodes.contains(p) ? 1 : v;
            });
            bool ok = true;
            std::vector<size_t> saves, restores;
            std::set<int> restore_nodes;
            for (const Access& a : accesses) {
                bool overlaps = false;
                for (int s : slots) {
                    overlaps |= a.slot < s + 4 && s < a.slot + a.width;
                }
                if (!overlaps) {
                    continue;
                }
                const RegNode& node = g.nodes[a.node];
                if (a.store) {
                    if (save_nodes.contains(a.node)) {
                        saves.push_back(node.instr);
                    }
                    else {
                        ok = false;
                        out.why += fmt::format(" r{}: store at 0x{:08X}", reg, instructions[node.instr].getVram());
                    }
                }
                else if (a.id == InstrId::cpu_lw && a.reg == reg && a.width == 4 && slots.contains(a.slot) && saved[a.node]) {
                    restores.push_back(node.instr);
                    restore_nodes.insert(a.node);
                }
                else {
                    ok = false;
                    out.why += fmt::format(" r{}: load at 0x{:08X} node {} slot {} width {}{}", reg, instructions[node.instr].getVram(), a.node, a.slot,
                                           a.width, saved[a.node] ? "" : " (unsaved)");
                }
            }
            // The same instruction (a delay slot's copies) has to be the same kind everywhere.
            for (size_t id = 0; id < count && ok; id++) {
                const RegNode& node = g.nodes[id];
                if (!seen[id] || node.kind != RegNode::Kind::Plain || node.instr == SIZE_MAX) {
                    continue;
                }
                bool listed_save = std::find(saves.begin(), saves.end(), node.instr) != saves.end();
                bool listed_restore = std::find(restores.begin(), restores.end(), node.instr) != restores.end();
                if (listed_save && !save_nodes.contains((int)id)) {
                    ok = false;
                }
                if (listed_restore && !restore_nodes.contains((int)id)) {
                    ok = false;
                }
                if ((node.def & gpr_bit(reg)) && !listed_restore) {
                    out.other_defs |= gpr_bit(reg);
                }
            }
            if (!ok) {
                out.other_defs &= ~gpr_bit(reg);
                if (out.why.find(fmt::format(" r{}:", reg)) == std::string::npos) {
                    out.why += fmt::format(" r{}: delay slot copies differ", reg);
                }
                continue;
            }
            std::sort(saves.begin(), saves.end());
            saves.erase(std::unique(saves.begin(), saves.end()), saves.end());
            std::sort(restores.begin(), restores.end());
            restores.erase(std::unique(restores.begin(), restores.end()), restores.end());
            out.regs |= gpr_bit(reg);
            out.saves[reg] = std::move(saves);
            out.restores[reg] = std::move(restores);
        }
    }

    bool load_function_view(const N64Recomp::Context& context, const N64Recomp::Function& func, std::vector<rabbitizer::InstructionCpu>& instructions,
                            N64Recomp::FunctionStats& stats, std::unordered_set<uint32_t>& jtbl_lw) {
        instructions.clear();
        instructions.reserve(func.words.size());
        uint32_t vram = func.vram;
        for (uint32_t word : func.words) {
            instructions.emplace_back(byteswap(word), vram);
            vram += 4;
        }
        if (!N64Recomp::analyze_function(context, func, instructions, stats)) {
            return false;
        }
        for (const auto& jtbl : stats.jump_tables) {
            jtbl_lw.insert(jtbl.lw_vram);
        }
        return true;
    }

    bool regcache_wanted(const N64Recomp::Context& context, const N64Recomp::Function& func) {
        const auto& cg = context.codegen;
        if (!cg.regcache || func.stubbed || func.words.empty()) {
            return false;
        }
        if (cg.regcache_exclude.contains(func.name)) {
            return false;
        }
        if (!cg.regcache_only.empty() && !cg.regcache_only.contains(func.name)) {
            return false;
        }
        // Hooks inside the body and return hooks would need a flush around arbitrary text.
        if (!func.return_hook.empty()) {
            return false;
        }
        for (const auto& [index, text] : func.function_hooks) {
            if (index > 0) {
                return false;
            }
        }
        return true;
    }

    // Runtime C code rather than recompiled MIPS: reimplemented, stubbed or ignored functions, and functions
    // whose first instruction's hook returns.
    bool runtime_function(const N64Recomp::Context& context, size_t index) {
        const auto& func = context.functions[index];
        if (func.reimplemented || func.stubbed || func.ignored || func.words.empty()) {
            return true;
        }
        auto hook0 = func.function_hooks.find(0);
        return hook0 != func.function_hooks.end() && hook0->second.find("return;") != std::string::npos;
    }

    // abi_trace: the TRACE_* statement for one instruction (recomp.h), or nothing.
    std::string trace_statement(const rabbitizer::InstructionCpu& instr, InstrId id, uint32_t vram) {
        using namespace N64Recomp;
        GprMask use = 0, def = 0;
        gpr_effects(instr, id, use, def);
        int rd = (int)instr.GetO32_rd();
        int rs = (int)instr.GetO32_rs();
        int rt = (int)instr.GetO32_rt();
        auto sink = [&](GprMask m, char kind) {
            return m ? fmt::format("TRACE_SINK(ctx, 0x{:X}u, 0x{:08X}u, '{}'); ", m, vram, kind) : std::string{};
        };
        auto alu = [&](GprMask u, GprMask d) {
            return (u || d) ? fmt::format("TRACE_ALU(ctx, 0x{:X}u, 0x{:X}u); ", u, d) : std::string{};
        };
        auto store = store_ops.find(id);
        if (store != store_ops.end()) {
            GprMask value = store->second.value_input == Operand::Rt ? gpr_bit(rt) : 0;
            return sink(value, 'S') + sink(gpr_bit(rs), 'A');
        }
        if (conditional_branch_ops.contains(id)) {
            return sink(use, 'B');
        }
        switch (id) {
            case InstrId::cpu_jr:
                return sink(gpr_bit(rs), 'J');
            case InstrId::cpu_jalr:
                return sink(gpr_bit(rs), 'J') + alu(0, gpr_bit(rd));
            case InstrId::cpu_mult: case InstrId::cpu_multu: case InstrId::cpu_div: case InstrId::cpu_divu:
            case InstrId::cpu_dmult: case InstrId::cpu_dmultu: case InstrId::cpu_ddiv: case InstrId::cpu_ddivu:
                return alu(use, 1);
            case InstrId::cpu_mthi: case InstrId::cpu_mtlo:
                return alu(gpr_bit(rs), 1);
            case InstrId::cpu_mfhi: case InstrId::cpu_mflo:
                return alu(1, gpr_bit(rd));
            case InstrId::cpu_mtc1: case InstrId::cpu_dmtc1: case InstrId::cpu_ctc1: case InstrId::cpu_mtc0:
                return sink(gpr_bit(rt), 'F');
            default:
                break;
        }
        auto binary = binary_ops.find(id);
        if (binary != binary_ops.end()) {
            switch (binary->second.type) {
                case BinaryOpType::LD: case BinaryOpType::LW: case BinaryOpType::LWU: case BinaryOpType::LH:
                case BinaryOpType::LHU: case BinaryOpType::LB: case BinaryOpType::LBU: case BinaryOpType::LDL:
                case BinaryOpType::LDR: case BinaryOpType::LWL: case BinaryOpType::LWR:
                    return fmt::format("TRACE_LOAD(ctx, 0x{:X}u, 0x{:X}u, 0x{:X}u, 0x{:08X}u); ", gpr_bit(rs), use, def, vram);
                default:
                    break;
            }
        }
        return alu(use, def);
    }

    // The caller-saved registers a call certainly writes. Runtime code sets v0 and nothing else it can be relied
    // on for (the report lists the runtime functions whose v0 or v1 callers read, each checked against its
    // libultra prototype); a stubbed function writes nothing; a static function, a syscall or an unknown target
    // nothing; an indirect call what every function whose address is taken writes.
    GprMask callee_must_def(const N64Recomp::Context& context, const RegNode& node) {
        if (node.callee >= 0) {
            size_t c = (size_t)node.callee;
            if (runtime_function(context, c)) {
                return context.functions[c].stubbed ? 0 : gpr_bit(2);
            }
            return c < context.regcache_must_def.size() ? context.regcache_must_def[c] : 0;
        }
        return node.indirect ? context.regcache_indirect_must_def : 0;
    }

    // The GPRs a call may read or write; ALL_GPRS when that isn't known. Runtime code reads the argument registers
    // and sp and writes v0 and v1.
    GprMask callee_may_touch(const N64Recomp::Context& context, const RegNode& node) {
        if (node.callee >= 0) {
            size_t c = (size_t)node.callee;
            if (runtime_function(context, c)) {
                return ARG_REGS | RETURN_REGS | SP_REG;
            }
            return c < context.regcache_may_touch.size() ? context.regcache_may_touch[c] : ALL_GPRS;
        }
        return node.indirect ? context.regcache_indirect_may_touch : ALL_GPRS;
    }

    // skip_saves: a call's callee-saved needs or clobbers (values: per function, indirect: the union for indirect
    // calls). Runtime code keeps the o32 ABI; so does a static function (compiled C the symbols don't name, whose
    // caller flushes and reloads around it anyway); a call to an address several overlays share can reach any of
    // their functions there.
    GprMask cs_call_value(const N64Recomp::Context& context, const RegNode& node, const std::vector<GprMask>& values, GprMask indirect) {
        auto of = [&](size_t c) -> GprMask {
            if (runtime_function(context, c)) {
                return 0;
            }
            return c < values.size() && context.reg_summaries[c].analyzed ? values[c] : CALLEE_SAVED;
        };
        if (node.callee >= 0) {
            return of((size_t)node.callee);
        }
        if (node.indirect) {
            return indirect;
        }
        auto it = context.functions_by_vram.find(node.target);
        if (node.target == 0 || it == context.functions_by_vram.end()) {
            return 0;
        }
        GprMask m = 0;
        for (size_t c : it->second) {
            m |= of(c);
        }
        return m;
    }

    void apply_call_kills(const N64Recomp::Context& context, RegGraph& g) {
        for (auto& node : g.nodes) {
            if (node.kind == RegNode::Kind::Call || node.kind == RegNode::Kind::TailCall) {
                node.call_kill = callee_must_def(context, node);
                node.call_keep = node.call_def == ALL_GPRS || context.regcache_may_touch.empty() ? 0 :
                                 ALL_GPRS & ~callee_may_touch(context, node);
            }
        }
    }

    // The caller-saved registers written on every path from the function's entry to each of its returns.
    GprMask must_def_of(const RegGraph& g) {
        size_t count = g.nodes.size();
        std::vector<std::vector<int>> preds(count);
        std::vector<bool> reached(count, false);
        std::vector<int> work{ 0 };
        reached[0] = true;
        while (!work.empty()) {
            int k = work.back();
            work.pop_back();
            for (int s : g.nodes[k].succs) {
                preds[s].push_back(k);
                if (!reached[s]) {
                    reached[s] = true;
                    work.push_back(s);
                }
            }
        }
        auto gen = [&](size_t k) -> GprMask {
            const RegNode& node = g.nodes[k];
            switch (node.kind) {
                case RegNode::Kind::Plain: return node.def;
                case RegNode::Kind::Call: case RegNode::Kind::TailCall: return node.def | node.call_kill;
                default: return 0;
            }
        };
        std::vector<GprMask> out(count, ALL_GPRS);
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t k = 0; k < count; k++) {
                if (!reached[k]) {
                    continue;
                }
                GprMask in = ALL_GPRS;
                if (k == 0) {
                    in = 0;
                }
                for (int p : preds[k]) {
                    in &= out[p];
                }
                GprMask o = in | gen(k);
                if (o != out[k]) {
                    out[k] = o;
                    changed = true;
                }
            }
        }
        GprMask result = ALL_GPRS;
        for (size_t k = 0; k < count; k++) {
            if (reached[k] && g.nodes[k].kind == RegNode::Kind::Exit) {
                GprMask in = 0;
                if (!preds[k].empty()) {
                    in = ALL_GPRS;
                    for (int p : preds[k]) {
                        in &= out[p];
                    }
                }
                result &= in;
            }
        }
        return result & CALLER_SAVED;
    }

    // regcache = "gpr+fpr": how an instruction uses the FPRs, through the operation tables the emitter uses.
    // view: 'S' single, 'D' double (both halves), 'W' raw low word, 'U' raw doubleword, 'H' anything else.
    struct FprAccess {
        int reg;
        char view;
        bool def;
    };

    void fpr_accesses(const rabbitizer::InstructionCpu& instr, InstrId id, std::vector<FprAccess>& out) {
        using namespace N64Recomp;
        out.clear();
        int fd = (int)instr.GetO32_fd();
        int fs = (int)instr.GetO32_fs();
        int ft = (int)instr.GetO32_ft();
        auto add = [&](Operand op, bool def) {
            switch (op) {
                case Operand::Fd: out.push_back({ fd, 'S', def }); break;
                case Operand::Fs: out.push_back({ fs, 'S', def }); break;
                case Operand::Ft: out.push_back({ ft, 'S', def }); break;
                case Operand::FdDouble: out.push_back({ fd, 'D', def }); break;
                case Operand::FsDouble: out.push_back({ fs, 'D', def }); break;
                case Operand::FtDouble: out.push_back({ ft, 'D', def }); break;
                case Operand::FdU32L: out.push_back({ fd, 'W', def }); break;
                case Operand::FsU32L: out.push_back({ fs, 'W', def }); break;
                case Operand::FtU32L: out.push_back({ ft, 'W', def }); break;
                case Operand::FdU64: out.push_back({ fd, 'U', def }); break;
                case Operand::FsU64: out.push_back({ fs, 'U', def }); break;
                case Operand::FtU64: out.push_back({ ft, 'U', def }); break;
                case Operand::FdU32H: out.push_back({ fd, 'H', def }); break;
                case Operand::FsU32H: out.push_back({ fs, 'H', def }); break;
                case Operand::FtU32H: out.push_back({ ft, 'H', def }); break;
                default: break;
            }
        };
        auto unary = unary_ops.find(id);
        if (unary != unary_ops.end()) {
            add(unary->second.output, true);
            add(unary->second.input, false);
        }
        auto binary = binary_ops.find(id);
        if (binary != binary_ops.end()) {
            add(binary->second.output, true);
            add(binary->second.operands.operands[0], false);
            add(binary->second.operands.operands[1], false);
        }
        auto store = store_ops.find(id);
        if (store != store_ops.end()) {
            add(store->second.value_input, false);
        }
    }

    // The FPR part of a function's plan: even registers whose every use is a single or raw word (fpr32 locals) or
    // a double or raw doubleword (fpr locals), and whose odd partner is never named. Calls and returns write back
    // every local that may be newer than the context, and a call is followed by reloads of the locals still needed.
    void plan_fprs(const RegGraph& g, const std::vector<rabbitizer::InstructionCpu>& instructions, N64Recomp::FunctionRegPlan& plan) {
        uint32_t single = 0, dbl = 0, bad = 0, arith = 0;
        std::vector<FprAccess> acc;
        size_t count = g.nodes.size();
        size_t n_instr = instructions.size();
        // Callee-saved FPRs are saved with sdc1 before the function writes them and restored with ldc1 after its
        // last use. Those two go straight to the context, so the register's uses in between decide its type.
        std::vector<uint32_t> all_use(n_instr, 0), all_def(n_instr, 0);
        for (size_t i = 0; i < n_instr; i++) {
            fpr_accesses(instructions[i], instructions[i].getUniqueId(), acc);
            for (const FprAccess& a : acc) {
                uint32_t bits = a.view == 'D' || a.view == 'U' ? (3u << (a.reg & ~1)) : (1u << a.reg);
                (a.def ? all_def[i] : all_use[i]) |= bits;
            }
        }
        std::vector<uint32_t> may_def(count, 0), touched(count, 0);
        std::vector<bool> fwd_reached(count, false);
        {
            std::vector<int> work{ 0 };
            fwd_reached[0] = true;
            while (!work.empty()) {
                int k = work.back();
                work.pop_back();
                const RegNode& n = g.nodes[k];
                uint32_t out = n.kind == RegNode::Kind::Plain ? (may_def[k] | (n.instr < n_instr ? all_def[n.instr] : 0)) : 0xFFFFFFFFu;
                for (int s : n.succs) {
                    if (!fwd_reached[s] || (may_def[s] | out) != may_def[s]) {
                        fwd_reached[s] = true;
                        may_def[s] |= out;
                        work.push_back(s);
                    }
                }
            }
            std::vector<uint32_t> t_in(count, 0);
            for (bool changed = true; changed;) {
                changed = false;
                for (size_t k = count; k-- > 0;) {
                    const RegNode& n = g.nodes[k];
                    uint32_t out = 0;
                    for (int s : n.succs) {
                        out |= t_in[s];
                    }
                    uint32_t in = n.kind == RegNode::Kind::Plain ?
                                  (out | (n.instr < n_instr ? all_use[n.instr] | all_def[n.instr] : 0)) :
                                  n.kind == RegNode::Kind::Exit ? 0u : 0xFFFFFFFFu;
                    if (out != touched[k] || in != t_in[k]) {
                        touched[k] = out;
                        t_in[k] = in;
                        changed = true;
                    }
                }
            }
        }
        std::vector<int> direct(n_instr, 0);  // 1: save, 2: restore, -1: not either
        for (size_t k = 0; k < count; k++) {
            const RegNode& n = g.nodes[k];
            if (n.kind != RegNode::Kind::Plain || n.instr >= n_instr || !fwd_reached[k] || direct[n.instr] < 0) {
                continue;
            }
            const auto& instr = instructions[n.instr];
            InstrId id = instr.getUniqueId();
            int ft = (int)instr.GetO32_ft();
            uint32_t pair = 3u << (ft & ~1);
            int kind = 0;
            if ((int)instr.GetO32_rs() == 29 && !(ft & 1)) {
                if (id == InstrId::cpu_sdc1 && !(may_def[k] & pair)) {
                    kind = 1;
                }
                else if (id == InstrId::cpu_ldc1 && !(touched[k] & pair)) {
                    kind = 2;
                }
            }
            direct[n.instr] = kind == 0 || (direct[n.instr] != 0 && direct[n.instr] != kind) ? -1 : kind;
        }
        uint32_t direct_regs = 0;
        // odd_def/odd_use: raw word writes and reads of an odd register, the high half of the pair below it (the
        // lui/mtc1 doubleword constant idiom), by the even register's bit.
        std::vector<uint32_t> use(n_instr, 0), def(n_instr, 0), kill(n_instr, 0), odd_def(n_instr, 0), odd_use(n_instr, 0);
        for (size_t i = 0; i < n_instr; i++) {
            if (direct[i] > 0) {
                int ft = (int)instructions[i].GetO32_ft();
                direct_regs |= 1u << ft;
                if (direct[i] == 2) {
                    kill[i] = 1u << ft;
                }
                continue;
            }
            fpr_accesses(instructions[i], instructions[i].getUniqueId(), acc);
            for (const FprAccess& a : acc) {
                if (a.reg & 1) {
                    if (a.view == 'W') {
                        (a.def ? odd_def[i] : odd_use[i]) |= 1u << (a.reg - 1);
                        single |= 1u << (a.reg - 1);
                    }
                    else {
                        bad |= 1u << (a.reg - 1);
                    }
                    continue;
                }
                uint32_t bit = 1u << a.reg;
                if (a.view == 'S' || a.view == 'W') {
                    single |= bit;
                }
                else if (a.view == 'D' || a.view == 'U') {
                    dbl |= bit;
                }
                else {
                    bad |= bit;
                }
                if (a.view == 'S' || a.view == 'D') {
                    arith |= bit;
                }
                (a.def ? def[i] : use[i]) |= bit;
            }
        }
        (void)single;
        (void)dbl;
        (void)arith;
        (void)direct_regs;
        uint32_t pairs = (single | dbl) & ~bad;
        // Per instruction and register: which half a write or read covers. A single or word write replaces only the
        // low half, so the high half (the odd register's slot) keeps the last doubleword written.
        std::vector<std::vector<FprAccess>> accesses(n_instr);
        for (size_t i = 0; i < n_instr; i++) {
            if (direct[i] <= 0) {
                fpr_accesses(instructions[i], instructions[i].getUniqueId(), accesses[i]);
            }
        }
        auto wide = [](const FprAccess& a) { return a.view == 'D' || a.view == 'U'; };
        // Webs, one register at a time. Definition sites: 0 the context at entry, then one per call and per direct
        // restore (the context again), then one per instruction that writes the register.
        for (int reg = 0; reg < 32; reg += 2) {
            uint32_t bit = 1u << reg;
            if (!(pairs & bit)) {
                continue;
            }
            std::vector<uint8_t> def64(n_instr, 0), use64(n_instr, 0);
            for (size_t i = 0; i < n_instr; i++) {
                for (const FprAccess& a : accesses[i]) {
                    if (a.reg == reg && wide(a)) {
                        (a.def ? def64[i] : use64[i]) = 1;
                    }
                }
                if (kill[i] & bit) {
                    def64[i] = 1;
                }
            }
            std::vector<int> site_of_instr(n_instr, -1), site_of_node(count, -1);
            std::vector<bool> site_loaded{ true };
            std::vector<size_t> site_instr{ SIZE_MAX };
            for (size_t k = 0; k < count; k++) {
                const RegNode& n = g.nodes[k];
                if (n.kind == RegNode::Kind::Call) {
                    site_of_node[k] = (int)site_loaded.size();
                    site_loaded.push_back(true);
                    site_instr.push_back(SIZE_MAX);
                }
            }
            for (size_t i = 0; i < n_instr; i++) {
                if ((def[i] | kill[i] | odd_def[i]) & bit) {
                    site_of_instr[i] = (int)site_loaded.size();
                    site_loaded.push_back((kill[i] & bit) != 0);
                    site_instr.push_back(i);
                }
            }
            size_t sites = site_loaded.size();
            size_t words = (sites + 63) / 64;
            // Forward: the definition sites reaching each node, for the low half (any write) and the high half
            // (doubleword writes, calls, restores and entry).
            std::vector<uint64_t> in_lo(count * words, 0), in_hi(count * words, 0);
            std::vector<bool> reached(count, false);
            std::vector<uint64_t> out_lo(words), out_hi(words);
            in_lo[0] |= 1;
            in_hi[0] |= 1;
            reached[0] = true;
            std::vector<int> work{ 0 };
            while (!work.empty()) {
                int k = work.back();
                work.pop_back();
                const RegNode& n = g.nodes[k];
                bool plain = n.kind == RegNode::Kind::Plain && n.instr < n_instr;
                int gen = n.kind == RegNode::Kind::Call ? site_of_node[k] : plain ? site_of_instr[n.instr] : -1;
                bool call = n.kind == RegNode::Kind::Call;
                bool gen_lo = gen >= 0 && (call || ((def[n.instr] | kill[n.instr]) & bit));
                bool gen_hi = gen >= 0 && (call || def64[n.instr] || (odd_def[n.instr] & bit));
                std::copy(in_lo.begin() + k * words, in_lo.begin() + (k + 1) * words, out_lo.begin());
                std::copy(in_hi.begin() + k * words, in_hi.begin() + (k + 1) * words, out_hi.begin());
                if (gen_lo) {
                    std::fill(out_lo.begin(), out_lo.end(), 0);
                    out_lo[gen / 64] |= 1ull << (gen % 64);
                }
                if (gen_hi) {
                    std::fill(out_hi.begin(), out_hi.end(), 0);
                    out_hi[gen / 64] |= 1ull << (gen % 64);
                }
                for (int s : n.succs) {
                    bool grew = !reached[s];
                    for (size_t w = 0; w < words; w++) {
                        uint64_t v = in_lo[s * words + w] | out_lo[w];
                        uint64_t h = in_hi[s * words + w] | out_hi[w];
                        grew |= v != in_lo[s * words + w] || h != in_hi[s * words + w];
                        in_lo[s * words + w] = v;
                        in_hi[s * words + w] = h;
                    }
                    if (grew) {
                        reached[s] = true;
                        work.push_back(s);
                    }
                }
            }
            // Join the definitions that reach a common read (a doubleword read reads both halves), or a common
            // writeback of a half (a call, a tail call or a return reached by an instruction's write of that half).
            std::vector<int> parent(sites);
            for (size_t s = 0; s < sites; s++) {
                parent[s] = (int)s;
            }
            std::function<int(int)> find = [&](int s) { return parent[s] == s ? s : parent[s] = find(parent[s]); };
            auto join_reaching = [&](const std::vector<uint64_t>& in, size_t k, int first) -> int {
                for (size_t s = 0; s < sites; s++) {
                    if (in[k * words + s / 64] & (1ull << (s % 64))) {
                        if (first < 0) {
                            first = (int)s;
                        }
                        else {
                            parent[find((int)s)] = find(first);
                        }
                    }
                }
                return first;
            };
            auto written = [&](const std::vector<uint64_t>& in, size_t k) {
                for (size_t s = 0; s < sites; s++) {
                    if (!site_loaded[s] && (in[k * words + s / 64] & (1ull << (s % 64)))) {
                        return true;
                    }
                }
                return false;
            };
            std::vector<int> use_site(n_instr, -1), odd_use_site(n_instr, -1), sync_lo(count, -1), sync_hi(count, -1);
            for (size_t k = 0; k < count; k++) {
                const RegNode& n = g.nodes[k];
                if (!reached[k]) {
                    continue;
                }
                if (n.kind == RegNode::Kind::Plain && n.instr < n_instr && ((use[n.instr] | odd_use[n.instr]) & bit)) {
                    if (use[n.instr] & bit) {
                        int s = join_reaching(in_lo, k, use_site[n.instr]);
                        if (use64[n.instr]) {
                            s = join_reaching(in_hi, k, s);
                        }
                        use_site[n.instr] = s;
                    }
                    if (odd_use[n.instr] & bit) {
                        odd_use_site[n.instr] = join_reaching(in_hi, k, odd_use_site[n.instr]);
                    }
                }
                else if (n.kind == RegNode::Kind::Call || n.kind == RegNode::Kind::TailCall || n.kind == RegNode::Kind::Exit) {
                    if (written(in_lo, k)) {
                        sync_lo[k] = join_reaching(in_lo, k, -1);
                    }
                    if (written(in_hi, k)) {
                        sync_hi[k] = join_reaching(in_hi, k, -1);
                    }
                }
            }
            // Each web's type: a single (fpr32) if it only ever covers the low half as a word, otherwise a doubleword
            // (fpr) whose halves the word accesses name (the odd register's as u32h). Only written or only loaded.
            enum : uint8_t { WIDTH32 = 1, WIDTH64 = 2, ARITH = 4, LOADED = 8, WRITTEN = 16, READ = 32, HIGH = 64 };
            std::vector<uint8_t> web_flags(sites, 0);
            for (size_t s = 0; s < sites; s++) {
                web_flags[find((int)s)] |= site_loaded[s] ? LOADED : WRITTEN;
            }
            for (size_t i = 0; i < n_instr; i++) {
                for (const FprAccess& a : accesses[i]) {
                    bool odd = a.reg == reg + 1;
                    if (a.reg != reg && !odd) {
                        continue;
                    }
                    int site = a.def ? site_of_instr[i] : odd ? odd_use_site[i] : use_site[i];
                    if (site < 0) {
                        continue;
                    }
                    uint8_t& f = web_flags[find(site)];
                    f |= odd ? HIGH : wide(a) ? WIDTH64 : WIDTH32;
                    f |= (a.view == 'S' || a.view == 'D') ? ARITH : 0;
                    f |= a.def ? 0 : READ;
                }
            }
            std::vector<int> web_index(sites, -1);
            for (size_t s = 0; s < sites; s++) {
                if (find((int)s) != (int)s) {
                    continue;
                }
                uint8_t f = web_flags[s];
                bool pure = (f & LOADED) == 0 || (f & WRITTEN) == 0;
                bool useful = (f & WRITTEN) || (f & READ);
                if ((f & (WIDTH32 | WIDTH64 | HIGH)) && pure && useful) {
                    bool is64 = (f & (WIDTH64 | HIGH)) != 0;
                    web_index[s] = (int)plan.fpr_webs.size();
                    plan.fpr_webs.push_back({ reg, is64, (f & ARITH) != 0, is64 && (f & (WIDTH32 | HIGH)) != 0 });
                }
            }
            auto web_of = [&](int site) { return site < 0 ? -1 : web_index[find(site)]; };
            for (size_t i = 0; i < n_instr; i++) {
                if ((def[i] | odd_def[i]) & bit) {
                    int w = web_of(site_of_instr[i]);
                    if (w >= 0) {
                        plan.fpr_def_web[(uint32_t)i * 32 + reg + ((odd_def[i] & bit) ? 1 : 0)] = w;
                    }
                }
                if (use[i] & bit) {
                    int w = web_of(use_site[i]);
                    if (w >= 0) {
                        plan.fpr_use_web[(uint32_t)i * 32 + reg] = w;
                    }
                }
                if (odd_use[i] & bit) {
                    int w = web_of(odd_use_site[i]);
                    if (w >= 0) {
                        plan.fpr_use_web[(uint32_t)i * 32 + reg + 1] = w;
                    }
                }
            }
            // Loads: at entry and after calls, for loaded webs something reads.
            auto loaded_read = [&](int site) {
                int w = web_of(site);
                return w >= 0 && (web_flags[find(site)] & READ) ? w : -1;
            };
            if (int w = loaded_read(0); w >= 0) {
                plan.fpr_entry.push_back(w);
            }
            for (size_t k = 0; k < count; k++) {
                const RegNode& n = g.nodes[k];
                if (!reached[k]) {
                    continue;
                }
                if (n.kind == RegNode::Kind::Call) {
                    if (int w = loaded_read(site_of_node[k]); w >= 0) {
                        plan.calls[n.key].fpr_reload.push_back(w);
                    }
                }
                // The low half's web writes back its view (all of a doubleword web, whose write also reached the high
                // half); a different web for the high half (a doubleword since overwritten low) writes back that half.
                int w_lo = web_of(sync_lo[k]);
                int w_hi = web_of(sync_hi[k]);
                std::vector<int> wb;
                if (w_lo >= 0) {
                    wb.push_back(w_lo);
                }
                if (w_hi >= 0 && w_hi != w_lo) {
                    wb.push_back(w_hi | N64Recomp::FunctionRegPlan::FPR_HIGH_HALF);
                }
                if (wb.empty()) {
                    continue;
                }
                auto& list = n.kind == RegNode::Kind::Exit ? (n.key == SIZE_MAX ? plan.fpr_end : plan.fpr_exits[n.key]) :
                             plan.calls[n.key].fpr_writeback;
                list.insert(list.end(), wb.begin(), wb.end());
            }
        }
        // A web stays a local only where that saves context traffic: its loads and writebacks must be fewer than
        // the context reads and writes its instructions would do. Each web keeps the context right at every sync
        // for what it owns, so dropping one leaves the others exact.
        size_t n_webs = plan.fpr_webs.size();
        if (n_webs == 0) {
            return;
        }
        std::vector<int> cost(n_webs, 0), saved(n_webs, 0);
        auto count_list = [&](const std::vector<int>& list) {
            for (int e : list) {
                cost[e & ~N64Recomp::FunctionRegPlan::FPR_HIGH_HALF]++;
            }
        };
        count_list(plan.fpr_entry);
        count_list(plan.fpr_end);
        for (const auto& [key, list] : plan.fpr_exits) {
            count_list(list);
        }
        for (const auto& [key, sync] : plan.calls) {
            count_list(sync.fpr_writeback);
            count_list(sync.fpr_reload);
        }
        for (const auto& [key, w] : plan.fpr_def_web) {
            saved[w]++;
        }
        for (const auto& [key, w] : plan.fpr_use_web) {
            saved[w]++;
        }
        std::vector<int> renumber(n_webs, -1);
        std::vector<N64Recomp::FunctionRegPlan::FprWeb> kept;
        for (size_t w = 0; w < n_webs; w++) {
            if (cost[w] < saved[w]) {
                renumber[w] = (int)kept.size();
                kept.push_back(plan.fpr_webs[w]);
            }
        }
        plan.fpr_webs = std::move(kept);
        auto remap_map = [&](std::unordered_map<uint32_t, int>& map) {
            for (auto it = map.begin(); it != map.end();) {
                if (renumber[it->second] < 0) {
                    it = map.erase(it);
                }
                else {
                    it->second = renumber[it->second];
                    ++it;
                }
            }
        };
        auto remap_list = [&](std::vector<int>& list) {
            std::vector<int> out;
            for (int e : list) {
                int w = renumber[e & ~N64Recomp::FunctionRegPlan::FPR_HIGH_HALF];
                if (w >= 0) {
                    out.push_back(w | (e & N64Recomp::FunctionRegPlan::FPR_HIGH_HALF));
                }
            }
            list = std::move(out);
        };
        remap_map(plan.fpr_def_web);
        remap_map(plan.fpr_use_web);
        remap_list(plan.fpr_entry);
        remap_list(plan.fpr_end);
        for (auto& [key, list] : plan.fpr_exits) {
            remap_list(list);
        }
        for (auto& [key, sync] : plan.calls) {
            remap_list(sync.fpr_writeback);
            remap_list(sync.fpr_reload);
        }
    }

    // codegen.fpr_abi_sync: FPR masks (bit n for f<n>). o32: arguments in f12-f15, results in f0-f3, f20-f31 callee-saved.
    constexpr uint32_t FPR_ARGS = 0xFu << 12;
    constexpr uint32_t FPR_RET = 0xFu;
    constexpr uint32_t FPR_CS = 0xFFFu << 20;
    constexpr uint32_t FPR_CALLER = ~FPR_CS;

    // The FPRs an instruction reads and writes as 32-bit registers (a double or doubleword is the even/odd pair).
    void fpr_effects(const rabbitizer::InstructionCpu& instr, uint32_t& use, uint32_t& def) {
        std::vector<FprAccess> acc;
        fpr_accesses(instr, instr.getUniqueId(), acc);
        use = def = 0;
        for (const auto& a : acc) {
            uint32_t m;
            switch (a.view) {
                case 'D': case 'U': m = 3u << (a.reg & ~1); break;
                case 'H': m = 1u << (a.reg | 1); break;
                default: m = 1u << a.reg; break;
            }
            (a.def ? def : use) |= m;
        }
    }

    // A call's FPR value (values per function, indirect for indirect calls): runtime code reads the argument
    // registers and is relied on for nothing it writes. So is a static function, compiled C that keeps the o32 ABI
    // (and reads only results and callee-saved FPRs after its own calls); a function the analysis couldn't follow
    // may read anything.
    uint32_t fpr_call_value(const N64Recomp::Context& context, const RegNode& node, const std::vector<uint32_t>& values, uint32_t indirect,
                            uint32_t runtime, uint32_t unknown) {
        auto of = [&](size_t c) -> uint32_t {
            if (runtime_function(context, c)) {
                // The libultra functions the runtime reimplements take and return no floats; a native hook is
                // C written against the context (collision.c reads f12 and f14).
                const auto& func = context.functions[c];
                return func.reimplemented || func.stubbed || func.ignored || func.words.empty() ? 0 : runtime;
            }
            return c < values.size() && context.reg_summaries[c].analyzed ? values[c] : unknown;
        };
        if (node.callee >= 0) {
            return of((size_t)node.callee);
        }
        if (node.indirect) {
            return indirect;
        }
        auto it = context.functions_by_vram.find(node.target);
        if (node.target == 0 || it == context.functions_by_vram.end()) {
            return runtime;
        }
        uint32_t m = unknown == 0 ? 0xFFFFFFFFu : 0;
        for (size_t c : it->second) {
            m = unknown == 0 ? (m & of(c)) : (m | of(c));
        }
        return m;
    }

    uint32_t fpr_call_needs(const N64Recomp::Context& context, const RegNode& node) {
        return fpr_call_value(context, node, context.regcache_fpr_needs, context.regcache_indirect_fpr_needs, FPR_ARGS, 0xFFFFFFFFu);
    }

    uint32_t fpr_call_kill(const N64Recomp::Context& context, const RegNode& node) {
        return fpr_call_value(context, node, context.regcache_fpr_must_def, context.regcache_indirect_fpr_must_def, 0, 0);
    }

    // FPR liveness over a function's graph: calls read their callees' needs and end what they certainly write;
    // returns (and tail calls, past their callee) read exit_read.
    void fpr_liveness(const N64Recomp::Context& context, const RegGraph& g, const std::vector<uint32_t>& use, const std::vector<uint32_t>& def,
                      uint32_t exit_read, std::vector<uint32_t>& live_in, std::vector<uint32_t>& live_out) {
        size_t n = g.nodes.size();
        live_in.assign(n, 0);
        live_out.assign(n, 0);
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t k = n; k-- > 0;) {
                const RegNode& node = g.nodes[k];
                uint32_t out = 0;
                for (int s : node.succs) {
                    out |= live_in[s];
                }
                uint32_t in = 0;
                switch (node.kind) {
                    case RegNode::Kind::Plain: in = use[k] | (out & ~def[k]); break;
                    case RegNode::Kind::Call: in = fpr_call_needs(context, node) | (out & ~fpr_call_kill(context, node)); break;
                    case RegNode::Kind::TailCall: in = fpr_call_needs(context, node) | exit_read; break;
                    case RegNode::Kind::Barrier: in = 0xFFFFFFFFu; break;
                    case RegNode::Kind::Exit: in = exit_read; break;
                }
                if (in != live_in[k] || out != live_out[k]) {
                    live_in[k] = in;
                    live_out[k] = out;
                    changed = true;
                }
            }
        }
    }

    // The caller-saved FPRs written on every path from the function's entry to each of its returns.
    uint32_t fpr_must_def_of(const N64Recomp::Context& context, const RegGraph& g, const std::vector<uint32_t>& def) {
        size_t count = g.nodes.size();
        std::vector<std::vector<int>> preds(count);
        std::vector<bool> reached(count, false);
        std::vector<int> work{ 0 };
        reached[0] = true;
        while (!work.empty()) {
            int k = work.back();
            work.pop_back();
            for (int s : g.nodes[k].succs) {
                preds[s].push_back(k);
                if (!reached[s]) {
                    reached[s] = true;
                    work.push_back(s);
                }
            }
        }
        auto gen = [&](size_t k) -> uint32_t {
            const RegNode& node = g.nodes[k];
            switch (node.kind) {
                case RegNode::Kind::Plain: return def[k];
                case RegNode::Kind::Call: case RegNode::Kind::TailCall: return fpr_call_kill(context, node);
                default: return 0;
            }
        };
        std::vector<uint32_t> out(count, 0xFFFFFFFFu);
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t k = 0; k < count; k++) {
                if (!reached[k]) {
                    continue;
                }
                uint32_t in = k == 0 ? 0 : 0xFFFFFFFFu;
                for (int p : preds[k]) {
                    in &= out[p];
                }
                uint32_t o = in | gen(k);
                if (o != out[k]) {
                    out[k] = o;
                    changed = true;
                }
            }
        }
        uint32_t result = 0xFFFFFFFFu;
        for (size_t k = 0; k < count; k++) {
            if (reached[k] && g.nodes[k].kind == RegNode::Kind::Exit) {
                uint32_t in = preds[k].empty() ? 0 : 0xFFFFFFFFu;
                for (int p : preds[k]) {
                    in &= out[p];
                }
                result &= in;
            }
        }
        return result & FPR_CALLER;
    }

    void fpr_node_effects(const RegGraph& g, const std::vector<rabbitizer::InstructionCpu>& instructions, std::vector<uint32_t>& use, std::vector<uint32_t>& def) {
        use.assign(g.nodes.size(), 0);
        def.assign(g.nodes.size(), 0);
        for (size_t k = 0; k < g.nodes.size(); k++) {
            const RegNode& node = g.nodes[k];
            if (node.kind == RegNode::Kind::Plain && node.instr != SIZE_MAX) {
                fpr_effects(instructions[node.instr], use[k], def[k]);
            }
        }
    }

    // codegen.fpr_abi_sync: drops FPR web writebacks nothing can read. Before a call: what the callee side reads,
    // the callee-saved FPRs (their stack saves read the context), and what is still read after the call that the
    // callee doesn't certainly write. At a return: f0-f3, the callee-saved FPRs and what callers read.
    void trim_fpr_syncs(const N64Recomp::Context& context, size_t func_index, const RegGraph& g,
                        const std::vector<rabbitizer::InstructionCpu>& instructions, N64Recomp::FunctionRegPlan& plan) {
        std::vector<uint32_t> use, def, live_in, live_out;
        fpr_node_effects(g, instructions, use, def);
        uint32_t exit_read = FPR_RET | FPR_CS |
            (func_index < context.regcache_fpr_exit_sync.size() ? context.regcache_fpr_exit_sync[func_index] : FPR_CALLER);
        fpr_liveness(context, g, use, def, exit_read, live_in, live_out);
        auto covers = [&](int entry) -> uint32_t {
            const auto& web = plan.fpr_webs[entry & ~N64Recomp::FunctionRegPlan::FPR_HIGH_HALF];
            if (entry & N64Recomp::FunctionRegPlan::FPR_HIGH_HALF) {
                return 2u << web.reg;
            }
            return web.is64 ? (3u << web.reg) : (1u << web.reg);
        };
        auto filter = [&](std::vector<int>& list, uint32_t needed) {
            list.erase(std::remove_if(list.begin(), list.end(), [&](int e) { return (covers(e) & needed) == 0; }), list.end());
        };
        std::unordered_map<size_t, uint32_t> needed;
        for (size_t k = 0; k < g.nodes.size(); k++) {
            const RegNode& node = g.nodes[k];
            if (node.kind == RegNode::Kind::Call) {
                needed[node.key] |= fpr_call_needs(context, node) | FPR_CS | (live_out[k] & ~fpr_call_kill(context, node));
            }
            else if (node.kind == RegNode::Kind::TailCall) {
                needed[node.key] |= live_in[k] | FPR_CS;
            }
            else if (node.kind == RegNode::Kind::Barrier) {
                needed[node.key] = 0xFFFFFFFFu;
            }
        }
        for (auto& [key, sync] : plan.calls) {
            auto it = needed.find(key);
            if (it != needed.end()) {
                filter(sync.fpr_writeback, it->second);
            }
        }
        for (auto& [key, list] : plan.fpr_exits) {
            filter(list, exit_read);
        }
        filter(plan.fpr_end, exit_read);
    }

    // What a call reads: what its callee's code (or code it calls) reads before writing.
    GprMask call_needs(const N64Recomp::Context& context, const RegNode& node) {
        if (node.callee >= 0) {
            return (size_t)node.callee < context.regcache_entry_needs.size() ? context.regcache_entry_needs[node.callee] : 0;
        }
        return node.indirect ? context.regcache_indirect_needs : 0;
    }

    // Functions an indirect call can reach: those whose start address is a word of the ROM (a pointer in data) or is
    // built in code by lui and addiu/ori. Functions of different overlays at one address are all counted.
    std::vector<bool> address_taken(const N64Recomp::Context& context) {
        size_t count = context.functions.size();
        std::unordered_map<uint32_t, std::vector<size_t>> by_vram;
        for (size_t i = 0; i < count; i++) {
            by_vram[context.functions[i].vram].push_back(i);
        }
        std::vector<bool> taken(count, false);
        auto mark = [&](uint32_t addr) {
            auto it = by_vram.find(addr);
            if (it != by_vram.end()) {
                for (size_t i : it->second) {
                    taken[i] = true;
                }
            }
        };
        const auto& rom = context.rom;
        for (size_t off = 0; off + 4 <= rom.size(); off += 4) {
            mark((uint32_t)rom[off] << 24 | (uint32_t)rom[off + 1] << 16 | (uint32_t)rom[off + 2] << 8 | rom[off + 3]);
        }
        for (const auto& func : context.functions) {
            uint32_t hi[32] = {};
            for (uint32_t w : func.words) {
                w = byteswap(w);
                uint32_t op = w >> 26, rs = (w >> 21) & 31, rt = (w >> 16) & 31, imm = w & 0xFFFF;
                if (op == 0x0F) {
                    hi[rt] = imm << 16;
                } else if (op == 0x09) {
                    mark(hi[rs] + (uint32_t)(int32_t)(int16_t)imm);
                } else if (op == 0x0D) {
                    mark(hi[rs] | imm);
                }
            }
        }
        return taken;
    }

    // What the function's own instructions and the code it calls read: liveness with calls reading their callees'
    // needs and killing the caller-saved registers, and nothing read at exits or by the sync protocol, so the result
    // doesn't depend on the protocol itself. A register passed through one call to a later one is live after the first.
    void genuine_liveness(const N64Recomp::Context& context, const RegGraph& g, std::vector<GprMask>& live_in, std::vector<GprMask>& live_out) {
        size_t count = g.nodes.size();
        live_in.assign(count, 0);
        live_out.assign(count, 0);
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t k = count; k-- > 0;) {
                const RegNode& node = g.nodes[k];
                GprMask out = 0;
                for (int s : node.succs) {
                    out |= live_in[s];
                }
                GprMask in;
                switch (node.kind) {
                    case RegNode::Kind::Plain: in = node.use | (out & ~node.def); break;
                    case RegNode::Kind::Call: in = node.use | call_needs(context, node) | (out & ~node.call_kill); break;
                    case RegNode::Kind::TailCall: in = node.use | call_needs(context, node); break;
                    default: in = 0; break;
                }
                if (out != live_out[k] || in != live_in[k]) {
                    live_out[k] = out;
                    live_in[k] = in;
                    changed = true;
                }
            }
        }
    }

    GprMask temps_read_after(const RegNode& node, const std::vector<GprMask>& live_out, size_t id) {
        if (node.kind != RegNode::Kind::Call) {
            return 0;
        }
        return live_out[id] & CALLER_SAVED & ~RETURN_REGS;
    }

    void summarize(const N64Recomp::Context& context, const RegGraph& g, N64Recomp::FunctionRegSummary& summary) {
        std::vector<GprMask> live_in, live_out;
        genuine_liveness(context, g, live_in, live_out);
        summary.calls = g.calls;
        summary.live_in = live_in[0];
        summary.temps_at_entry = live_in[0] & CALLER_SAVED & ~ARG_REGS;
        summary.temps_after_call = 0;
        summary.callees.clear();
        summary.after_call_sites.clear();
        for (size_t id = 0; id < g.nodes.size(); id++) {
            const RegNode& node = g.nodes[id];
            if (node.kind != RegNode::Kind::Call && node.kind != RegNode::Kind::TailCall) {
                continue;
            }
            if (node.callee >= 0) {
                summary.callees.push_back((size_t)node.callee);
            }
            GprMask after = temps_read_after(node, live_out, id);
            if (after) {
                summary.temps_after_call |= after;
                // After a call to an unknown target that isn't indirect (a static function, an event) everything
                // gets reloaded and nothing cached can have written the registers, so there's no callee to tell.
                if (node.indirect || node.callee >= 0) {
                    summary.after_call_sites.emplace_back(node.indirect ? -1 : node.callee, after);
                }
            }
        }
    }
}

// Analyzes every function's registers: whether it preserves the callee-saved registers, and which caller-saved
// registers anything reads at entry or after a call (those get synced like arguments). Writes the report if asked.
bool N64Recomp::analyze_registers(Context& context) {
    size_t count = context.functions.size();
    context.reg_summaries.assign(count, FunctionRegSummary{});
    std::vector<std::vector<rabbitizer::InstructionCpu>> views(count);
    std::vector<FunctionStats> stats(count);
    std::vector<std::unordered_set<uint32_t>> jtbl_lw(count);
    // A function whose first instruction's hook returns is native code (runtime C) that keeps the o32 ABI.
    std::vector<bool> native(count, false);
    // First the callee-saved checks, which only depend on each function's own code.
    for (size_t i = 0; i < count; i++) {
        const Function& func = context.functions[i];
        FunctionRegSummary& summary = context.reg_summaries[i];
        if (func.ignored || func.reimplemented || func.stubbed || func.words.empty()) {
            continue;
        }
        auto hook0 = func.function_hooks.find(0);
        if (hook0 != func.function_hooks.end() && hook0->second.find("return;") != std::string::npos) {
            native[i] = true;
            summary.analyzed = true;
            summary.preserves = true;
            summary.live_in = ARG_REGS | gpr_bit(29);
            continue;
        }
        if (!load_function_view(context, func, views[i], stats[i], jtbl_lw[i])) {
            summary.failure = "analyze_function failed";
            continue;
        }
        RegGraph g;
        if (!build_reg_graph(context, func, views[i], stats[i], jtbl_lw[i], 0, g)) {
            summary.failure = g.failure;
            continue;
        }
        summary.analyzed = true;
        summary.preserves = check_preserves(g, views[i], summary.preserve_failure);
    }
    // Then what each function reads of the caller-saved registers, now that callers know which callees preserve.
    // A call reads what its callee needs, so this goes round until no function's needs grow.
    std::vector<RegGraph> graphs(count);
    for (size_t i = 0; i < count; i++) {
        if (context.reg_summaries[i].analyzed && !native[i]) {
            build_reg_graph(context, context.functions[i], views[i], stats[i], jtbl_lw[i], 0, graphs[i]);
        }
    }
    std::vector<bool> taken = address_taken(context);
    // What each function certainly writes, so the liveness below knows which values can come through its calls.
    context.regcache_must_def.assign(count, CALLER_SAVED);
    context.regcache_indirect_must_def = CALLER_SAVED;
    for (bool changed = true; changed;) {
        changed = false;
        GprMask indirect = CALLER_SAVED;
        for (size_t i = 0; i < count; i++) {
            if (taken[i]) {
                RegNode probe{};
                probe.callee = (int)i;
                indirect &= callee_must_def(context, probe);
            }
        }
        if (indirect != context.regcache_indirect_must_def) {
            context.regcache_indirect_must_def = indirect;
            changed = true;
        }
        for (size_t i = 0; i < count; i++) {
            if (!context.reg_summaries[i].analyzed || native[i]) {
                continue;
            }
            apply_call_kills(context, graphs[i]);
            GprMask m = must_def_of(graphs[i]);
            if (m != context.regcache_must_def[i]) {
                context.regcache_must_def[i] = m;
                changed = true;
            }
        }
    }
    // What each function or anything it calls may read or write. Hooks run arbitrary code against the context.
    std::vector<GprMask> own(count, ALL_GPRS);
    for (size_t i = 0; i < count; i++) {
        const Function& func = context.functions[i];
        if (!context.reg_summaries[i].analyzed || native[i] || !func.function_hooks.empty() || !func.return_hook.empty()) {
            continue;
        }
        own[i] = 0;
        for (const RegNode& node : graphs[i].nodes) {
            own[i] |= node.use | node.def;
        }
    }
    context.regcache_may_touch = own;
    context.regcache_indirect_may_touch = 0;
    for (bool changed = true; changed;) {
        changed = false;
        GprMask indirect = 0;
        for (size_t i = 0; i < count; i++) {
            if (taken[i]) {
                RegNode probe{};
                probe.callee = (int)i;
                indirect |= callee_may_touch(context, probe);
            }
        }
        if (indirect != context.regcache_indirect_may_touch) {
            context.regcache_indirect_may_touch = indirect;
            changed = true;
        }
        for (size_t i = 0; i < count; i++) {
            if (own[i] == ALL_GPRS) {
                continue;
            }
            GprMask m = context.regcache_may_touch[i];
            for (const RegNode& node : graphs[i].nodes) {
                if (node.kind == RegNode::Kind::Call || node.kind == RegNode::Kind::TailCall) {
                    m |= callee_may_touch(context, node);
                }
            }
            if (m != context.regcache_may_touch[i]) {
                context.regcache_may_touch[i] = m;
                changed = true;
            }
        }
    }
    for (size_t i = 0; i < count; i++) {
        if (context.reg_summaries[i].analyzed && !native[i]) {
            apply_call_kills(context, graphs[i]);
        }
    }
    // Runtime code is taken to read all four argument registers.
    context.regcache_entry_needs.assign(count, 0);
    for (size_t i = 0; i < count; i++) {
        if (runtime_function(context, i) && !context.reg_summaries[i].analyzed) {
            context.regcache_entry_needs[i] = ARG_REGS;
        }
    }
    context.regcache_indirect_needs = 0;
    for (bool changed = true; changed;) {
        changed = false;
        GprMask indirect = 0;
        for (size_t i = 0; i < count; i++) {
            FunctionRegSummary& summary = context.reg_summaries[i];
            if (!summary.analyzed) {
                indirect |= taken[i] ? context.regcache_entry_needs[i] : 0;
                continue;
            }
            if (!native[i]) {
                summarize(context, graphs[i], summary);
            }
            GprMask needs = summary.live_in & CALLER_SAVED;
            if (needs != context.regcache_entry_needs[i]) {
                context.regcache_entry_needs[i] = needs;
                changed = true;
            }
            indirect |= taken[i] ? needs : 0;
        }
        if (indirect != context.regcache_indirect_needs) {
            context.regcache_indirect_needs = indirect;
            changed = true;
        }
    }
    // codegen.skip_saves: the callee-saved registers each function reads for their value (stores of the entry value
    // to a slot only its restores read don't count), and those it may return changed.
    size_t unknown_targets = 0;
    std::vector<std::string> save_why(count);
    if (context.codegen.skip_saves) {
        std::vector<GprMask> written(count, 0);
        std::vector<std::vector<GprMask>> uses(count);
        for (size_t i = 0; i < count; i++) {
            if (!context.reg_summaries[i].analyzed || native[i]) {
                continue;
            }
            SaveSlots slots;
            find_saves(graphs[i], views[i], slots);
            save_why[i] = slots.why;
            uses[i].resize(graphs[i].nodes.size());
            for (size_t k = 0; k < graphs[i].nodes.size(); k++) {
                const RegNode& node = graphs[i].nodes[k];
                uses[i][k] = node.use;
                if (node.kind == RegNode::Kind::Plain) {
                    written[i] |= node.def & CALLEE_SAVED;
                    for (int reg = 16; reg < 32; reg++) {
                        const auto& s = slots.saves[reg];
                        if ((slots.regs & gpr_bit(reg)) && std::find(s.begin(), s.end(), node.instr) != s.end()) {
                            uses[i][k] &= ~gpr_bit(reg);
                        }
                    }
                }
                else if ((node.kind == RegNode::Kind::Call || node.kind == RegNode::Kind::TailCall) && node.callee < 0 && !node.indirect) {
                    unknown_targets++;
                    save_why[i] += fmt::format(" unknown call target at 0x{:08X}", context.functions[i].vram + (uint32_t)node.key * 4);
                }
            }
        }
        context.regcache_cs_needs.assign(count, 0);
        context.regcache_indirect_cs_needs = 0;
        auto cs_call_needs = [&](const RegNode& node) -> GprMask {
            return cs_call_value(context, node, context.regcache_cs_needs, context.regcache_indirect_cs_needs);
        };
        for (bool changed = true; changed;) {
            changed = false;
            GprMask indirect = 0;
            for (size_t i = 0; i < count; i++) {
                indirect |= taken[i] ? context.regcache_cs_needs[i] : 0;
            }
            if (indirect != context.regcache_indirect_cs_needs) {
                context.regcache_indirect_cs_needs = indirect;
                changed = true;
            }
            for (size_t i = 0; i < count; i++) {
                if (uses[i].empty()) {
                    continue;
                }
                const RegGraph& g = graphs[i];
                size_t n = g.nodes.size();
                std::vector<GprMask> live(n, 0);
                for (bool again = true; again;) {
                    again = false;
                    for (size_t k = n; k-- > 0;) {
                        const RegNode& node = g.nodes[k];
                        GprMask out = 0;
                        for (int s : node.succs) {
                            out |= live[s];
                        }
                        GprMask in = 0;
                        switch (node.kind) {
                            case RegNode::Kind::Plain: in = uses[i][k] | (out & ~node.def); break;
                            case RegNode::Kind::Call: case RegNode::Kind::Barrier: in = uses[i][k] | cs_call_needs(node) | out; break;
                            case RegNode::Kind::TailCall: in = uses[i][k] | cs_call_needs(node); break;
                            case RegNode::Kind::Exit: in = 0; break;
                        }
                        in &= CALLEE_SAVED;
                        if (in != live[k]) {
                            live[k] = in;
                            again = true;
                        }
                    }
                }
                if (live[0] != context.regcache_cs_needs[i]) {
                    context.regcache_cs_needs[i] = live[0];
                    changed = true;
                }
            }
        }
        // A function that doesn't preserve a callee-saved register it writes returns it changed; so does one whose
        // callee returns a register changed that the function doesn't save and restore itself (a tail call's
        // callee returns straight to the caller).
        context.regcache_cs_clobber.assign(count, 0);
        context.regcache_indirect_cs_clobber = 0;
        auto cs_call_clobber = [&](const RegNode& node) -> GprMask {
            return cs_call_value(context, node, context.regcache_cs_clobber, context.regcache_indirect_cs_clobber);
        };
        for (bool changed = true; changed;) {
            changed = false;
            GprMask indirect = 0;
            for (size_t i = 0; i < count; i++) {
                indirect |= taken[i] ? context.regcache_cs_clobber[i] : 0;
            }
            if (indirect != context.regcache_indirect_cs_clobber) {
                context.regcache_indirect_cs_clobber = indirect;
                changed = true;
            }
            for (size_t i = 0; i < count; i++) {
                if (uses[i].empty()) {
                    continue;
                }
                GprMask m = context.reg_summaries[i].preserves ? 0 : written[i];
                for (const RegNode& node : graphs[i].nodes) {
                    if (node.kind == RegNode::Kind::Call || node.kind == RegNode::Kind::Barrier) {
                        m |= cs_call_clobber(node) & ~written[i];
                    }
                    else if (node.kind == RegNode::Kind::TailCall) {
                        m |= cs_call_clobber(node);
                    }
                }
                if (m != context.regcache_cs_clobber[i]) {
                    context.regcache_cs_clobber[i] = m;
                    changed = true;
                }
            }
        }
    }
    GprMask extra = 0;
    for (size_t i = 0; i < count; i++) {
        const FunctionRegSummary& summary = context.reg_summaries[i];
        extra |= summary.analyzed ? (summary.temps_at_entry | summary.temps_after_call) : 0;
    }
    context.regcache_extra_sync = extra;
    // Before a call to f: what f or anything it calls reads at entry. Before an indirect call: anything's.
    context.regcache_call_sync.assign(count, 0);
    context.regcache_indirect_sync = 0;
    for (size_t i = 0; i < count; i++) {
        context.regcache_call_sync[i] = context.reg_summaries[i].temps_at_entry;
        context.regcache_indirect_sync |= context.reg_summaries[i].temps_at_entry;
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i = 0; i < count; i++) {
            GprMask m = context.regcache_call_sync[i];
            for (size_t callee : context.reg_summaries[i].callees) {
                m |= context.regcache_call_sync[callee];
            }
            if (m != context.regcache_call_sync[i]) {
                context.regcache_call_sync[i] = m;
                changed = true;
            }
        }
    }
    // At f's returns: what a caller reads after calling f, or something that calls f.
    context.regcache_exit_sync.assign(count, 0);
    for (size_t i = 0; i < count; i++) {
        for (const auto& [callee, regs] : context.reg_summaries[i].after_call_sites) {
            if (callee < 0) {
                for (auto& m : context.regcache_exit_sync) {
                    m |= regs;
                }
                continue;
            }
            std::vector<size_t> work{ (size_t)callee };
            std::vector<bool> visited(count, false);
            visited[callee] = true;
            while (!work.empty()) {
                size_t f = work.back();
                work.pop_back();
                context.regcache_exit_sync[f] |= regs;
                for (size_t next : context.reg_summaries[f].callees) {
                    if (!visited[next]) {
                        visited[next] = true;
                        work.push_back(next);
                    }
                }
            }
        }
    }
    // codegen.ret_trim: v0/v1 as some caller may read them after a call. A function's returns read what its callers
    // read, so a callee's value that the function returns unchanged counts too; a tail callee returns straight to
    // the function's callers; an indirect call that reads them can reach any function whose address is taken.
    // The runtime only calls thread entries, whose return value it ignores.
    // Static functions (code the symbols don't name, found while recompiling) aren't analyzed: what their jal or
    // jalr can reach may have anything read after the call. Each is scanned up to the next known function.
    std::vector<bool> static_called(count, false);
    if (context.codegen.ret_trim || context.codegen.fpr_abi_sync) {
        std::set<uint32_t> static_targets;
        for (size_t i = 0; i < count; i++) {
            for (const RegNode& node : graphs[i].nodes) {
                if ((node.kind == RegNode::Kind::Call || node.kind == RegNode::Kind::TailCall) && node.callee < 0 && !node.indirect &&
                    node.target != 0 && !context.functions_by_vram.contains(node.target)) {
                    static_targets.insert(node.target);
                }
            }
        }
        for (uint32_t start : static_targets) {
            const N64Recomp::Section* section = nullptr;
            for (const auto& s : context.sections) {
                if (start >= s.ram_addr && start < s.ram_addr + s.size && s.rom_addr + s.size <= context.rom.size()) {
                    section = &s;
                }
            }
            if (section == nullptr) {
                continue;
            }
            uint32_t end = std::min(section->ram_addr + section->size, start + 0x2000u);
            for (const auto& func : context.functions) {
                if (func.vram > start && func.vram < end) {
                    end = func.vram;
                }
            }
            for (uint32_t vram = start; vram + 4 <= end; vram += 4) {
                const uint8_t* p = &context.rom[section->rom_addr + (vram - section->ram_addr)];
                uint32_t w = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
                if (w >> 26 == 3) {
                    uint32_t target = (vram & 0xF0000000u) | ((w & 0x3FFFFFFu) << 2);
                    auto f = context.functions_by_vram.find(target);
                    if (f != context.functions_by_vram.end()) {
                        for (size_t c : f->second) {
                            static_called[c] = true;
                        }
                    }
                }
                else if (w >> 26 == 0 && (w & 0x3F) == 9) {
                    for (size_t t = 0; t < count; t++) {
                        static_called[t] = static_called[t] || taken[t];
                    }
                }
            }
        }
    }
    if (context.codegen.ret_trim) {
        context.regcache_ret_read.assign(count, 0);
        for (size_t c = 0; c < count; c++) {
            context.regcache_ret_read[c] = static_called[c] ? RETURN_REGS : 0;
        }
        for (bool changed = true; changed;) {
            changed = false;
            auto mark = [&](size_t f, GprMask regs) {
                if ((context.regcache_ret_read[f] | regs) != context.regcache_ret_read[f]) {
                    context.regcache_ret_read[f] |= regs;
                    changed = true;
                }
            };
            for (size_t i = 0; i < count; i++) {
                if (!context.reg_summaries[i].analyzed || native[i]) {
                    continue;
                }
                const RegGraph& g = graphs[i];
                size_t n = g.nodes.size();
                GprMask exit_read = context.regcache_ret_read[i];
                std::vector<GprMask> live(n, 0), live_out(n, 0);
                for (bool again = true; again;) {
                    again = false;
                    for (size_t k = n; k-- > 0;) {
                        const RegNode& node = g.nodes[k];
                        GprMask out = 0;
                        for (int s : node.succs) {
                            out |= live[s];
                        }
                        GprMask in = 0;
                        switch (node.kind) {
                            case RegNode::Kind::Plain: in = node.use | (out & ~node.def); break;
                            case RegNode::Kind::Call: in = node.use | call_needs(context, node) | (out & ~node.call_kill); break;
                            case RegNode::Kind::TailCall: in = node.use | call_needs(context, node); break;
                            case RegNode::Kind::Barrier: in = 0; break;
                            case RegNode::Kind::Exit: in = exit_read; break;
                        }
                        if (in != live[k] || out != live_out[k]) {
                            live[k] = in;
                            live_out[k] = out;
                            again = true;
                        }
                    }
                }
                for (size_t k = 0; k < n; k++) {
                    const RegNode& node = g.nodes[k];
                    GprMask regs = 0;
                    if (node.kind == RegNode::Kind::Call) {
                        regs = live_out[k] & RETURN_REGS;
                    }
                    else if (node.kind == RegNode::Kind::TailCall) {
                        regs = exit_read;
                    }
                    if (!regs) {
                        continue;
                    }
                    if (node.callee >= 0) {
                        mark((size_t)node.callee, regs);
                    }
                    else if (node.indirect) {
                        for (size_t t = 0; t < count; t++) {
                            if (taken[t]) {
                                mark(t, regs);
                            }
                        }
                    }
                    else {
                        auto it = context.functions_by_vram.find(node.target);
                        if (it != context.functions_by_vram.end()) {
                            for (size_t c : it->second) {
                                mark(c, regs);
                            }
                        }
                    }
                }
            }
        }
    }
    // codegen.fpr_abi_sync: the FPR counterparts of the entry needs, must-define sets and exit syncs above.
    std::vector<std::string> fpr_after_sites;
    if (context.codegen.fpr_abi_sync) {
        std::vector<std::vector<uint32_t>> fuse(count), fdef(count);
        for (size_t i = 0; i < count; i++) {
            if (context.reg_summaries[i].analyzed && !native[i]) {
                fpr_node_effects(graphs[i], views[i], fuse[i], fdef[i]);
            }
        }
        context.regcache_fpr_must_def.assign(count, FPR_CALLER);
        context.regcache_indirect_fpr_must_def = FPR_CALLER;
        for (bool changed = true; changed;) {
            changed = false;
            uint32_t indirect = FPR_CALLER;
            for (size_t i = 0; i < count; i++) {
                if (taken[i]) {
                    RegNode probe{};
                    probe.callee = (int)i;
                    indirect &= fpr_call_kill(context, probe);
                }
            }
            if (indirect != context.regcache_indirect_fpr_must_def) {
                context.regcache_indirect_fpr_must_def = indirect;
                changed = true;
            }
            for (size_t i = 0; i < count; i++) {
                if (fuse[i].empty()) {
                    continue;
                }
                uint32_t m = fpr_must_def_of(context, graphs[i], fdef[i]);
                if (m != context.regcache_fpr_must_def[i]) {
                    context.regcache_fpr_must_def[i] = m;
                    changed = true;
                }
            }
        }
        context.regcache_fpr_needs.assign(count, 0);
        context.regcache_indirect_fpr_needs = 0;
        std::vector<uint32_t> live_in, live_out;
        for (bool changed = true; changed;) {
            changed = false;
            uint32_t indirect = 0;
            for (size_t i = 0; i < count; i++) {
                if (taken[i]) {
                    RegNode probe{};
                    probe.callee = (int)i;
                    indirect |= fpr_call_needs(context, probe);
                }
            }
            if (indirect != context.regcache_indirect_fpr_needs) {
                context.regcache_indirect_fpr_needs = indirect;
                changed = true;
            }
            for (size_t i = 0; i < count; i++) {
                if (fuse[i].empty()) {
                    continue;
                }
                fpr_liveness(context, graphs[i], fuse[i], fdef[i], 0, live_in, live_out);
                if (live_in[0] != context.regcache_fpr_needs[i]) {
                    context.regcache_fpr_needs[i] = live_in[0];
                    changed = true;
                }
            }
        }
        // At f's returns: the caller-saved FPRs besides f0-f3 that a caller reads after calling f, or something
        // that returns through f.
        context.regcache_fpr_exit_sync.assign(count, 0);
        auto spread = [&](size_t callee, uint32_t regs) {
            std::vector<size_t> work{ callee };
            std::vector<bool> visited(count, false);
            visited[callee] = true;
            while (!work.empty()) {
                size_t f = work.back();
                work.pop_back();
                context.regcache_fpr_exit_sync[f] |= regs;
                for (size_t next : context.reg_summaries[f].callees) {
                    if (!visited[next]) {
                        visited[next] = true;
                        work.push_back(next);
                    }
                }
            }
        };
        for (size_t i = 0; i < count; i++) {
            if (fuse[i].empty()) {
                continue;
            }
            fpr_liveness(context, graphs[i], fuse[i], fdef[i], 0, live_in, live_out);
            for (size_t k = 0; k < graphs[i].nodes.size(); k++) {
                const RegNode& node = graphs[i].nodes[k];
                uint32_t after = node.kind == RegNode::Kind::Call ? live_out[k] & FPR_CALLER & ~FPR_RET : 0;
                if (!after) {
                    continue;
                }
                fpr_after_sites.push_back(fmt::format("{} 0x{:08X} -> {}: 0x{:08X}", context.functions[i].name,
                    context.functions[i].vram + (uint32_t)node.key * 4, node.callee >= 0 ? context.functions[node.callee].name :
                    (node.indirect ? std::string{"(indirect)"} : fmt::format("0x{:08X}", node.target)), after));
                if (node.callee >= 0) {
                    spread((size_t)node.callee, after);
                }
                else if (node.indirect) {
                    for (size_t t = 0; t < count; t++) {
                        if (taken[t]) {
                            spread(t, after);
                        }
                    }
                }
                else {
                    auto it = context.functions_by_vram.find(node.target);
                    if (it != context.functions_by_vram.end()) {
                        for (size_t c : it->second) {
                            spread(c, after);
                        }
                    }
                }
            }
        }
    }
    context.regcache_ready = true;

    if (!context.codegen.abi_report.empty()) {
        std::ofstream report{ context.codegen.abi_report };
        auto regs = [](GprMask m) {
            std::string s;
            for (int r = 1; r < 32; r++) {
                if (m & (1u << r)) {
                    s += fmt::format(" r{}", r);
                }
            }
            return s.empty() ? std::string{" none"} : s;
        };
        size_t analyzed = 0, failed = 0, nonpreserving = 0, entry_temps = 0, call_temps = 0;
        for (size_t i = 0; i < count; i++) {
            const auto& s = context.reg_summaries[i];
            if (s.analyzed) {
                analyzed++;
                nonpreserving += s.preserves ? 0 : 1;
                entry_temps += s.temps_at_entry ? 1 : 0;
                call_temps += s.temps_after_call ? 1 : 0;
            }
            else if (!s.failure.empty()) {
                failed++;
            }
        }
        fmt::print(report, "# Register analysis (N64Recomp codegen.regcache)\n");
        fmt::print(report, "functions analyzed: {}, not analyzable: {}, not preserving callee-saved registers or sp: {}\n", analyzed, failed, nonpreserving);
        fmt::print(report, "functions reading caller-saved registers at entry: {}, after a call: {}\n", entry_temps, call_temps);
        fmt::print(report, "X (caller-saved registers any function reads at entry or after a call, synced where they can matter):{}\n",
                   regs(extra));
        fmt::print(report, "functions whose address is taken: {}; an indirect call reads:{}\n",
                   std::count(taken.begin(), taken.end(), true), regs(context.regcache_indirect_needs));
        if (context.codegen.fpr_abi_sync) {
            fmt::print(report, "fpr_abi_sync: an indirect call reads FPRs 0x{:08X} and certainly writes 0x{:08X}\n",
                       context.regcache_indirect_fpr_needs, context.regcache_indirect_fpr_must_def);
            fmt::print(report, "fpr_abi_sync: calls followed by reads of caller-saved FPRs (besides f0-f3): {}\n", fpr_after_sites.size());
            for (const auto& site : fpr_after_sites) {
                fmt::print(report, "  after {}\n", site);
            }
            for (size_t i = 0; i < count; i++) {
                if (context.reg_summaries[i].analyzed && (context.regcache_fpr_exit_sync[i] || (context.regcache_fpr_needs[i] & ~FPR_ARGS))) {
                    fmt::print(report, "  {}: FPR needs 0x{:08X}, must-def 0x{:08X}, read after calls 0x{:08X}\n", context.functions[i].name,
                               context.regcache_fpr_needs[i], context.regcache_fpr_must_def[i], context.regcache_fpr_exit_sync[i]);
                }
            }
        }
        if (context.codegen.ret_trim) {
            size_t v0 = 0, v1 = 0;
            for (size_t i = 0; i < count; i++) {
                v0 += (context.regcache_ret_read[i] & gpr_bit(2)) ? 1 : 0;
                v1 += (context.regcache_ret_read[i] & gpr_bit(3)) ? 1 : 0;
            }
            fmt::print(report, "ret_trim: functions whose v0 a caller may read: {}, v1: {} (of {})\n", v0, v1, count);
        }
        if (context.codegen.skip_saves) {
            size_t needing = 0, clobbering = 0;
            GprMask all_needs = 0, all_clobber = 0;
            for (size_t i = 0; i < count; i++) {
                needing += context.regcache_cs_needs[i] ? 1 : 0;
                clobbering += context.regcache_cs_clobber[i] ? 1 : 0;
                all_needs |= context.regcache_cs_needs[i];
                all_clobber |= context.regcache_cs_clobber[i];
            }
            fmt::print(report, "skip_saves: functions reading callee-saved registers for their value: {} ({}), returning them changed: {} ({}); "
                       "calls to unknown targets: {}; an indirect call reads:{}, returns changed:{}\n",
                       needing, regs(all_needs), clobbering, regs(all_clobber), unknown_targets,
                       regs(context.regcache_indirect_cs_needs), regs(context.regcache_indirect_cs_clobber));
            for (size_t i = 0; i < count; i++) {
                if (context.regcache_cs_needs[i] || context.regcache_cs_clobber[i] || !save_why[i].empty()) {
                    fmt::print(report, "  {}: reads{}, returns changed{};{}\n", context.functions[i].name,
                               regs(context.regcache_cs_needs[i]), regs(context.regcache_cs_clobber[i]), save_why[i]);
                }
            }
        }
        // Runtime code is taken to set v0 (callee_must_def); these are the calls that rely on it.
        std::map<std::string, std::vector<std::string>> runtime_returns;
        for (size_t i = 0; i < count; i++) {
            if (!context.reg_summaries[i].analyzed || native[i]) {
                continue;
            }
            std::vector<GprMask> live_in, live_out;
            genuine_liveness(context, graphs[i], live_in, live_out);
            for (size_t id = 0; id < graphs[i].nodes.size(); id++) {
                const RegNode& node = graphs[i].nodes[id];
                if (node.kind == RegNode::Kind::Call && node.callee >= 0 && runtime_function(context, (size_t)node.callee) &&
                    (live_out[id] & RETURN_REGS)) {
                    runtime_returns[context.functions[node.callee].name].push_back(
                        fmt::format("{} 0x{:08X}{}", context.functions[i].name, context.functions[i].vram + (uint32_t)node.key * 4,
                                    regs(live_out[id] & RETURN_REGS)));
                }
            }
        }
        fmt::print(report, "runtime functions whose v0/v1 a caller reads after the call: {}\n", runtime_returns.size());
        for (const auto& [name, sites] : runtime_returns) {
            fmt::print(report, "  {}: {} sites, e.g. {}\n", name, sites.size(), sites[0]);
        }
        fmt::print(report, "\n");
        for (size_t i = 0; i < count; i++) {
            const auto& s = context.reg_summaries[i];
            const auto& func = context.functions[i];
            if (!s.failure.empty()) {
                fmt::print(report, "{}: not analyzable: {}\n", func.name, s.failure);
            }
            if (!s.analyzed) {
                continue;
            }
            if (!s.preserves) {
                fmt::print(report, "{}: does not preserve: {}\n", func.name, s.preserve_failure);
            }
            if (s.temps_at_entry) {
                fmt::print(report, "{}: reads at entry:{}\n", func.name, regs(s.temps_at_entry));
            }
            if (s.temps_after_call) {
                fmt::print(report, "{}: reads after a call:{}\n", func.name, regs(s.temps_after_call));
            }
        }
    }
    return true;
}

// The plan for one function, from the whole-program facts gathered by analyze_registers.
static void plan_registers(const N64Recomp::Context& context, size_t func_index, const std::vector<rabbitizer::InstructionCpu>& instructions,
                           const N64Recomp::FunctionStats& stats, const std::unordered_set<uint32_t>& jtbl_lw, N64Recomp::FunctionRegPlan& plan) {
    const N64Recomp::Function& func = context.functions[func_index];
    plan = N64Recomp::FunctionRegPlan{};
    plan.vram = func.vram;
    plan.trace = context.codegen.abi_trace;
    if ((context.codegen.abi_poison || context.codegen.abi_trace) && context.regcache_ready && func_index < context.reg_summaries.size() &&
        context.reg_summaries[func_index].analyzed) {
        // Junk in every caller-saved register the analysis says nothing reads here: at entry, what neither this
        // function nor anything it calls reads at entry and no caller reads after it returns; after a call, what
        // this function doesn't read after it and no caller reads after this function returns.
        constexpr GprMask POISON_CALL = gpr_bit(1) | ARG_REGS | mask_range(8, 15) | gpr_bit(24) | gpr_bit(25);
        constexpr GprMask POISON_ENTRY = mask_range(1, 3) | mask_range(8, 15) | gpr_bit(24) | gpr_bit(25);
        GprMask keep = context.regcache_exit_sync[func_index];
        RegGraph g;
        if (build_reg_graph(context, func, instructions, stats, jtbl_lw, 0, g)) {
            apply_call_kills(context, g);
            std::vector<GprMask> genuine_in, genuine_out;
            genuine_liveness(context, g, genuine_in, genuine_out);
            plan.poison = context.codegen.abi_poison;
            plan.poison_entry = POISON_ENTRY & ~(context.regcache_call_sync[func_index] | keep);
            for (size_t id = 0; id < g.nodes.size(); id++) {
                const RegNode& node = g.nodes[id];
                if (node.kind == RegNode::Kind::Call || node.kind == RegNode::Kind::TailCall) {
                    auto& call = plan.calls[node.key];
                    call.poison = POISON_CALL & ~(genuine_out[id] | keep);
                    call.runtime = node.callee >= 0 && runtime_function(context, (size_t)node.callee);
                }
            }
        }
        return;
    }
    if (!context.regcache_ready || !regcache_wanted(context, func)) {
        return;
    }
    if (func_index >= context.reg_summaries.size() || !context.reg_summaries[func_index].analyzed) {
        // Functions found while recompiling (static_*) keep the context; their callers flush around them.
        return;
    }
    RegGraph g;
    if (!build_reg_graph(context, func, instructions, stats, jtbl_lw, 0, g)) {
        return;
    }
    apply_call_kills(context, g);
    // Caller-saved registers synced at a call: what the callee side reads at entry, and what this function reads
    // after the call (its local is reloaded, so the context must hold it if the callee doesn't change it).
    std::vector<GprMask> genuine_in, genuine_out;
    genuine_liveness(context, g, genuine_in, genuine_out);
    for (size_t id = 0; id < g.nodes.size(); id++) {
        RegNode& node = g.nodes[id];
        if ((node.kind != RegNode::Kind::Call && node.kind != RegNode::Kind::TailCall) || node.call_use == ALL_GPRS) {
            continue;
        }
        bool known = !node.indirect && node.callee >= 0 && (size_t)node.callee < context.regcache_call_sync.size();
        node.call_use |= known ? context.regcache_call_sync[node.callee] : context.regcache_indirect_sync;
        node.call_use |= temps_read_after(node, genuine_out, id);
    }
    // skip_saves: a callee's stack frame no longer needs the caller's callee-saved registers, so a call only syncs
    // those the callee side reads for their value or may return changed; the others can be locals across calls.
    bool skip_saves = context.codegen.skip_saves && !context.regcache_cs_needs.empty();
    GprMask cs_free = skip_saves ? CALLEE_SAVED : 0;
    if (skip_saves) {
        // The SH-4 has seven callee-saved registers and ctx and rb take two: only the most used callee-saved
        // registers whose values live across a call become locals; the rest stay in the context.
        GprMask across = 0;
        for (size_t id = 0; id < g.nodes.size(); id++) {
            const RegNode& node = g.nodes[id];
            if (node.kind == RegNode::Kind::Call && node.call_def != ALL_GPRS) {
                across |= genuine_out[id] & CALLEE_SAVED;
            }
        }
        if (__builtin_popcount(across) > context.codegen.skip_saves_budget) {
            std::vector<std::pair<int, int>> ranked;
            for (int reg = 16; reg < 32; reg++) {
                if (across & gpr_bit(reg)) {
                    int uses = 0;
                    for (const RegNode& node : g.nodes) {
                        uses += node.kind == RegNode::Kind::Plain && ((node.use | node.def) & gpr_bit(reg)) ? 1 : 0;
                    }
                    ranked.emplace_back(-uses, reg);
                }
            }
            std::sort(ranked.begin(), ranked.end());
            for (size_t k = (size_t)context.codegen.skip_saves_budget; k < ranked.size(); k++) {
                cs_free &= ~gpr_bit(ranked[k].second);
            }
        }
        for (RegNode& node : g.nodes) {
            if ((node.kind != RegNode::Kind::Call && node.kind != RegNode::Kind::TailCall) || node.call_use == ALL_GPRS) {
                continue;
            }
            GprMask needs = cs_call_value(context, node, context.regcache_cs_needs, context.regcache_indirect_cs_needs);
            GprMask clobber = cs_call_value(context, node, context.regcache_cs_clobber, context.regcache_indirect_cs_clobber);
            node.call_use = (node.call_use & ~CALLEE_SAVED) | needs | clobber;
            node.call_def |= clobber;
        }
    }
    if (skip_saves && context.reg_summaries[func_index].preserves) {
        SaveSlots slots;
        find_saves(g, instructions, slots);
        size_t n = g.nodes.size();
        for (int reg = 16; reg < 32; reg++) {
            GprMask bit = gpr_bit(reg);
            if (!(slots.regs & bit)) {
                continue;
            }
            const auto& saves = slots.saves[reg];
            const auto& restores = slots.restores[reg];
            auto listed = [](const std::vector<size_t>& v, size_t instr) { return std::find(v.begin(), v.end(), instr) != v.end(); };
            char form = 'n';
            if ((slots.other_defs & bit) && !(cs_free & bit)) {
                // Left in the context, where its writes go straight away.
                form = 'l';
            }
            else if (slots.other_defs & bit) {
                // The context keeps the entry value up to a restore unless a call or barrier on the way writes the
                // local back.
                std::vector<bool> reach(n, false), after(n, false);
                std::vector<int> work{ 0 };
                reach[0] = true;
                while (!work.empty()) {
                    int k = work.back();
                    work.pop_back();
                    for (int s : g.nodes[k].succs) {
                        if (!reach[s]) {
                            reach[s] = true;
                            work.push_back(s);
                        }
                    }
                }
                for (size_t k = 0; k < n; k++) {
                    const RegNode& node = g.nodes[k];
                    bool call = node.kind == RegNode::Kind::Call || node.kind == RegNode::Kind::TailCall || node.kind == RegNode::Kind::Barrier;
                    if (reach[k] && call && (node.call_use & ~node.call_keep & bit)) {
                        for (int s : node.succs) {
                            if (!after[s]) {
                                after[s] = true;
                                work.push_back(s);
                            }
                        }
                    }
                }
                while (!work.empty()) {
                    int k = work.back();
                    work.pop_back();
                    for (int s : g.nodes[k].succs) {
                        if (!after[s]) {
                            after[s] = true;
                            work.push_back(s);
                        }
                    }
                }
                form = 'c';
                for (size_t k = 0; k < n; k++) {
                    const RegNode& node = g.nodes[k];
                    if (after[k] && node.kind == RegNode::Kind::Plain && node.instr != SIZE_MAX && listed(restores, node.instr)) {
                        form = 'l';
                    }
                }
            }
            for (RegNode& node : g.nodes) {
                if (node.kind != RegNode::Kind::Plain || node.instr == SIZE_MAX) {
                    continue;
                }
                if (listed(saves, node.instr) && form != 'l') {
                    node.use &= ~bit;
                }
                if (listed(restores, node.instr)) {
                    if (form == 'n') {
                        node.def &= ~bit;
                    }
                    else if (form == 'c') {
                        node.clean |= bit;
                    }
                }
            }
            for (size_t instr : saves) {
                plan.save_ops[instr] = { reg, form, false };
            }
            for (size_t instr : restores) {
                plan.save_ops[instr] = { reg, form, true };
            }
            if (form == 'l') {
                plan.save_locals |= bit;
            }
        }
    }
    GprMask exit_regs = EXIT_REGS | (func_index < context.regcache_exit_sync.size() ? context.regcache_exit_sync[func_index] : 0);
    if (context.codegen.ret_trim && func_index < context.regcache_ret_read.size()) {
        exit_regs &= ~(RETURN_REGS & ~context.regcache_ret_read[func_index]);
    }
    RegFlow f;
    solve_reg_flow(g, exit_regs, f);
    plan.locals = true;
    plan.poison = false;
    // Registers whose value has to survive a call stay in the context.
    GprMask cross_call = 0;
    GprMask used = 0;
    for (size_t id = 0; id < g.nodes.size(); id++) {
        const RegNode& node = g.nodes[id];
        used |= node.use | node.def;
        if (f.reached[id] && node.kind == RegNode::Kind::Call) {
            cross_call |= f.live_out[id] & ~protocol_kill(node) & ~cs_free;
        }
    }
    GprMask cached = used & ~cross_call;
    for (const auto& [instr, op] : plan.save_ops) {
        if (op.form == 'c' && !(cached & gpr_bit(op.reg))) {
            // A restore from the context needs the register's writes to stay in the local.
            plan = N64Recomp::FunctionRegPlan{};
            return;
        }
    }
    plan.cached = cached;
    plan.entry_load = f.live_in[0] & cached;
    for (size_t id = 0; id < g.nodes.size(); id++) {
        const RegNode& node = g.nodes[id];
        if (!f.reached[id]) {
            continue;
        }
        switch (node.kind) {
            case RegNode::Kind::Call:
            case RegNode::Kind::TailCall:
            case RegNode::Kind::Barrier: {
                auto& sync = plan.calls[node.key];
                sync.writeback |= f.writeback[id] & cached;
                if (node.kind == RegNode::Kind::Call) {
                    sync.reload |= f.live_out[id] & node.call_def & cached & ~node.call_keep;
                }
                break;
            }
            case RegNode::Kind::Exit:
                if (node.key == SIZE_MAX) {
                    plan.end_writeback |= f.writeback[id] & cached;
                }
                else {
                    plan.exits[node.key] |= f.writeback[id] & cached;
                }
                break;
            default:
                break;
        }
    }
    plan.declared = cached;
    auto hook0 = func.function_hooks.find(0);
    plan.hoist_entry_hook = hook0 != func.function_hooks.end();
    if (plan.hoist_entry_hook) {
        // A loop back to the first instruction would run its hook again; leave such a function on the context.
        bool loops_to_start = false;
        for (const auto& instr : instructions) {
            if ((instr.isBranch() || instr.getUniqueId() == InstrId::cpu_j) && (uint32_t)instr.getBranchVramGeneric() == func.vram) {
                loops_to_start = true;
            }
        }
        for (const auto& jtbl : stats.jump_tables) {
            for (uint32_t entry : jtbl.entries) {
                loops_to_start |= entry == func.vram;
            }
        }
        if (loops_to_start) {
            plan = N64Recomp::FunctionRegPlan{};
            plan.poison = context.codegen.abi_poison;
        }
    }
    if (plan.locals && context.codegen.fprcache) {
        plan_fprs(g, instructions, plan);
        if (context.codegen.fpr_abi_sync && !context.regcache_fpr_needs.empty()) {
            trim_fpr_syncs(context, func_index, g, instructions, plan);
        }
    }
    plan.leaf_inline = plan.locals && !plan.hoist_entry_hook && g.calls == 0 && func.function_hooks.empty() &&
                       instructions.size() <= (size_t)context.codegen.inline_leaves;
}

template <typename GeneratorType>
bool recompile_function_impl(GeneratorType& generator, const N64Recomp::Context& context, size_t func_index, std::ostream& output_file, std::span<std::vector<uint32_t>> static_funcs_out, bool tag_reference_relocs) {
    const N64Recomp::Function& orig_func = context.functions[func_index];
    //fmt::print("Recompiling {}\n", func.name);
    std::vector<rabbitizer::InstructionCpu> instructions;
    N64Recomp::FunctionRegPlan reg_plan{};
    std::optional<N64Recomp::Function> hoisted_func;

    // Disassemble and analyze before emitting anything, so the register plan is known up front.
    N64Recomp::FunctionStats stats{};
    std::unordered_set<uint32_t> jtbl_lw_instructions{};
    if (!orig_func.stubbed) {
        if (!load_function_view(context, orig_func, instructions, stats, jtbl_lw_instructions)) {
            fmt::print(stderr, "Failed to analyze {}\n", orig_func.name);
            output_file.clear();
            return false;
        }
        plan_registers(context, func_index, instructions, stats, jtbl_lw_instructions, reg_plan);
        if (reg_plan.hoist_entry_hook) {
            // The first instruction's hook runs before the entry loads, as one at the function's start.
            hoisted_func = orig_func;
            auto& hooks = hoisted_func->function_hooks;
            std::string text = hooks[0];
            hooks.erase(0);
            hooks[-1] = hooks.contains(-1) ? hooks[-1] + "\n    " + text : text;
        }
    }
    else if (context.codegen.abi_poison) {
        reg_plan.poison = true;
    }
    const N64Recomp::Function& func = hoisted_func ? *hoisted_func : orig_func;
    generator.set_reg_plan(&reg_plan);

    generator.emit_function_start(func.name, func_index);

    if (context.trace_mode) {
        fmt::print(output_file,
            "    TRACE_ENTRY()\n",
            func.name);
    }

    // Skip analysis and recompilation of this function is stubbed.
    if (!func.stubbed) {
        // Use a set to sort and deduplicate labels
        std::set<uint32_t> branch_labels;

        auto hook_find = func.function_hooks.find(-1);
        if (hook_find != func.function_hooks.end()) {
            fmt::print(output_file, "    {}\n", hook_find->second);
        }
        generator.emit_function_prologue();

        // First pass, collect branch labels
        for (const auto& instr : instructions) {
            // If this is a branch or a direct jump, add it to the local label list
            if (instr.isBranch() || instr.getUniqueId() == rabbitizer::InstrId::UniqueId::cpu_j) {
                branch_labels.insert((uint32_t)instr.getBranchVramGeneric());
            }
        }

        // Add jump table labels into function
        for (const auto& jtbl : stats.jump_tables) {
            for (uint32_t jtbl_entry : jtbl.entries) {
                branch_labels.insert(jtbl_entry);
            }
        }
        uint32_t vram = func.vram;

        // Second pass, emit code for each instruction and emit labels
        auto cur_label = branch_labels.cbegin();
        vram = func.vram;
        int num_link_branches = 0;
        int num_likely_branches = 0;
        bool needs_link_branch = false;
        bool in_likely_delay_slot = false;
        const auto& section = context.sections[func.section_index];
        size_t reloc_index = 0;
        for (size_t instr_index = 0; instr_index < instructions.size(); ++instr_index) {
            bool had_link_branch = needs_link_branch;
            bool is_branch_likely = false;
            // If we're in the delay slot of a likely instruction, emit a goto to skip the instruction before any labels
            if (in_likely_delay_slot) {
                generator.emit_goto(fmt::format("skip_{}", num_likely_branches));
            }
            // If there are any other branch labels to insert and we're at the next one, insert it
            if (cur_label != branch_labels.end() && vram >= *cur_label) {
                generator.emit_label(fmt::format("L_{:08X}", *cur_label));
                ++cur_label;
            }

            // Advance the reloc index until we reach the last one or until we get to/pass the current instruction
            while ((reloc_index + 1) < section.relocs.size() && section.relocs[reloc_index].address < vram) {
                reloc_index++;
            }

            // Process the current instruction and check for errors
            if (process_instruction(generator, context, func, func_index, stats, jtbl_lw_instructions, instr_index, instructions, output_file, false, needs_link_branch, num_link_branches, reloc_index, needs_link_branch, is_branch_likely, tag_reference_relocs, static_funcs_out) == false) {
                fmt::print(stderr, "Error in recompiling {}, clearing output file\n", func.name);
                output_file.clear();
                return false;
            }
            // If a link return branch was generated, advance the number of link return branches
            if (had_link_branch) {
                num_link_branches++;
            }
            // Now that the instruction has been processed, emit a skip label for the likely branch if needed
            if (in_likely_delay_slot) {
                fmt::print(output_file, "    ");
                generator.emit_label(fmt::format("skip_{}", num_likely_branches));
                num_likely_branches++;
            }
            // Mark the next instruction as being in a likely delay slot if the 
            in_likely_delay_slot = is_branch_likely;
            // Advance the vram address by the size of one instruction
            vram += 4;
        }
    }

    // Terminate the function
    generator.emit_function_end();
    
    return true;
}

// Wrap the templated function with CGenerator as the template parameter.
bool N64Recomp::recompile_function(const N64Recomp::Context& context, size_t function_index, std::ostream& output_file, std::span<std::vector<uint32_t>> static_funcs_out, bool tag_reference_relocs) {
    CGenerator generator{output_file, context.gpr_32bit, &context.codegen};
    return recompile_function_impl(generator, context, function_index, output_file, static_funcs_out, tag_reference_relocs);
}

bool N64Recomp::recompile_function_custom(Generator& generator, const Context& context, size_t function_index, std::ostream& output_file, std::span<std::vector<uint32_t>> static_funcs_out, bool tag_reference_relocs) {
    return recompile_function_impl(generator, context, function_index, output_file, static_funcs_out, tag_reference_relocs);
}
